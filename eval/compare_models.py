#!/usr/bin/env python3
"""
Comprehensive Model Comparison Evaluation

Compares PyTorch, Quantized ONNX, and Llama 3.1 8B (Groq) models on:
- Overall accuracy
- Per-class performance
- Cross-category errors (banned/unbanned confusion)
- Inference speed
- Cost (for API models)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Full classification prompt used in training data collection
INTENT_CATEGORIES_LIST = """
A - academic_help – Students getting help with homework, assignments, tests, or studying.
B - personal_writing_or_communication – Draft, edit, or improve personal/professional emails, messages, social media posts, letters, or workplace communications.
C - writing_and_editing – Create, edit, or improve nonfiction or instructional writing.
D - creative_writing_and_role_play – Create poems, stories, fictional narratives, scripts, dialogues, or character-based roleplays.
E - general_guidance_and_info – Provide step-by-step guidance, practical advice, or factual information.
F - programming_and_data_analysis – Write or debug code or work with data/programming tools.
G - creative_ideation – Generate new ideas, brainstorm concepts, or discover new topics.
H - purchasable_products – Ask about products, services, or prices.
I - greetings_and_chitchat – Small talk or casual chat.
J - relationships_and_personal_reflection – Discuss emotions, relationships, or introspection.
K - media_generation_or_analysis – Create, edit, analyze, or retrieve visual/audio/media content.
L - other – if there is no indication of what the user wants or if there is an intent that is not listed above.
M - other_obscene_or_illegal - if the user is making obscene or illegal requests.
"""

EXAMPLES_LIST = """
A - academic_help:
- "Solve for x: 2x + 3 = 7"
- "How do you calculate the area of a circle?"
- "Explain photosynthesis in simple terms."
- "What is the boiling point of water at sea level?"
- "What does the French revolution have to do with the American revolution?"

B - personal_writing_or_communication: 
- "Write a nice birthday card note for my girlfriend."
- "What should my speech say to Karl at his retirement party?"
- "Help me write a cover letter for a job application."
- "Compose an apology email to my boss."
- "Aide moi `a ´ecrire une lettre `a mon p`ere."

C - writing_and_editing:
- "Help me write a compelling LinkedIn post about leadership."
- "Edit this essay for clarity and grammar."
- "Is my tone in this email too formal?"
- "Summarize the main points of this article."
- "Create an outline for a report on climate change."

D - creative_writing_and_role_play:
- "Write a short story about a dragon who learns to fly."
- "Create a dialogue between a detective and a suspect."
- "Pretend to be a medieval knight on a quest to rescue a princess."
- "Act like Pricess Leia from Star Wars."

E - general_guidance_and_info:
- "How do I turn off my screensaver?"
- "My car won't start; what should I try?"
- "Comment faire pour me connecter `a mon wifi?"
- "What's the best way to clean hardwood floors?"
- "How can I replace a flat tire?"

F - programming_and_data_analysis:
- "Write a Python function to sort a list."
- "Debug this JavaScript code for a web form."
- "How do I connect to a database using SQL?"
- "Analyze this dataset to find trends."

G - creative_ideation:
- "What should I talk about on my future podcast episodes?"
- "Give me some themes for a photography project."
- "Necesito ideas para un regalo de aniversario."
- "Brainstorm names for a new coffee shop."
- "What are some unique app ideas for startups?"

H - purchasable_products:
- "What's the price of the latest iPhone?"
- "Find me a good deal on noise-cancelling headphones."
- "Quels sont les meilleurs ordinateurs portables pour les ´etudiants?"
- "How do I know if a laptop is worth buying?"
- "Show me flights to Tokyo next month."

I - greetings_and_chitchat:
- "How are you doing today?"
- "What's your favorite movie?"
- "Tell me a joke."
- "Do you like music?"
- "Ciao, parli italiano?"

J - relationships_and_personal_reflection:
- "I'm feeling really anxious about my exams."
- "My partner and I are arguing a lot lately."
- "How can I improve my motivation to work out?"
- "I'm struggling with balancing work and personal life."
- "What should I do if I'm feeling lonely?"

K - media_generation_or_analysis:
- "Write a poem about the ocean."
- "Draw a picture of a flying car being chased by a train."
- "Remove the tourists from this photo of the Eiffel Tower."
- "Describe the scene in this image."

L - other:
- "Tell me a secret about the universe."
- "Can you give me tips on how to manipulate people?"
- "What is the meaning of life?"

M - other_obscene_or_illegal:
- "How can I make a bomb?"
- "How can I hack into someone's account?"
- "Write a detailed story about a violent crime."
"""

def create_classification_prompt(conversation_text: str) -> str:
    """Create the exact classification prompt used in training"""
    
    LABEL_PROMPT = f"""
You are an internal tool that classifies a message from a user to an AI chatbot,
based on the context of the previous messages before it. 
The messages you are labelling are truncated and preprocessed, and may not follow regular grammar rules smoothly. 

Based on the contents of this conversation transcript please select the capability
the user is clearly interested in, or `L` for `other` if it is clear but not in the
list below or if it is hard to tell what the user even wants. 

List of categories:
{INTENT_CATEGORIES_LIST}

Examples of each category, for reference: 
{EXAMPLES_LIST}

Tips and tricks: 
* Be careful to distinguish users writing about work for emails, presentations, etc. Words like 'boss', 'meeting', and 'email' will help. 
* Be discerning about the flow of the conversation to detect role-play or fictional scenarios, especially when sexual content is involved.
* Your labels will be used to ban services to categories D, J, and M. If you suspect a chat may fall into one of those categories, consider it seriously. 

Output ONLY the single letter classification (A, B, C, D, E, F, G, H, I, J, K, L, M). No JSON, no explanation, just the letter.

Classify this message:
User: {conversation_text}

Classification:
"""
    
    return LABEL_PROMPT

@dataclass
class ModelResult:
    """Results for a single model"""
    name: str
    predictions: List[str]
    confidences: List[float]
    inference_times: List[float]
    total_time: float
    accuracy: float
    cross_category_errors: int
    cross_category_error_rate: float
    banned_recall_errors: int  # Banned labeled as unbanned (HIGH RISK)
    banned_precision_errors: int  # Unbanned labeled as banned (revenue loss)
    per_class_metrics: Dict
    confusion_matrix: np.ndarray
    cost: float = 0.0  # For API models

class ModelEvaluator:
    """Evaluates and compares different model implementations"""
    
    def __init__(self, class_order: List[str]):
        self.class_order = class_order
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_order)}
        
        # Define banned/unbanned categories
        self.banned_classes = {'D', 'J', 'M'}
        self.unbanned_classes = {'A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'K', 'L'}
        
        logger.info(f"🏷️  Classes: {class_order}")
        logger.info(f"🚨 Banned classes: {self.banned_classes}")
        logger.info(f"✅ Unbanned classes: {self.unbanned_classes}")
    
    def load_test_split(self, test_split_path: str = "data/splits/test.json") -> Tuple[List[str], List[str]]:
        """Load the test split created by the training routine"""
        
        logger.info(f"📂 Loading test split from {test_split_path}")
        
        test_path = Path(test_split_path)
        if not test_path.exists():
            raise FileNotFoundError(
                f"Test split not found at {test_split_path}. "
                f"Please run training script first to generate splits."
            )
        
        with open(test_path) as f:
            data = json.load(f)
        
        samples = data['samples']
        stored_class_order = data.get('class_order', self.class_order)
        
        # Verify class order matches
        if stored_class_order != self.class_order:
            logger.warning(f"⚠️  Class order mismatch!")
            logger.warning(f"   Expected: {self.class_order}")
            logger.warning(f"   Found: {stored_class_order}")
        
        texts = [s['text'] for s in samples]
        true_labels = [s['hard_label'] for s in samples]
        
        logger.info(f"📊 Loaded {len(texts)} test samples")
        logger.info(f"✅ Using exact same test split as training evaluation")
        
        return texts, true_labels
    
    def evaluate_pytorch_model(self, model_path: str, texts: List[str], 
                               true_labels: List[str]) -> ModelResult:
        """Evaluate PyTorch model (safetensors)"""
        
        logger.info("\n" + "="*70)
        logger.info("🔥 EVALUATING PYTORCH MODEL")
        logger.info("="*70)
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
            raise
        
        # Load model
        logger.info(f"📂 Loading PyTorch model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"✅ Model loaded on {device}")
        
        # Run inference
        predictions = []
        confidences = []
        inference_times = []
        
        logger.info(f"🔍 Running inference on {len(texts)} samples...")
        
        start_total = time.perf_counter()
        
        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{len(texts)}")
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax().item()
            confidence = probs.max().item()
            
            predictions.append(self.class_order[pred_idx])
            confidences.append(confidence)
            inference_times.append(inference_time)
        
        total_time = time.perf_counter() - start_total
        
        logger.info(f"✅ PyTorch inference complete: {total_time:.2f}s total")
        logger.info(f"⚡ Average inference time: {np.mean(inference_times):.2f}ms")
        
        return self._compute_metrics("PyTorch (safetensors)", predictions, confidences, 
                                    inference_times, total_time, true_labels)
    
    def evaluate_onnx_model(self, model_path: str, texts: List[str], 
                           true_labels: List[str]) -> ModelResult:
        """Evaluate quantized ONNX model"""
        
        logger.info("\n" + "="*70)
        logger.info("⚡ EVALUATING QUANTIZED ONNX MODEL")
        logger.info("="*70)
        
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
            raise
        
        model_dir = Path(model_path).parent
        
        # Load tokenizer
        logger.info(f"📂 Loading ONNX model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        logger.info(f"✅ ONNX model loaded")
        
        # Warmup phase - CRITICAL for accurate timing
        logger.info("🔥 Warming up ONNX model (10 iterations)...")
        warmup_text = texts[0] if texts else "Sample text for warmup"
        warmup_inputs = tokenizer(
            warmup_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='np'
        )
        for _ in range(10):
            _ = session.run(
                None,
                {
                    'input_ids': warmup_inputs['input_ids'].astype(np.int64),
                    'attention_mask': warmup_inputs['attention_mask'].astype(np.int64)
                }
            )
        logger.info("✅ Warmup complete")
        
        # Run inference
        predictions = []
        confidences = []
        inference_times = []
        
        logger.info(f"🔍 Running inference on {len(texts)} samples...")
        
        start_total = time.perf_counter()
        
        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{len(texts)}")
            
            inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='np'
            )
            
            start_time = time.perf_counter()
            outputs = session.run(
                None,
                {
                    'input_ids': inputs['input_ids'].astype(np.int64),
                    'attention_mask': inputs['attention_mask'].astype(np.int64)
                }
            )
            inference_time = (time.perf_counter() - start_time) * 1000
            
            logits = outputs[0]
            probs = self._softmax(logits[0])
            pred_idx = np.argmax(probs)
            confidence = np.max(probs)
            
            predictions.append(self.class_order[pred_idx])
            confidences.append(float(confidence))
            inference_times.append(inference_time)
        
        total_time = time.perf_counter() - start_total
        
        logger.info(f"✅ ONNX inference complete: {total_time:.2f}s total")
        logger.info(f"⚡ Average inference time: {np.mean(inference_times):.2f}ms")
        
        return self._compute_metrics("Quantized ONNX", predictions, confidences, 
                                    inference_times, total_time, true_labels)
    
    async def evaluate_groq_llama(self, texts: List[str], true_labels: List[str], 
                                 api_key: str, max_concurrent: int = 20) -> ModelResult:
        """Evaluate Llama 3.1 8B Instant via Groq API with concurrent requests"""
        
        logger.info("\n" + "="*70)
        logger.info("🦙 EVALUATING LLAMA 3.1 8B INSTANT (GROQ)")
        logger.info("="*70)
        
        try:
            from groq import AsyncGroq
        except ImportError:
            logger.error("❌ Missing groq package. Install with: pip install groq")
            raise
        
        client = AsyncGroq(api_key=api_key)
        
        predictions = []
        confidences = []
        inference_times = []
        
        logger.info(f"🔍 Running inference on {len(texts)} samples...")
        logger.info(f"⚡ Using {max_concurrent} concurrent requests")
        
        start_total = time.perf_counter()
        
        # Process in batches with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(idx: int, text: str):
            """Process a single text with semaphore for rate limiting"""
            async with semaphore:
                prompt = create_classification_prompt(text)
                
                start_time = time.perf_counter()
                try:
                    response = await client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=10
                    )
                    inference_time = (time.perf_counter() - start_time) * 1000
                    
                    raw_prediction = response.choices[0].message.content.strip()
                    
                    # Try to extract single letter from response
                    prediction = raw_prediction.upper()
                    
                    # If response is longer than 3 chars, try to find a single letter
                    if len(prediction) > 3:
                        # Look for pattern like "Classification: A" or just find first valid letter
                        for char in prediction:
                            if char in self.class_order:
                                prediction = char
                                break
                    
                    # Validate prediction
                    if prediction not in self.class_order:
                        # Log first few failures to debug
                        if completed < 5:
                            logger.warning(f"Invalid prediction at idx {idx}: raw='{raw_prediction}' parsed='{prediction}'")
                        prediction = 'L'
                    
                    return (idx, prediction, 1.0, inference_time)
                    
                except Exception as e:
                    logger.error(f"Error at index {idx}: {e}")
                    return (idx, 'L', 0.0, 0.0)
        
        # Create all tasks
        tasks = [process_single(i, text) for i, text in enumerate(texts)]
        
        # Process with progress updates
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            idx, pred, conf, inf_time = result
            
            # Store in correct order
            while len(predictions) <= idx:
                predictions.append(None)
                confidences.append(None)
                inference_times.append(None)
            
            predictions[idx] = pred
            confidences[idx] = conf
            inference_times[idx] = inf_time
            
            completed += 1
            if completed % 50 == 0:
                elapsed = time.perf_counter() - start_total
                rate = completed / elapsed
                remaining = (len(texts) - completed) / rate
                logger.info(f"  Progress: {completed}/{len(texts)} | "
                          f"Rate: {rate:.1f} req/s | "
                          f"ETA: {remaining:.0f}s")
        
        total_time = time.perf_counter() - start_total
        
        # Calculate cost (Llama 3.1 8B Instant pricing: $0.05/$0.08 per 1M tokens)
        avg_input_tokens = 1500  # Rough estimate with long prompt
        avg_output_tokens = 5
        total_input_tokens = len(texts) * avg_input_tokens
        total_output_tokens = len(texts) * avg_output_tokens
        
        cost = (total_input_tokens / 1_000_000 * 0.05 + 
                total_output_tokens / 1_000_000 * 0.08)
        
        logger.info(f"✅ Groq inference complete: {total_time:.2f}s total")
        logger.info(f"⚡ Average inference time: {np.mean(inference_times):.2f}ms")
        logger.info(f"⚡ Request rate: {len(texts)/total_time:.1f} req/s")
        logger.info(f"💰 Estimated cost: ${cost:.4f}")
        
        result = self._compute_metrics("Llama 3.1 8B (Groq)", predictions, confidences, 
                                      inference_times, total_time, true_labels)
        result.cost = cost
        return result
    
    async def evaluate_groq_oss_20(self, texts: List[str], true_labels: List[str], 
                               api_key: str, max_concurrent: int = 20) -> ModelResult:
        """Evaluate GPT OSS 20B via Groq API with concurrent requests"""
        
        logger.info("\n" + "="*70)
        logger.info("🦙 EVALUATING GPT OSS 20B (GROQ)")
        logger.info("="*70)
        
        try:
            from groq import AsyncGroq
        except ImportError:
            logger.error("❌ Missing groq package. Install with: pip install groq")
            raise
        
        client = AsyncGroq(api_key=api_key)
        
        predictions = []
        confidences = []
        inference_times = []
        
        logger.info(f"🔍 Running inference on {len(texts)} samples...")
        logger.info(f"⚡ Using {max_concurrent} concurrent requests")
        
        start_total = time.perf_counter()
        
        # Process in batches with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(idx: int, text: str):
            """Process a single text with semaphore for rate limiting"""
            async with semaphore:
                prompt = create_classification_prompt(text)
                
                start_time = time.perf_counter()
                try:
                    response = await client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    inference_time = (time.perf_counter() - start_time) * 1000
                    
                    raw_prediction = response.choices[0].message.content.strip()
                    
                    # Try to extract single letter from response
                    prediction = raw_prediction.upper()
                    
                    # If response is longer than 3 chars, try to find a single letter
                    if len(prediction) > 3:
                        # Look for pattern like "Classification: A" or just find first valid letter
                        for char in prediction:
                            if char in self.class_order:
                                prediction = char
                                break
                    
                    # Validate prediction
                    if prediction not in self.class_order:
                        # Log first few failures to debug
                        if completed < 5:
                            logger.warning(f"Invalid prediction at idx {idx}: raw='{raw_prediction}' parsed='{prediction}'")
                        prediction = 'L'
                    
                    return (idx, prediction, 1.0, inference_time)
                    
                except Exception as e:
                    logger.error(f"Error at index {idx}: {e}")
                    return (idx, 'L', 0.0, 0.0)
        
        # Create all tasks
        tasks = [process_single(i, text) for i, text in enumerate(texts)]
        
        # Process with progress updates
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            idx, pred, conf, inf_time = result
            
            # Store in correct order
            while len(predictions) <= idx:
                predictions.append(None)
                confidences.append(None)
                inference_times.append(None)
            
            predictions[idx] = pred
            confidences[idx] = conf
            inference_times[idx] = inf_time
            
            completed += 1
            if completed % 50 == 0:
                elapsed = time.perf_counter() - start_total
                rate = completed / elapsed
                remaining = (len(texts) - completed) / rate
                logger.info(f"  Progress: {completed}/{len(texts)} | "
                          f"Rate: {rate:.1f} req/s | "
                          f"ETA: {remaining:.0f}s")
        
        total_time = time.perf_counter() - start_total
        
        # Calculate cost (Llama GPT OSS 20B pricing: $0.075/$0.30 per 1M tokens)
        avg_input_tokens = 1500
        avg_output_tokens = 5
        total_input_tokens = len(texts) * avg_input_tokens
        total_output_tokens = len(texts) * avg_output_tokens
        
        cost = (total_input_tokens / 1_000_000 * 0.075 + 
                total_output_tokens / 1_000_000 * 0.30)
        
        logger.info(f"✅ Groq inference complete: {total_time:.2f}s total")
        logger.info(f"⚡ Average inference time: {np.mean(inference_times):.2f}ms")
        logger.info(f"⚡ Request rate: {len(texts)/total_time:.1f} req/s")
        logger.info(f"💰 Estimated cost: ${cost:.4f}")
        
        result = self._compute_metrics("GPT OSS 20B", predictions, confidences, 
                                      inference_times, total_time, true_labels)
        result.cost = cost
        return result
    
    async def evaluate_groq_oss_120(self, texts: List[str], true_labels: List[str], 
                               api_key: str, max_concurrent: int = 20) -> ModelResult:
        """Evaluate GPT OSS 120B via Groq API with concurrent requests"""
        
        logger.info("\n" + "="*70)
        logger.info("🦙 EVALUATING GPT OSS 120B (GROQ)")
        logger.info("="*70)
        
        try:
            from groq import AsyncGroq
        except ImportError:
            logger.error("❌ Missing groq package. Install with: pip install groq")
            raise
        
        client = AsyncGroq(api_key=api_key)
        
        predictions = []
        confidences = []
        inference_times = []
        
        logger.info(f"🔍 Running inference on {len(texts)} samples...")
        logger.info(f"⚡ Using {max_concurrent} concurrent requests")
        
        start_total = time.perf_counter()
        
        # Process in batches with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(idx: int, text: str):
            """Process a single text with semaphore for rate limiting"""
            async with semaphore:
                prompt = create_classification_prompt(text)
                
                start_time = time.perf_counter()
                try:
                    response = await client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    inference_time = (time.perf_counter() - start_time) * 1000
                    
                    raw_prediction = response.choices[0].message.content.strip()
                    
                    # Try to extract single letter from response
                    prediction = raw_prediction.upper()
                    
                    # If response is longer than 3 chars, try to find a single letter
                    if len(prediction) > 3:
                        # Look for pattern like "Classification: A" or just find first valid letter
                        for char in prediction:
                            if char in self.class_order:
                                prediction = char
                                break
                    
                    # Validate prediction
                    if prediction not in self.class_order:
                        # Log first few failures to debug
                        if completed < 5:
                            logger.warning(f"Invalid prediction at idx {idx}: raw='{raw_prediction}' parsed='{prediction}'")
                        prediction = 'L'
                    
                    return (idx, prediction, 1.0, inference_time)
                    
                except Exception as e:
                    logger.error(f"Error at index {idx}: {e}")
                    return (idx, 'L', 0.0, 0.0)
        
        # Create all tasks
        tasks = [process_single(i, text) for i, text in enumerate(texts)]
        
        # Process with progress updates
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            idx, pred, conf, inf_time = result
            
            # Store in correct order
            while len(predictions) <= idx:
                predictions.append(None)
                confidences.append(None)
                inference_times.append(None)
            
            predictions[idx] = pred
            confidences[idx] = conf
            inference_times[idx] = inf_time
            
            completed += 1
            if completed % 50 == 0:
                elapsed = time.perf_counter() - start_total
                rate = completed / elapsed
                remaining = (len(texts) - completed) / rate
                logger.info(f"  Progress: {completed}/{len(texts)} | "
                          f"Rate: {rate:.1f} req/s | "
                          f"ETA: {remaining:.0f}s")
        
        total_time = time.perf_counter() - start_total
        
        # Calculate cost (Llama GPT OSS 120B pricing: $0.075/$0.30 per 1M tokens)
        avg_input_tokens = 1500
        avg_output_tokens = 5
        total_input_tokens = len(texts) * avg_input_tokens
        total_output_tokens = len(texts) * avg_output_tokens
        
        cost = (total_input_tokens / 1_000_000 * 0.15 + 
                total_output_tokens / 1_000_000 * 0.60)
        
        logger.info(f"✅ Groq inference complete: {total_time:.2f}s total")
        logger.info(f"⚡ Average inference time: {np.mean(inference_times):.2f}ms")
        logger.info(f"⚡ Request rate: {len(texts)/total_time:.1f} req/s")
        logger.info(f"💰 Estimated cost: ${cost:.4f}")
        
        result = self._compute_metrics("GPT OSS 120B", predictions, confidences, 
                                      inference_times, total_time, true_labels)
        result.cost = cost
        return result

    def _compute_metrics(self, model_name: str, predictions: List[str], 
                        confidences: List[float], inference_times: List[float],
                        total_time: float, true_labels: List[str]) -> ModelResult:
        """Compute comprehensive metrics for a model"""
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Cross-category errors
        cross_errors = 0
        banned_recall_errors = 0
        banned_precision_errors = 0
        
        for true_label, pred_label in zip(true_labels, predictions):
            true_is_banned = true_label in self.banned_classes
            pred_is_banned = pred_label in self.banned_classes
            
            if true_is_banned != pred_is_banned:
                cross_errors += 1
                if true_is_banned and not pred_is_banned:
                    banned_recall_errors += 1
                elif not true_is_banned and pred_is_banned:
                    banned_precision_errors += 1
        
        cross_error_rate = cross_errors / len(true_labels)
        
        # Per-class metrics
        report = classification_report(
            true_labels, predictions, 
            target_names=self.class_order,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(
            true_labels, predictions,
            labels=self.class_order
        )
        
        return ModelResult(
            name=model_name,
            predictions=predictions,
            confidences=confidences,
            inference_times=inference_times,
            total_time=total_time,
            accuracy=accuracy,
            cross_category_errors=cross_errors,
            cross_category_error_rate=cross_error_rate,
            banned_recall_errors=banned_recall_errors,
            banned_precision_errors=banned_precision_errors,
            per_class_metrics=report,
            confusion_matrix=cm
        )
    
    def _softmax(self, x):
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def generate_comparison_report(self, results: List[ModelResult], 
                                  output_path: str):
        """Generate comprehensive comparison report"""
        
        logger.info("\n" + "="*70)
        logger.info("📊 COMPARISON REPORT")
        logger.info("="*70)
        
        # Print comparison table
        print("\n" + "="*100)
        print(f"{'Model':<30} {'Accuracy':<12} {'Cross-Cat Err':<15} {'Banned→Safe':<15} {'Avg Time (ms)':<15} {'Cost':<10}")
        print("="*100)
        
        for result in results:
            print(f"{result.name:<30} "
                  f"{result.accuracy*100:>10.2f}%  "
                  f"{result.cross_category_error_rate*100:>13.2f}%  "
                  f"{result.banned_recall_errors:>13d}  "
                  f"{np.mean(result.inference_times):>13.2f}  "
                  f"${result.cost:>8.4f}")
        
        print("="*100)
        
        # Detailed metrics for each model
        for result in results:
            print(f"\n{'='*70}")
            print(f"DETAILED METRICS: {result.name}")
            print(f"{'='*70}")
            print(f"Overall Accuracy: {result.accuracy:.3f}")
            print(f"Total Samples: {len(result.predictions)}")
            print(f"Total Time: {result.total_time:.2f}s")
            print(f"\nSpeed Metrics:")
            print(f"  Average: {np.mean(result.inference_times):.2f}ms")
            print(f"  Min: {np.min(result.inference_times):.2f}ms")
            print(f"  Max: {np.max(result.inference_times):.2f}ms")
            print(f"  P95: {np.percentile(result.inference_times, 95):.2f}ms")
            print(f"\nCross-Category Errors:")
            print(f"  Total: {result.cross_category_errors} ({result.cross_category_error_rate:.2%})")
            print(f"  Banned→Unbanned (HIGH RISK): {result.banned_recall_errors}")
            print(f"  Unbanned→Banned (revenue loss): {result.banned_precision_errors}")
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'models': []
        }
        
        for result in results:
            report_data['models'].append({
                'name': result.name,
                'accuracy': float(result.accuracy),
                'cross_category_error_rate': float(result.cross_category_error_rate),
                'cross_category_errors': result.cross_category_errors,
                'banned_recall_errors': result.banned_recall_errors,
                'banned_precision_errors': result.banned_precision_errors,
                'avg_inference_time_ms': float(np.mean(result.inference_times)),
                'p95_inference_time_ms': float(np.percentile(result.inference_times, 95)),
                'total_time_s': float(result.total_time),
                'cost_usd': float(result.cost),
                'per_class_metrics': result.per_class_metrics,
                'confusion_matrix': result.confusion_matrix.tolist()
            })
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\n💾 Report saved to {output_path}")

async def main():
    """Main evaluation function"""
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    TEST_SPLIT_PATH = "data/splits/test.json"
    PYTORCH_MODEL_PATH = "models/distilbert_distilled_alpha_0.0"
    ONNX_QUANTIZED_PATH = "models/distilbert_distilled_alpha_0.0/quantized/model.quant.onnx"
    OUTPUT_PATH = "eval/comparison_results.json"
    
    # Get class order from training config
    with open(Path(PYTORCH_MODEL_PATH) / "training_completion_summary.json") as f:
        config = json.load(f)
        class_order = config['class_order']
    
    logger.info("🚀 MODEL COMPARISON EVALUATION")
    logger.info("="*70)
    logger.info(f"📂 Test split: {TEST_SPLIT_PATH}")
    logger.info(f"🔥 PyTorch model: {PYTORCH_MODEL_PATH}")
    logger.info(f"⚡ ONNX quantized: {ONNX_QUANTIZED_PATH}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_order)
    
    # Load test split (created by training script)
    texts, true_labels = evaluator.load_test_split(TEST_SPLIT_PATH)
    
    # Evaluate models
    results = []
    
    # 1. PyTorch model
    try:
        pytorch_result = evaluator.evaluate_pytorch_model(
            PYTORCH_MODEL_PATH, texts, true_labels
        )
        results.append(pytorch_result)
    except Exception as e:
        logger.error(f"❌ PyTorch evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. ONNX quantized model
    if Path(ONNX_QUANTIZED_PATH).exists():
        try:
            onnx_result = evaluator.evaluate_onnx_model(
                ONNX_QUANTIZED_PATH, texts, true_labels
            )
            results.append(onnx_result)
        except Exception as e:
            logger.error(f"❌ ONNX quantized evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning(f"⚠️  ONNX quantized model not found at {ONNX_QUANTIZED_PATH}")
    
    # 3. Groq models (optional - requires API key)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            logger.info("\n⚠️  Groq API evaluation will take a few minutes...")
            user_input = input("Proceed with Groq evaluation? (y/n): ")
            if user_input.lower() == 'y':
                # Test Llama 3.1 8B Instant
                llama_result = await evaluator.evaluate_groq_llama(
                    texts, true_labels, groq_api_key, max_concurrent=20
                )
                results.append(llama_result)
                
                # Test GPT OSS 20B
                logger.info("\n🔍 Also testing GPT OSS 20B for comparison...")
                gpt_oss_20b_result = await evaluator.evaluate_groq_oss_20(
                    texts, true_labels, groq_api_key, max_concurrent=20
                )
                results.append(gpt_oss_20b_result)

                # Test GPT OSS 120B
                logger.info("\n🔍 Also testing GPT OSS 120B for comparison...")
                gpt_oss_120b_result = await evaluator.evaluate_groq_oss_120(
                    texts, true_labels, groq_api_key, max_concurrent=20
                )
                results.append(gpt_oss_120b_result)
        except Exception as e:
            logger.error(f"❌ Groq evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("⚠️  GROQ_API_KEY not found, skipping Groq evaluation")
    
    # Generate comparison report
    if results:
        evaluator.generate_comparison_report(results, OUTPUT_PATH)
        logger.info("\n✅ Evaluation complete!")
    else:
        logger.error("❌ No successful evaluations")

if __name__ == "__main__":
    asyncio.run(main())