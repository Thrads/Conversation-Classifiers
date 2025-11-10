# ***Thrad.ai Conversation Classifiers GitHub***

A repo with background and guidance for working with the open-source `thrad-distilbert-conversation-classifier` and `thrad-bert-conversation-classifier` models from HuggingFace [[LINK HERE](https://huggingface.co/Thrad)]! 

Our models are trained to classify the content of user LLM chats, with special attention payed to protecting businesses from interacting with chats deemed innapropriate. They outperform LLMs 100-200X their size by parameter count by at least 1/3, and can run inference in <50ms on everyday hardware. 

To contribute to the project, create an issue in this repository, or contact Scott Biggs or Marco Visentin at Thrad.ai. We encourage anyone interested to use our models to help make ads-in-AI and the AI ecosystem at large safer for businesses and more private for users. 

For more technical breakdown of the design, training, and evaluation process, refer to the technical blog pasted on our HuggingFace page here [[LINK HERE](https://huggingface.co/Thrad)].

If you use this model in your business or a research project, please cite us. 

# Performance and Cost: 
Costs calculated based on the rates on 8 Nov 2025. 
```bash 
DistilBERT, N = 2224
================================================================================================
Model                          Accuracy     Cross-Cat Err           Banned→Safe         Cost      
================================================================================================
PyTorch (safetensors)           83.77%           5.17%                     67         $  0.0000
Quantized ONNX                  83.81%           5.17%                     75         $  0.0000
Llama 3.1 8B (Groq)             40.65%          14.43%                    289         $  0.1677
GPT OSS 20B (Groq)              35.03%          17.67%                    387         $  0.2535
GPT OSS 120B (Groq)             60.52%          10.79%                    231         $  0.5071
=================================================================================================
```
You can review a more detailed breakdown of these stats in `eval/distilBERT_results.json`. Statistics are calculated on a held out test dataset of 2224 samples. 

```bash
BERT, N = 2224
====================================================================================================
Model                              Accuracy     Cross-Cat Err     Banned→Safe        Cost      
====================================================================================================
PyTorch (safetensors)               74.37%           7.73%             87          $  0.0000
Llama 3.1 8B (Groq)                 41.91%          14.39%            282          $  0.1677
GPT OSS 20B                         31.21%          17.67%            385          $  0.2535
GPT OSS 120B                        58.81%          11.02%            232          $  0.5071
====================================================================================================
```
You can review a more detailed breakdown of these stats in `eval/BERT_results.json`. Statistics are calculated on a held out test dataset of 2224 samples. 


# Categories: 

```
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
```

# Message Cleaning: 

Beginning with a ```records.json``` in the data folder. Example of structure: 
```json    
    "chat_id": "sample",
    "context": "sample",
    "query": "sample",
    "messages": [
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "user",
        "content": "stuff"
      },
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "assistant",
        "content": "stuff"
      },
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "user",
        "content": "stuff"
      },
      {
        "id": "sample",
        "createdAt": "sample",
        "role": "assistant",
        "content": "stuff"
      }
    ],
    "formatted_chat": "User: stuff"
```

# Extracting chat content: 

```bash 
python src/extract_sequences.py
```

```bash 
python src/preprocess_sequences.py
```
Very simple extraction scripts. 

# Labelling
Get Gemini 2.5 Flash and GPT 4o mini lables (can be run in parallel with a simple nohup .sh script):
```bash
python src/get_gemini_flash_labels.py
```

```bash 
python src/get_gpt_4o_mini_labels.py
```

To verify labels with gemini flash run (will save a NEW json of only AGREED on samples). 

```bash 
python src/compare_and_filter.py
```
Provides some reporting/visibility on agreement rates and class breakdowns in 'true' data. 

# Training: 

For training distilBERTs and BERTs, run: 
```bash
python src/train_distilBERT.py
```
You can modify the base model you train by changing the `model_name` string in the config. To use soft labels, modify `alpha` in your .env and the `DistillationConfig` class. 


# Export as ONNX, in case autosave fails:

**DistilBERT**
```bash
python src/export_to_onnx.py --model-path models/distilbert_512_intent_classifier --output-path models/distilbert_512_intent_classifier_onnx --max-length 512 --test
```

# Evaluation: 
Runs on the `data/splits/test.json` generated by the training routine to enforce data seperation. 

```bash
python eval/compare_models.py 
```

# Citation: 
```bash 
@misc{thrad-convo-classifier,
    title={Thrad Conversation Classifiers}, 
    author={Scott Biggs, Marco Visentin},
    year={2025}
}
```
