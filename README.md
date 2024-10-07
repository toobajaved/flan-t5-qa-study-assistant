# FLAN-T5 QA Study Assistant

This project is a **question-answering model** based on the **FLAN-T5** architecture. It was created for the **ShiftKey Labs Generative AI certification** program. The model was trained to answer questions based on a given context, like the famous **SQuAD** dataset-style tasks.

## Project Contents

This repository contains 4 files:

1. **README.md** – Provides an overview of the project and usage instructions.
2. **FLAN_T5_QA_Study_Assistant.ipynb** – The original notebook that trains and runs the FLAN-T5 model for question-answering tasks.
3. **Study_Assistant_test.ipynb** – A test notebook that demonstrates how the model performs on several example questions and contexts.
4. **LICENSE** – This project is licensed under the MIT License.


### Check out the model on Hugging Face:
You can view the model and try it out directly on Hugging Face [here](https://huggingface.co/tootooba/flan-t5-qa-study-assistant).


## What Does This Model Do?

This model can answer questions based on a passage of text you provide. For example, you can give it a piece of text about the **Eiffel Tower**, and then ask a question like, "When was the Eiffel Tower constructed?" The model will return the correct answer!

## How to Use the Model

Here’s a simple way to use this model for question-answering using **transformers**, or you can use the **Hugging Face Inference API** directly.

### Option 1: Running the Model Locally with `transformers`

1. **Install the transformers library**:
   - Open your terminal or command prompt.
   - Run this command:
     ```bash
     pip install transformers
     ```

2. **Run the Model**:

   Copy the following Python code and run it in your Python environment or Jupyter Notebook:
   
   ```python
   from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

   # Load the model and tokenizer
   model = AutoModelForSeq2SeqLM.from_pretrained("tootooba/flan-t5-qa-study-assistant").to("cuda")
   tokenizer = AutoTokenizer.from_pretrained("tootooba/flan-t5-qa-study-assistant")

   # Define the context and question
   context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed between 1887 and 1889 as the entrance arch for the 1889 World's Fair."
   question = "When was the Eiffel Tower constructed?"

   # Tokenize inputs and move them to GPU
   inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True).to("cuda")

   # Generate the answer
   outputs = model.generate(inputs.input_ids)
   answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

   print("Answer:", answer)

### Option 2: Using Hugging Face Inference API

You can also use the **Hugging Face Inference API** to run the model without setting it up locally.

1. **Get Your API Token**:
   - Go to [Hugging Face Tokens](https://huggingface.co/settings/ttokens), and create a new token if you don’t have one already.

2. **Run the Model via API**:

   Here’s a simple Python code to send a request to the **Hugging Face Inference API** for the **FLAN-T5 QA Study Assistant**:

   ```python
   import requests

   # Your Hugging Face Model Inference API URL
   API_URL = "https://api-inference.huggingface.co/models/tootooba/flan-t5-qa-study-assistant"
   
   # Replace with your actual Hugging Face API token
   headers = {"Authorization": "Bearer YOUR_HF_API_TOKEN"}

   def query(payload):
       response = requests.post(API_URL, headers=headers, json=payload)
       return response.json()

   # Define the context and question
   context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed between 1887 and 1889 as the entrance arch for the 1889 World's Fair."
   question = "When was the Eiffel Tower constructed?"

   # Send the request and get the result
   result = query({
       "inputs": {
           "question": question,
           "context": context
       }
   })

   # Print the answer
   print("Answer:", result)
