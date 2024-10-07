# FLAN-T5 QA Study Assistant

This project is a **question-answering model** based on the **FLAN-T5** architecture. It was created for the **ShiftKey Labs Generative AI certification** program. The model was trained to answer questions based on a given context, like the famous **SQuAD** dataset-style tasks.

## What Does This Model Do?

This model can answer questions based on a passage of text you provide. For example, you can give it a piece of text about the **Eiffel Tower**, and then ask a question like, "When was the Eiffel Tower constructed?" The model will return the correct answer!

## How to Use the Model

Here’s a simple way to use this model for question-answering using **transformers**.

### Steps to Run:

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
