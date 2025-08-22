from transformers import pipeline

# Use Apache 2.0 licensed model
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(context, question):
    """
    Generate an answer using Flan-T5 model given context + question.
    """
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = qa_model(prompt, max_length=150, do_sample=True)
    return response[0]['generated_text']
