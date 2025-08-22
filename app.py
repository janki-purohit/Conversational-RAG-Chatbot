import gradio as gr
from load_data import load_pdf
from rag_pipline import RAGPipeline
from qa_model import generate_answer

rag_instance = None

def upload_pdf(file):
    global rag_instance
    text = load_pdf(file.name)
    rag_instance = RAGPipeline()
    chunks = rag_instance.chunk_text(text)
    rag_instance.build_index(chunks)
    return "✅ PDF uploaded successfully! You can now ask questions."

def ask_question(question):
    if rag_instance is None:
        return "⚠️ Please upload a PDF first."
    context_chunks = rag_instance.query(question)
    context = " ".join(context_chunks)
    answer = generate_answer(context, question)
    return answer

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Conversational RAG Chatbot (Apache 2.0 Model)")
    with gr.Row():
        file_upload = gr.File(label="Upload PDF")
        upload_output = gr.Textbox(label="Status")
    file_upload.upload(upload_pdf, inputs=file_upload, outputs=upload_output)

    question = gr.Textbox(label="Ask a Question")
    answer = gr.Textbox(label="Answer")
    submit = gr.Button("Submit")
    submit.click(ask_question, inputs=question, outputs=answer)

demo.launch()
