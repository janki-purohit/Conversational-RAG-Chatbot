# 📄 Conversational RAG Chatbot (Apache 2.0 Model)

A conversational chatbot powered by **Retrieval-Augmented Generation (RAG)**.  
Upload a PDF, ask questions, and get answers using Hugging Face models, FAISS, and Gradio.  

This project was built as part of my AI engineering learning journey and aligns with real-world applications like Atlan's AI-powered data tools. 🚀

---

## ✨ Features
- 📂 Upload any PDF (research papers, resumes, documents)  
- 🔎 Retrieve relevant text chunks with **Sentence Transformers + FAISS**  
- 🤖 Generate answers using **Google Flan-T5 (Apache 2.0 licensed)**  
- 🖥️ Interactive **Gradio Web UI**  
- 🌐 Deployable on **Hugging Face Spaces**  

---

## 🛠 Tech Stack
- **Python 3.9+**
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF text extraction  
- [SentenceTransformers](https://www.sbert.net/) – embeddings  
- [FAISS](https://github.com/facebookresearch/faiss) – vector search  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) – LLMs  
- [Gradio](https://www.gradio.app/) – frontend  

---

## 🚀 Getting Started (Local)

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Conversational-RAG-Chatbot.git
   cd Conversational-RAG-Chatbot
   
2. Install requirements:
pip install -r requirements.txt

3.📜 License

Uses google/flan-t5-base
 under Apache 2.0 License
.



