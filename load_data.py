from PyPDF2 import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

if __name__ == "__main__":
    text = load_pdf("data/Sample.pdf")
    print(text[:1000])