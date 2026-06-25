import fitz  # PyMuPDF
import docx
import io

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTENSIONS = {".pdf"}

def is_image(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")

def extract_text_from_file(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT. Does NOT handle images — use vision.py for those."""
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer

    if filename.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        if not text.strip():
            return "[PDF appears to be scanned/image-based — no text could be extracted]"
        return text

    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        return f"[Unsupported file type: {filename}]"
