# Technical Brief v2 — Consumer Rights Legal Assistant (Bangladesh)
## Build a working prototype using RAG + Groq + Streamlit

---

## What You Are Building

A web app where a user describes a consumer dispute, uploads evidence files (PDFs and images), and optionally provides a company's Terms & Conditions (by paste or PDF upload). The app analyzes the inputs against the **Consumer Rights Protection Act (CRPA) 2009 of Bangladesh** using a RAG pipeline and returns a structured legal information report. It provides **legal information only — not legal advice**.

---

## Project Structure

```
consumer-rights-app/
├── app.py                    # Streamlit frontend
├── backend/
│   ├── rag.py                # RAG pipeline (ChromaDB + LangChain)
│   ├── ingest.py             # One-time script to embed CRPA into ChromaDB
│   ├── file_parser.py        # Extracts text from PDFs, DOCX, TXT files
│   ├── vision.py             # Sends images to Llama 3.2 Vision on Groq
│   └── report_generator.py  # Assembles all evidence text before passing to RAG
├── knowledge_base/
│   └── crpa_2009.pdf         # Consumer Rights Protection Act 2009 (Bangladesh)
│                             # Download from: bdlaws.minlaw.gov.bd
├── chroma_db/                # Auto-created by ingest.py — do not touch
├── .env                      # GROQ_API_KEY=your_key_here
└── requirements.txt
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Frontend | Streamlit | Python-based UI, zero frontend code needed |
| LLM (Legal Analysis) | Groq — `llama-3.1-70b-versatile` | Free tier, best reasoning for legal text |
| LLM (Image Understanding) | Groq — `llama-3.2-11b-vision-preview` | Free tier, handles real-world messy images |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace) | Free, runs on CPU, supports Bengali + English |
| Vector DB | ChromaDB (local, persistent) | Stored in chroma_db/ folder |
| RAG Framework | LangChain | Handles chunking, retrieval, and LLM chaining |
| File Parsing | PyMuPDF (PDFs), python-docx (Word) | OCR removed — images go to vision model instead |

---

## Requirements.txt

```
streamlit
langchain
langchain-community
langchain-groq
chromadb
sentence-transformers
pymupdf
python-docx
python-dotenv
pillow
groq
```

> Note: pytesseract is NOT used. Images are handled by the Llama 3.2 Vision model on Groq, which is far more reliable than OCR for real-world evidence photos and screenshots.

---

## Step 1 — Knowledge Base Ingestion (ingest.py)

Run this **once** before launching the app. It loads the CRPA PDF, chunks it by legal section, embeds it, and stores it in ChromaDB.

```python
# backend/ingest.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_knowledge_base():
    loader = PyMuPDFLoader("knowledge_base/crpa_2009.pdf")
    documents = loader.load()

    # Chunk by legal section — never split mid-sentence within a clause
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "।", ".", " "]  # includes Bengali sentence separator
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    print(f"Successfully ingested {len(chunks)} chunks into ChromaDB.")

if __name__ == "__main__":
    ingest_knowledge_base()
```

---

## Step 2 — Image Vision Handler (vision.py)

This is the most important file for evidence handling. Instead of OCR, images are sent directly to Llama 3.2 Vision on Groq. The model understands the visual content and returns a detailed text description, which is then treated as evidence text in the RAG pipeline.

```python
# backend/vision.py
import os
import base64
import io
from groq import Groq
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

VISION_PROMPT = """You are analyzing an evidence image submitted as part of a consumer rights dispute in Bangladesh.

Look at this image carefully and extract ALL relevant information you can see, including:
- Order details (order ID, date, product name, price, quantity)
- Product condition (damaged, defective, incorrect item, missing parts)
- Seller or platform name
- Any communication between buyer and seller (chat messages, emails)
- Delivery information (courier, tracking, delivery date)
- Refund or return status
- Any dates, amounts, or reference numbers visible
- Any other detail that could be relevant to a consumer complaint

Be thorough and specific. Do not summarize — extract every detail visible in the image.
If the image is unclear or unreadable, say so explicitly."""

def describe_image(uploaded_file) -> str:
    """Send an image to Llama 3.2 Vision on Groq and return a detailed text description."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Read and encode image as base64
    image_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer after reading

    # Resize if too large (Groq vision has size limits)
    image = Image.open(io.BytesIO(image_bytes))
    if max(image.size) > 1568:
        image.thumbnail((1568, 1568), Image.LANCZOS)
        buffer = io.BytesIO()
        fmt = image.format if image.format else "JPEG"
        image.save(buffer, format=fmt)
        image_bytes = buffer.getvalue()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Detect media type
    filename = uploaded_file.name.lower()
    if filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif filename.endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"  # fallback

    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"[Image could not be analyzed: {str(e)}]"
```

---

## Step 3 — File Parser for PDFs and Documents (file_parser.py)

Handles non-image files. Note: images are NOT processed here — they go to vision.py instead.

```python
# backend/file_parser.py
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
```

---

## Step 4 — RAG Pipeline (rag.py)

Loads ChromaDB, retrieves the most relevant CRPA sections, and queries Groq with a strict legal prompt.

```python
# backend/rag.py
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

SYSTEM_PROMPT = """You are a legal information assistant specializing in the Consumer Rights Protection Act (CRPA) 2009 of Bangladesh.

Your task is to analyze a consumer dispute and identify which specific sections of the CRPA 2009 may have been violated, based ONLY on the legal text retrieved and provided to you in the context below. Do NOT use your training knowledge to cite laws — only use what is in the retrieved context.

STRICT RULES:
1. Only cite CRPA sections that appear in the retrieved legal text provided to you.
2. For every section you cite, you MUST state:
   - The section number
   - What that section says in plain language
   - Exactly how the user's situation maps to that provision
   Never cite a section number without explaining what it says and why it applies.
3. Do NOT give legal advice. Do NOT say "you should sue", "file a case", or recommend any legal strategy.
4. If Terms & Conditions are provided, check if any clauses were breached and list them separately.
5. If you cannot find a clear legal ground in the retrieved text, say so explicitly. Do not guess or hallucinate section numbers.
6. Always end with the disclaimer exactly as written below.

Respond ONLY in this structured format:

---
SUMMARY OF COMPLAINT
[One paragraph restating the complaint in formal language, based on the user's description and evidence]

LAWS POTENTIALLY VIOLATED UNDER CRPA 2009
[For each applicable section:]
• Section [X] — [Name of provision]
  What the law says: [Plain-language explanation of the section]
  How it applies here: [Specific mapping to the user's situation]

If no violation found: "Based on the information provided and the retrieved legal text, no clear violation of the CRPA 2009 was identified. This does not mean no violation occurred — consult a qualified lawyer for a full assessment."

TERMS & CONDITIONS VIOLATIONS
[Only include if T&C was provided]
• [Clause or section of T&C] — [How it was breached]
If T&C not provided: Omit this section entirely.

---
DISCLAIMER: This report provides legal information based on the Consumer Rights Protection Act 2009 of Bangladesh. It does not constitute legal advice and does not establish or confirm any legal claim. For legal proceedings, consult a qualified lawyer or contact the Directorate of National Consumer Rights Protection (DNCRP) at dncrp.gov.bd.
---
"""

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    # Retrieve top 6 chunks for better section coverage
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1  # Low temperature = more consistent, less creative
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"{SYSTEM_PROMPT}\n\nRetrieved legal text from CRPA 2009:\n{{context}}\n\nCase details:\n{{question}}"
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

def analyze_case(description: str, evidence_text: str, tnc_text: str) -> str:
    chain = get_rag_chain()

    query = f"INCIDENT DESCRIPTION:\n{description}\n"
    if evidence_text.strip():
        query += f"\nEVIDENCE FROM FILES AND IMAGES:\n{evidence_text}\n"
    if tnc_text.strip():
        query += f"\nSELLER / PLATFORM TERMS & CONDITIONS:\n{tnc_text}\n"

    result = chain.run(query)
    return result
```

---

## Step 5 — Streamlit Frontend (app.py)

The complete UI. Evidence files (PDFs and images) are handled separately — images go to the vision model, PDFs go to the text extractor. T&C can be pasted OR uploaded as a PDF.

```python
# app.py
import streamlit as st
from backend.file_parser import extract_text_from_file, is_image, is_pdf
from backend.vision import describe_image
from backend.rag import analyze_case

st.set_page_config(
    page_title="Consumer Rights Assistant — Bangladesh",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Consumer Rights Legal Assistant")
st.caption("Bangladesh — Based on the Consumer Rights Protection Act 2009")
st.info(
    "This tool provides **legal information only** based on the CRPA 2009. "
    "It does not constitute legal advice.",
    icon="ℹ️"
)

st.divider()

# ── INPUT 1: Incident Description ──────────────────────────────────────────────
st.subheader("1. Describe Your Complaint")
description = st.text_area(
    "What happened? Include what you ordered, what went wrong, and how the seller responded.",
    height=180,
    placeholder=(
        "Example: I ordered a Samsung smartphone from Daraz on [date] for BDT 25,000. "
        "It arrived with a cracked screen and missing charger. I contacted the seller "
        "3 times over 2 weeks via Daraz chat but received no response and no refund."
    )
)

# ── INPUT 2: Evidence Files (PDFs + Images) ────────────────────────────────────
st.subheader("2. Upload Evidence")
st.caption("Accepted: PDF, PNG, JPG, JPEG, DOCX, TXT — Images are analyzed by a vision AI, not OCR")
uploaded_files = st.file_uploader(
    "Upload receipts, screenshots, photos of damaged product, chat screenshots, invoices",
    accept_multiple_files=True,
    type=["pdf", "png", "jpg", "jpeg", "docx", "txt", "webp"]
)

# ── INPUT 3: Terms & Conditions ────────────────────────────────────────────────
st.subheader("3. Seller Terms & Conditions (Optional)")
st.caption("Provide the seller's T&C so the app can check if they violated their own policies")

tnc_input_method = st.radio(
    "How would you like to provide the T&C?",
    options=["Paste text", "Upload PDF"],
    horizontal=True
)

tnc_text = ""

if tnc_input_method == "Paste text":
    tnc_text = st.text_area(
        "Paste the seller or platform's Terms & Conditions here",
        height=150,
        placeholder="Paste Daraz, Shajgoj, or any other seller's T&C here..."
    )

elif tnc_input_method == "Upload PDF":
    tnc_file = st.file_uploader(
        "Upload T&C as PDF",
        type=["pdf"],
        key="tnc_uploader"
    )
    if tnc_file is not None:
        with st.spinner("Extracting text from T&C PDF..."):
            tnc_text = extract_text_from_file(tnc_file)
        if tnc_text.strip():
            st.success(f"T&C extracted successfully ({len(tnc_text)} characters)")
        else:
            st.warning("Could not extract text from the T&C PDF. Try pasting the text instead.")

st.divider()

# ── ANALYZE BUTTON ─────────────────────────────────────────────────────────────
if st.button("⚖️ Analyze My Case", type="primary", use_container_width=True):
    if not description.strip():
        st.warning("Please describe your complaint before analyzing.")
        st.stop()

    evidence_text = ""

    if uploaded_files:
        st.write("**Processing evidence files...**")
        progress = st.progress(0)
        total = len(uploaded_files)

        for i, f in enumerate(uploaded_files):
            progress.progress((i + 1) / total)

            if is_image(f.name):
                with st.spinner(f"Analyzing image: {f.name}"):
                    description_from_vision = describe_image(f)
                    evidence_text += f"\n--- Image Evidence: {f.name} ---\n{description_from_vision}\n"

            else:
                with st.spinner(f"Extracting text from: {f.name}"):
                    extracted = extract_text_from_file(f)
                    evidence_text += f"\n--- Document Evidence: {f.name} ---\n{extracted}\n"

        progress.empty()

    with st.spinner("Analyzing your case against the CRPA 2009..."):
        report = analyze_case(description, evidence_text, tnc_text)

    st.divider()
    st.subheader("📋 Legal Information Report")
    st.markdown(report)

    st.download_button(
        label="⬇️ Download Report as TXT",
        data=report,
        file_name="consumer_rights_report.txt",
        mime="text/plain",
        use_container_width=True
    )
```

---

## Step 6 — Environment Setup

```bash
# 1. Create project folder and enter it
mkdir consumer-rights-app && cd consumer-rights-app

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with your Groq API key
# Get your free key at: console.groq.com
echo "GROQ_API_KEY=your_key_here" > .env

# 5. Place the CRPA 2009 PDF in the knowledge_base/ folder
# Download from: bdlaws.minlaw.gov.bd
# Search: "Consumer Rights Protection Act 2009"

# 6. Run ingestion ONCE to build the vector database
python backend/ingest.py

# 7. Launch the app
streamlit run app.py
```

---

## Deployment — Hugging Face Spaces (Free)

1. Create an account at **huggingface.co**
2. Create a new Space → choose **Streamlit** as the SDK
3. Push the entire project folder to the Space repository
4. Go to Space **Settings → Repository Secrets** → add: `GROQ_API_KEY = your_key_here`
5. The Space auto-builds and gives you a shareable public URL

> **Important:** Run `ingest.py` locally first and commit the `chroma_db/` folder to the repo. This ensures the vector database is already built when deployed — HuggingFace Spaces does not run one-off scripts automatically.

---

## How Evidence Flows Through the System

```
User uploads files
        │
        ├── Is it an image? (PNG, JPG, JPEG, WEBP)
        │         │
        │         ▼
        │   Llama 3.2 Vision (Groq)
        │   "Describe all evidence details visible in this image"
        │         │
        │         ▼
        │   Detailed text description
        │
        └── Is it a PDF / DOCX / TXT?
                  │
                  ▼
            PyMuPDF / python-docx
            Extract raw text
                  │
                  ▼
            Plain text content

All text (from images + documents) combined
        │
        ▼
Incident description + evidence text + T&C text
        │
        ▼
LangChain RetrievalQA
        │
        ├── Embed query → ChromaDB → Retrieve top 6 CRPA sections
        │
        ▼
Llama 3.1 70B (Groq) + retrieved CRPA sections + case details
        │
        ▼
Structured Legal Information Report
```

---

## Key Constraints — Do Not Change These

- The LLM **must only cite CRPA sections retrieved from ChromaDB** — not from its training data. This is enforced by the prompt and the `temperature=0.1` setting.
- If no legal ground is found, the report **must say so explicitly** — the prompt forbids hallucinated citations.
- The disclaimer **must appear at the bottom of every report** — it is hardcoded in the system prompt.
- The app **must never give strategic legal advice** — the prompt explicitly forbids phrases like "you should sue" or "file a case."
- Do **not** use pytesseract or any OCR library. All image understanding goes through `vision.py` → Llama 3.2 Vision.
