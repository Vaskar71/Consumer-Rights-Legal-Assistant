# 🛡️ ConsumerShield — Bangladesh Consumer Rights Legal Assistant

ConsumerShield is a powerful, AI-driven legal assistant designed to help Bangladeshi citizens understand their rights under the **Consumer Rights Protection Act (CRPA) 2009**. By describing a consumer grievance and providing evidence, users can receive instant, plain-language analysis of their situation and discover exactly which sections of the law may protect them.

## 🌟 Key Features

- **⚖️ RAG-Powered Legal Analysis:** Queries a local vector database (`ChromaDB`) of the CRPA 2009 and constitutional articles to provide highly accurate, grounded legal context.
- **📄 Multi-modal Evidence Processing:** Upload receipts, invoices, or screenshots of chats. ConsumerShield uses state-of-the-art vision models and text parsers to extract text from images, PDFs, Word documents, and more.
- **📜 Terms & Conditions Cross-referencing:** Paste or upload a seller's Return/Refund policies or Terms & Conditions. The assistant will cross-reference the incident to see if the seller violated their own rules.
- **💬 Conversational Interface:** A premium, dark-themed Streamlit chat interface designed for maximum readability and user experience.
- **🚫 Anti-Hallucination Guardrails:** The system is strictly prompted to only cite laws actually present in the retrieved text and avoids giving actionable "legal advice."

## 🚀 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) with custom CSS injection for a polished UI.
- **Backend / LLM:** 
  - Text Analysis: `llama-3.3-70b-versatile` via **Groq** API for blazing-fast inference.
  - Vision/Image Processing: `llama-3.2-90b-vision-preview` via **Groq** API.
- **Vector Database:** [Chroma](https://www.trychroma.com/) for local embedding storage and semantic search.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` via HuggingFace.
- **Document Parsing:** `PyPDF2`, `python-docx`, `pytesseract`.

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- A [Groq](https://console.groq.com/) API Key.

### 1. Clone the repository
```bash
git clone https://github.com/Vaskar71/Consumer-Rights-Legal-Assistant.git
cd Consumer-Rights-Legal-Assistant
```

### 2. Set up the virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
*(Ensure you have the required dependencies listed in your requirements.txt file)*

### 4. Set Environment Variables
Create a `.env` file in the root directory (or use Streamlit secrets) and add your API key:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_TEXT_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=llama-3.2-90b-vision-preview
```

### 5. Run the Application
```bash
streamlit run consumer-rights-app/app.py
```

## ⚠️ Disclaimer
**ConsumerShield is a legal information tool, not a lawyer.** The information provided by this application is based on the Consumer Rights Protection Act 2009 of Bangladesh and is for educational and informational purposes only. It does not constitute legal advice. For formal proceedings, consult a qualified lawyer or contact the Directorate of National Consumer Rights Protection (DNCRP) at [dncrp.gov.bd](https://dncrp.gov.bd).
