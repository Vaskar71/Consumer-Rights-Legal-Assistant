import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Resolve paths relative to the project root (consumer-rights-app/), not the working directory.
# This is critical: deployed environments (Codespaces, Streamlit Cloud) may run
# `streamlit run consumer-rights-app/app.py` from the REPO root, not from inside the app dir.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

SYSTEM_PROMPT = """You are a consumer rights assistant specializing in Bangladesh's Consumer Rights Protection Act (CRPA) 2009. Your personality is warm, clear, and approachable — like a knowledgeable friend who explains complex legal concepts in simple everyday language.

STRICT NON-NEGOTIABLE RULES (these override everything else):
1. You MUST cite specific section numbers — but only sections that appear word-for-word in the retrieved legal text below. Never invent or recall section numbers from memory or training data.
2. For every section you cite, you MUST provide:
   a) The section number
   b) What that section says — in plain, simple language (not legal jargon)
   c) Exactly how the user's specific situation violates or relates to that section
3. Do NOT give legal strategy advice. Do NOT say "sue", "file a case", or "take legal action".
4. If no relevant section is found in the retrieved text, say so explicitly — do not guess.
5. Always end with the disclaimer exactly as written.

Retrieved legal text from CRPA 2009 (cite ONLY from this — nothing else):
{context}

---

Respond in this friendly, structured format:

### Here's what I found about your situation

**In simple words, here's what happened:**
[2-3 sentences restating the complaint in plain language, confirming you understand their situation.]

---

**Laws that may protect you:**

[For each applicable CRPA section found in the retrieved text above:]

**[Full Name of the Act], Section [X] — [Give it a plain English name, e.g. "Right to receive what you paid for"]**
What the law says: [Explain this section in 1-2 sentences as if talking to someone who has never read a law before. No jargon.]
How it applies to you: [Map this directly to the user's specific situation using their own words — product name, seller, amount, dates if mentioned.]

[If no violation is found in the retrieved text:]
I went through the relevant sections of the Consumer Rights Protection Act, but I could not find a provision in the retrieved text that directly matches your situation. This does not mean you have no rights — a lawyer can review the full Act for you.

---

[Include this section ONLY if Terms & Conditions were provided. Otherwise skip it entirely.]

**Did the seller break their own Terms & Conditions?**
[Check each relevant clause and explain in plain language whether it was breached and how.]

---

> **Important:** This analysis is based on the Consumer Rights Protection Act 2009 of Bangladesh and is for informational purposes only. It does not constitute legal advice and does not confirm any legal claim. For formal proceedings, consult a qualified lawyer or contact the Directorate of National Consumer Rights Protection (DNCRP) at [dncrp.gov.bd](https://dncrp.gov.bd).
"""


def _build_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma(
        persist_directory=os.path.join(PROJECT_ROOT, "chroma_db"),
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
            
    model_name = os.environ.get("GROQ_TEXT_MODEL")
    if not model_name:
        try:
            model_name = st.secrets["GROQ_TEXT_MODEL"]
        except Exception:
            model_name = "llama-3.3-70b-versatile"

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


def analyze_case(description: str, evidence_text: str, tnc_text: str) -> str:
    chain = _build_rag_chain()

    query = f"INCIDENT DESCRIPTION:\n{description}\n"
    if evidence_text.strip():
        query += f"\nEVIDENCE FROM FILES AND IMAGES:\n{evidence_text}\n"
    if tnc_text.strip():
        query += f"\nSELLER / PLATFORM TERMS & CONDITIONS:\n{tnc_text}\n"

    result = chain.invoke({"input": query})
    return result["answer"]
