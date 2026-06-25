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
