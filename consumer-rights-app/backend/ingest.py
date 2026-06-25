import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ingest_knowledge_base():
    pdf_files = glob.glob(os.path.join(PROJECT_ROOT, "knowledge_base", "*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in knowledge_base directory.")
        return

    all_chunks = []
    
    # Chunk by legal section — never split mid-sentence within a clause
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "।", ".", " "]  # includes Bengali sentence separator
    )

    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyMuPDFLoader(pdf_file)
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No content to ingest.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=os.path.join(PROJECT_ROOT, "chroma_db")
    )
    vectorstore.persist()
    print(f"Successfully ingested {len(all_chunks)} chunks from {len(pdf_files)} files into ChromaDB.")

if __name__ == "__main__":
    ingest_knowledge_base()
