from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
