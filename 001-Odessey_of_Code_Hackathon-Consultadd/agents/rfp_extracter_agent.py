# agents/rfp_extractor_agent.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")


def load_llm():
    return ChatGroq(
        temperature=0.4,
        model_name="llama3-8b-8192",
        api_key=groq_api_key
    )


def read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def build_vector_store(chunks):
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(chunks, embedding=embedding_model)
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {e}")


def query_rfp_summary(vectorstore, query, k=6):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
You are an expert RFP analyzer. Given the context below extracted from a government RFP document:

1. Extract key eligibility criteria like:
   - State registration requirement
   - Years of experience required
   - Past performance
   - Certifications needed
   - Submission deadlines

2. Identify any potential deal-breakers.

3. Provide a structured JSON format of key points.

4. Also provide a descriptive natural language summary.

RFP Context:
{context}
"""

    llm = load_llm()
    response = llm.invoke(system_prompt)
    return response.content
# ------------------------- WRAPPER FUNCTION -------------------------
def run_rfp_extraction(file_path):
    try:
        print("[INFO] Reading and processing PDF...")
        text = read_pdf(file_path)
        chunks = chunk_text(text)
        store = build_vector_store(chunks)

        print("[INFO] Querying LLM for eligibility info...")
        return query_rfp_summary(
            vectorstore=store,
            query="eligibility, certifications, state of registration, experience, past performance, deadlines"
        )
    except Exception as err:
        raise RuntimeError(f"[AGENT ERROR] {err}")
