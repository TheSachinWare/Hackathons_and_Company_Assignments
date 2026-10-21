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
        temperature=0.3,
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


def analyze_contract_risks(vectorstore, k=6):
    docs = vectorstore.similarity_search("termination clause, liability, penalties, arbitration, indemnity, biased terms", k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
You are a contract risk analysis expert.

Given the contract excerpts below, perform the following:

1. Identify **biased, one-sided, or risky clauses** that could put ConsultAdd at a legal or commercial disadvantage.
   Examples:
     - Unilateral termination rights
     - Vague penalty terms
     - Excessive liabilities
     - Mandatory arbitration without appeal
     - Biased indemnification
     - Hidden compliance obligations

2. For each such clause:
   - Quote the exact sentence/paragraph from the document
   - Explain why it may be a concern
   - Suggest a **modification** to make it fair/balanced

3. Mention any **missing clauses** that are normally advisable, such as:
   - Notice period for termination
   - Dispute resolution process
   - Data protection & confidentiality
   - Limitation of liability

Return a well-organized markdown or JSON output highlighting:
- 🔍 Risky Clause
- ⚠️ Concern
- ✅ Suggested Modification

Contract Context:
{context}
"""

    llm = load_llm()
    response = llm.invoke(system_prompt)
    return response.content

# ------------------------- WRAPPER FUNCTION -------------------------
def run_contract_risk_analysis(file_path):
    try:
        print("[INFO] Reading and processing contract...")
        text = read_pdf(file_path)
        chunks = chunk_text(text)
        store = build_vector_store(chunks)

        print("[INFO] Analyzing contract risks and biased clauses...")
        return analyze_contract_risks(store)
    except Exception as err:
        raise RuntimeError(f"[AGENT ERROR] {err}")
