import streamlit as st
import tempfile
import os
import json
from dotenv import load_dotenv

from agents.rfp_extracter_agent import run_rfp_extraction
from agents.compliance_checker import run_compliance_decision
from agents.contractrisk_analyzer import run_contract_risk_analysis

# Chatbot imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

# Chatbot helper function
def process_documents(uploaded_files):
    documents = []

    for file in uploaded_files:
        suffix = file.name.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
            file_data = file.read()
            if not file_data:
                continue
            tmp_file.write(file_data)
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path) if suffix == "pdf" else Docx2txtLoader(tmp_path)
            docs = loader.load()
            documents.extend(docs)
        finally:
            os.remove(tmp_path)

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)

    llm = ChatOpenAI(
        temperature=0,
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        openai_api_key=together_api_key,
        base_url="https://api.together.xyz/v1"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

    return qa_chain

# Streamlit UI
st.set_page_config(page_title="ConsultAdd RFP Analyzer", layout="wide")
st.title("📊 ConsultAdd - RFP Analyzer Suite")

st.markdown("A unified dashboard to understand and evaluate RFPs effectively using AI-driven insights.")

# Sidebar Uploads
with st.sidebar:
    st.header("📂 Upload PDFs")
    rfp_file = st.file_uploader("📎 RFP Document", type=["pdf"])
    company_file = st.file_uploader("🏢 Company Profile", type=["pdf"])

    if not rfp_file or not company_file:
        st.warning("Please upload both RFP and Company PDFs to continue.")

# Main Content
if rfp_file and company_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_rfp:
        tmp_rfp.write(rfp_file.read())
        rfp_path = tmp_rfp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_company:
        tmp_company.write(company_file.read())
        company_path = tmp_company.name

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Key Requirements & Deal Breakers",
        "🧠 Know Your RFP (Chatbot)",
        "✅ Compliance Checker",
        "⚠️ Contract Risk Analyzer"
    ])

    # --- Tab 1: RFP Extractor ---
    with tab1:
        with st.spinner("🔍 Extracting RFP insights..."):
            if "rfp_summary" not in st.session_state:
                st.session_state.rfp_summary = run_rfp_extraction(rfp_path)

        rfp_summary = st.session_state.rfp_summary
        st.success("✅ Key RFP Requirements Identified")
        st.markdown("Below are the extracted key requirements and potential deal-breakers:")
        st.markdown(rfp_summary)

    # --- Tab 2: Chatbot ---
    with tab2:
        st.subheader("💬 Ask anything about the RFP")
        st.markdown("This is a smart Q&A agent that helps you understand the uploaded RFP.")

        if "qa_chain" not in st.session_state:
            with st.spinner("⚙️ Setting up chatbot..."):
                rfp_file.seek(0)  # Reset file pointer
                st.session_state.qa_chain = process_documents([rfp_file])
                st.session_state.chat_history = []

        user_question = st.text_input("Type your question about the RFP...")

        if user_question and st.session_state.qa_chain:
            result = st.session_state.qa_chain.invoke({"question": user_question})
            answer = result["answer"]
            st.session_state.chat_history.append((user_question, answer))

        for q, a in st.session_state.chat_history[::-1]:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

    # --- Tab 3: Compliance Checker ---
    with tab3:
        with st.spinner("⚖️ Checking compliance with company profile..."):
            if "rfp_summary" not in st.session_state:
                st.session_state.rfp_summary = run_rfp_extraction(rfp_path)

            compliance_result = run_compliance_decision(st.session_state.rfp_summary, company_path)

        st.success("🎯 Compliance Analysis Complete")
        st.subheader("📋 Compliance Report")
        st.json(compliance_result)

    # --- Tab 4: Contract Risk Analyzer ---
    with tab4:
        with st.spinner("🔍 Analyzing contract risks and biased clauses..."):
            risk_report = run_contract_risk_analysis(rfp_path)

        st.success("⚠️ Contract Risk Analysis Done")
        st.subheader("📌 Identified Risks & Suggestions")
        st.markdown(risk_report)

else:
    st.info("👈 Upload both RFP and Company profile PDFs to begin.")
