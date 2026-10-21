import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")


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

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)

    # Build LLM
    llm = ChatOpenAI(
        temperature=0,
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        openai_api_key=together_api_key,
        base_url="https://api.together.xyz/v1"
    )

    # Conversational QA Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

    return qa_chain
