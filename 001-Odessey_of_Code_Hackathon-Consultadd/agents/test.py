from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0.3,
    model_name="llama3-8b-8192",
    api_key=groq_api_key
)

response = llm.invoke("What are the eligibility requirements for a government RFP?")
print(response.content)
