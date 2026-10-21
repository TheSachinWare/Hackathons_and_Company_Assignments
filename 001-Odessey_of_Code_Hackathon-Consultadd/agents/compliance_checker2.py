#this one is a iteratively refined version of the prompt template 
#we are not using this in the streamlit AI 



import os
import json
import warnings
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Suppress warnings
warnings.filterwarnings("ignore")

# Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Load the LLM
def load_llm():
    return ChatGroq(
        temperature=0.3,
        model_name="llama3-8b-8192",
        api_key=groq_api_key
    )


# Read company profile from PDF
def read_company_pdf(file_path):
    reader = PdfReader(file_path)
    company_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            company_text += text
    return company_text


# Core evaluation logic
def validate_compliance(rfp_json, company_text: str):
    # Handle JSON string or dict
    if isinstance(rfp_json, str):
        try:
            rfp_json = json.loads(rfp_json)
        except json.JSONDecodeError as e:
            return f"❌ Failed to parse RFP JSON: {e}"

    # Convert dict into bullet-point string
    rfp_summary = "\n".join(f"- {key}: {value}" for key, value in rfp_json.items())

    prompt_template = PromptTemplate(
        input_variables=["rfp_summary", "company_profile"],
        template="""
You are an expert in Request for Proposal (RFP) evaluation.

Evaluate the company's profile against these 10 RFP compliance criteria:

1. Business registration in required state  
2. Relevant industry certifications (e.g., ISO, CMMI)  
3. Experience in similar projects (past performance)  
4. Financial stability or audited financials  
5. Required licenses or clearances  
6. Compliance with submission deadlines  
7. Staff qualifications and team structure  
8. Insurance coverage  
9. Local office presence  
10. Diversity certifications (e.g., HUB, DBE) if explicitly required

Score each criterion as 1 (fulfilled) or 0 (not fulfilled or not mentioned).  
Calculate a total score out of 10.

Then provide a decision:
- If score >= 7 → **Eligible**
- If score < 7 → **Not Eligible**

Return the result in plain English using this format:

----
**Eligibility Decision: ✅ Eligible / ❌ Not Eligible**

The company meets X out of 10 key criteria.

Explain briefly why they are eligible or not, and which areas were strong or lacking.
----

Only return the summary. No JSON, no tables.

Company Profile:
{company_profile}

RFP Summary:
{rfp_summary}
"""
    )

    llm = load_llm()
    chain = LLMChain(prompt=prompt_template, llm=llm)

    try:
        raw_response = chain.run(rfp_summary=rfp_summary, company_profile=company_text)
    except Exception as e:
        return f"❌ LLM processing failed: {str(e)}"

    return raw_response.strip()


# Main runner
def run_compliance_decision(rfp_json, company_pdf_path: str):
    company_text = read_company_pdf(company_pdf_path)
    result = validate_compliance(rfp_json, company_text)
    return result
