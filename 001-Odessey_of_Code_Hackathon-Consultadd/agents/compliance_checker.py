import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import json
import os
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def load_llm():
    return ChatGroq(temperature=0.3, model_name="llama3-8b-8192", api_key=groq_api_key)

def read_company_pdf(file_path):
    reader = PdfReader(file_path)
    company_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            company_text += text
    return company_text


def validate_compliance(rfp_json: dict, company_text: str):
    rfp_json_str = json.dumps(rfp_json, indent=2)

    prompt_template = PromptTemplate(
        input_variables=["rfp_json", "company_profile"],
        template=""" 
     You are a compliance decision-making expert for government tenders and RFPs.

     Based on the RFP requirements and the company profile, decide:

- Whether the company is eligible (true/false).
- Provide reasoning.
- Highlight any mismatches or missing criteria.

Only return a valid JSON object. Do not include any explanations, introductions, or extra text.

Return response in EXACT JSON format:
{{
  "eligible": true/false,
  "decision_summary": "Summary of reasoning...",
  "highlighted_mismatches": ["criteria1", "criteria2"]
}}

RFP JSON:
{rfp_json}

Company Profile:
{company_profile}
"""

    )

    llm = load_llm()
    chain = LLMChain(prompt=prompt_template, llm=llm)
    response = chain.run(rfp_json=rfp_json_str, company_profile=company_text)

    # Try parsing the response
    try:
        parsed_response = json.loads(response)
    except json.JSONDecodeError:
        parsed_response = {
            "error": "LLM returned invalid JSON. Raw response:",
            "raw": response
        }

    return parsed_response

# ------------------------ Main Wrapper ------------------------
def run_compliance_decision(rfp_json, company_pdf_path):
    company_text = read_company_pdf(company_pdf_path)
    return validate_compliance(rfp_json, company_text)