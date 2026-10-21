# rfp_compliance_workflow.py

from phidata.task import LLMTask
from phidata.agent import Agent
from phidata.workflow import Workflow

from agents.rfp_extracter_agent import run_rfp_extraction
from agents.compliance_checker import run_compliance_decision

# ---------------------- Define Agent: RFP Extractor ----------------------
rfp_extractor_agent = Agent(
    name="rfp_extractor",
    description="Extracts key eligibility criteria and summaries from RFPs",
    tasks=[
        LLMTask(
            name="extract_rfp_data",
            run=run_rfp_extraction  # function that takes in file_path and returns summary + structured data
        )
    ]
)

# ---------------------- Define Agent: Compliance Validator ----------------------
compliance_validator_agent = Agent(
    name="compliance_validator",
    description="Validates a company's profile against the extracted RFP criteria",
    tasks=[
        LLMTask(
            name="validate_company",
            run=run_compliance_decision_natural  # function takes in rfp_summary and company_pdf_path
        )
    ]
)

# ---------------------- Define Workflow ----------------------
def rfp_to_compliance_workflow(rfp_pdf_path: str, company_pdf_path: str):
    print("[WORKFLOW] Step 1: Extracting RFP Info...")
    rfp_summary = rfp_extractor_agent.run_task(
        "extract_rfp_data",
        file_path=rfp_pdf_path
    )

    print("[WORKFLOW] Step 2: Validating Compliance...")
    decision = compliance_validator_agent.run_task(
        "validate_company",
        rfp_summary=rfp_summary,
        company_pdf_path=company_pdf_path
    )

    return decision

# Initialize the workflow
rfp_compliance_workflow = Workflow(
    name="rfp_compliance_workflow",
    run=rfp_to_compliance_workflow
)

# Optional: Run test (for CLI/debug)
if __name__ == "__main__":
    rfp_pdf = "./sample_data/sample_rfp.pdf"
    company_pdf = "./sample_data/company_profile.pdf"
    result = rfp_compliance_workflow.run(rfp_pdf, company_pdf)
    print("\n[✅ Final Decision]:\n")
    print(result)
