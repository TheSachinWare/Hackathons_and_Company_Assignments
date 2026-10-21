import json
import hashlib
import sqlite3
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import declarative_base, sessionmaker

# --- Database setup ---
Base = declarative_base()

class ComplianceResult(Base):
    __tablename__ = "compliance_results"
    input_hash = Column(String, primary_key=True)
    result_json = Column(String)

engine = create_engine("sqlite:///compliance_cache.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# --- Utility to create unique hash from RFP + company text ---
def generate_input_hash(rfp_json: dict, company_text: str) -> str:
    combined = json.dumps(rfp_json, sort_keys=True) + company_text.strip()
    return hashlib.sha256(combined.encode()).hexdigest()

# --- Check cache ---
def get_cached_result(rfp_json: dict, company_text: str):
    input_hash = generate_input_hash(rfp_json, company_text)
    session = Session()
    record = session.query(ComplianceResult).filter_by(input_hash=input_hash).first()
    if record:
        return json.loads(record.result_json)
    return None

# --- Save result to cache ---
def store_result(rfp_json: dict, company_text: str, result_dict: dict):
    input_hash = generate_input_hash(rfp_json, company_text)
    session = Session()
    record = ComplianceResult(input_hash=input_hash, result_json=json.dumps(result_dict))
    session.merge(record)  # merge = insert or update
    session.commit()
