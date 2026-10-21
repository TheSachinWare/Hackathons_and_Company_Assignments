"""
db.py
-----
Mock loan-account store. SQLite for the hackathon demo; in production this
would be a read-replica of the core banking / LOS (loan origination system)
database, accessed via a narrow, read-only service account -- the chat
assistant should NEVER hold write credentials to the ledger of record.

SECURITY NOTE: every query in this file uses parameterized placeholders (?).
Never use f-strings / .format() / % to build SQL with user input -- that is
the #1 classic SQLi vector and there is no reason to ever do it here.
"""

import sqlite3
import datetime
from contextlib import contextmanager

from security import hash_secret

DB_PATH = "bajaj_demo.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(seed: bool = True):
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                otp_hash TEXT,               -- hashed, time-boxed OTP (demo: static for repeatability)
                credit_score INTEGER NOT NULL,
                relationship_months INTEGER NOT NULL,
                monthly_income INTEGER,
                existing_mf_investor INTEGER DEFAULT 0,
                risk_consent_given INTEGER DEFAULT 0   -- explicit opt-in before any MF pitch, see crosssell.py
            );

            CREATE TABLE IF NOT EXISTS loans (
                loan_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                loan_type TEXT NOT NULL,
                principal INTEGER NOT NULL,
                outstanding_balance INTEGER NOT NULL,
                emi_amount INTEGER NOT NULL,
                interest_rate_pct REAL NOT NULL,
                tenure_months INTEGER NOT NULL,
                months_remaining INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                status TEXT NOT NULL              -- ACTIVE / CLOSED / DELINQUENT
            );

            CREATE TABLE IF NOT EXISTS repayment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                loan_id TEXT NOT NULL REFERENCES loans(loan_id),
                due_date TEXT NOT NULL,
                paid_date TEXT,
                dpd INTEGER NOT NULL DEFAULT 0    -- days-past-due for that installment
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                customer_id TEXT,
                action TEXT NOT NULL,
                detail TEXT
            );
            """
        )
        if seed:
            _seed(conn)


def _seed(conn):
    cur = conn.execute("SELECT COUNT(*) FROM customers")
    if cur.fetchone()[0] > 0:
        return  # already seeded

    demo_otp_hash = hash_secret("123456")  # demo-only static OTP; see README for real OTP flow

    customers = [
        # customer_id, name, phone, credit_score, relationship_months, income, existing_mf_investor
        ("CUST001", "Aarav Sharma", "9876543210", 812, 30, 140000, 0),
        ("CUST002", "Priya Verma", "9876543211", 705, 9, 65000, 0),
        ("CUST003", "Rohit Nair", "9876543212", 640, 4, 45000, 0),
        ("CUST004", "Sneha Iyer", "9876543213", 760, 18, 95000, 1),
    ]
    for cid, name, phone, score, rel_months, income, existing in customers:
        conn.execute(
            "INSERT INTO customers (customer_id, name, phone, otp_hash, credit_score, "
            "relationship_months, monthly_income, existing_mf_investor, risk_consent_given) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (cid, name, phone, demo_otp_hash, score, rel_months, income, existing),
        )

    today = datetime.date.today()
    loans = [
        ("LOAN001", "CUST001", "Home Loan", 5_000_000, 3_800_000, 42000, 8.5, 240, 190,
         (today - datetime.timedelta(days=900)).isoformat(), "ACTIVE"),
        ("LOAN002", "CUST002", "Personal Loan", 500_000, 260_000, 15500, 13.0, 36, 17,
         (today - datetime.timedelta(days=570)).isoformat(), "ACTIVE"),
        ("LOAN003", "CUST003", "Personal Loan", 300_000, 210_000, 11200, 15.5, 30, 21,
         (today - datetime.timedelta(days=270)).isoformat(), "ACTIVE"),
        ("LOAN004", "CUST004", "Car Loan", 900_000, 400_000, 19800, 9.2, 60, 24,
         (today - datetime.timedelta(days=1080)).isoformat(), "ACTIVE"),
    ]
    for row in loans:
        conn.execute(
            "INSERT INTO loans (loan_id, customer_id, loan_type, principal, outstanding_balance, "
            "emi_amount, interest_rate_pct, tenure_months, months_remaining, start_date, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    # Repayment history: CUST001 & CUST004 = clean; CUST002 = one 15-DPD blip;
    # CUST003 = a 45-DPD event in the last 12 months (should block MF pitch eligibility)
    history = []
    for i in range(12):
        d = (today - datetime.timedelta(days=30 * i)).isoformat()
        history.append(("LOAN001", d, d, 0))
        history.append(("LOAN004", d, d, 0))
        history.append(("LOAN002", d, d, 15 if i == 3 else 0))
        history.append(("LOAN003", d, d, 45 if i == 2 else 0))
    for loan_id, due, paid, dpd in history:
        conn.execute(
            "INSERT INTO repayment_history (loan_id, due_date, paid_date, dpd) VALUES (?, ?, ?, ?)",
            (loan_id, due, paid, dpd),
        )


# --------------------------- read-only query helpers -----------------------

def get_customer(customer_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT customer_id, name, phone, otp_hash, credit_score, relationship_months, "
            "monthly_income, existing_mf_investor, risk_consent_given FROM customers WHERE customer_id = ?",
            (customer_id,),
        )
        return cur.fetchone()


def find_customer_by_phone(phone: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM customers WHERE phone = ?", (phone,))
        return cur.fetchone()


def get_loans_for_customer(customer_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM loans WHERE customer_id = ? ORDER BY status DESC", (customer_id,)
        )
        return cur.fetchall()


def get_loan(loan_id: str, customer_id: str) -> sqlite3.Row | None:
    """Always scoped by customer_id too -> a loan_id alone is never sufficient
    to fetch a record; this is a second IDOR checkpoint at the data layer,
    independent of the authorization check already done in the API layer."""
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM loans WHERE loan_id = ? AND customer_id = ?", (loan_id, customer_id)
        )
        return cur.fetchone()


def get_repayment_history(loan_id: str, months: int = 12) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM repayment_history WHERE loan_id = ? ORDER BY due_date DESC LIMIT ?",
            (loan_id, months),
        )
        return cur.fetchall()


def set_risk_consent(customer_id: str, consent: bool) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE customers SET risk_consent_given = ? WHERE customer_id = ?",
            (1 if consent else 0, customer_id),
        )


def write_audit(customer_id: str | None, action: str, detail: str = "") -> None:
    """Append-only audit trail. In production this should also ship to an
    immutable log sink (e.g. CloudWatch/SIEM) that the application itself
    cannot delete from, for tamper-evidence during incident response."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log (ts, customer_id, action, detail) VALUES (?, ?, ?, ?)",
            (datetime.datetime.utcnow().isoformat(), customer_id, action, detail),
        )
