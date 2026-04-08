"""
Seed script for the LLM Engineer Assessment database.
Generates realistic staffing data across multiple regions and divisions.
Run: python data/seed_database.py
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "staffing.db")

random.seed(42)

# --- Reference Data ---

REGIONS = ["Southeast", "West Coast", "Northeast", "Midwest", "Southwest"]

DIVISIONS = ["IT", "Engineering", "Finance", "Healthcare", "Administrative"]

STATUSES_JOB = ["Open", "Filled", "On Hold", "Closed"]
STATUSES_CANDIDATE = ["Available", "Placed", "Interviewing", "Inactive"]
STATUSES_PLACEMENT = ["Active", "Completed", "Terminated", "Extended"]

CLIENT_NAMES = [
    "Acme Corp", "Pinnacle Health Systems", "TechVault Inc", "Metro Financial Group",
    "Cascade Engineering", "Summit Healthcare", "BlueRidge Analytics", "Pacific Rim Tech",
    "Granite Solutions", "Lakeshore Medical", "RedOak Consulting", "Horizon Aerospace",
    "Clearwater Energy", "Atlas Manufacturing", "Beacon Financial", "Crestline Pharma",
    "Ironwood Systems", "Silverline Networks", "Northstar Logistics", "Triton Defense"
]

JOB_TITLES = {
    "IT": [
        "Software Engineer", "Data Analyst", "DevOps Engineer", "Cloud Architect",
        "IT Support Specialist", "QA Engineer", "Database Administrator",
        "Cybersecurity Analyst", "Systems Administrator", "Full Stack Developer"
    ],
    "Engineering": [
        "Mechanical Engineer", "Civil Engineer", "Electrical Engineer",
        "Project Engineer", "CAD Designer", "Quality Engineer",
        "Manufacturing Engineer", "Process Engineer", "Structural Engineer",
        "Controls Engineer"
    ],
    "Finance": [
        "Financial Analyst", "Accountant", "Tax Specialist", "Auditor",
        "Payroll Specialist", "Accounts Payable Clerk", "Controller",
        "Budget Analyst", "Credit Analyst", "Treasury Analyst"
    ],
    "Healthcare": [
        "Registered Nurse", "Medical Technologist", "Pharmacist",
        "Radiology Technician", "Physical Therapist", "Lab Technician",
        "Medical Coder", "Health Information Specialist", "Clinical Research Coordinator",
        "Occupational Therapist"
    ],
    "Administrative": [
        "Executive Assistant", "Office Manager", "Receptionist",
        "HR Coordinator", "Administrative Assistant", "Facilities Coordinator",
        "Procurement Specialist", "Training Coordinator", "Compliance Officer",
        "Records Manager"
    ]
}

FIRST_NAMES = [
    "James", "Maria", "Robert", "Jennifer", "Michael", "Linda", "William", "Patricia",
    "David", "Elizabeth", "Carlos", "Sarah", "Ahmed", "Jessica", "Wei", "Ashley",
    "Raj", "Amanda", "Tomas", "Stephanie", "Yuki", "Michelle", "Oleg", "Samantha",
    "Kwame", "Brittany", "Liam", "Fatima", "Noah", "Priya", "Ethan", "Aaliyah",
    "Lucas", "Mei", "Andrei", "Gabriela", "Felix", "Naomi", "Hassan", "Chloe"
]

LAST_NAMES = [
    "Smith", "Garcia", "Johnson", "Williams", "Brown", "Jones", "Davis", "Martinez",
    "Rodriguez", "Anderson", "Taylor", "Thomas", "Jackson", "White", "Harris", "Clark",
    "Lewis", "Robinson", "Walker", "Young", "Kim", "Patel", "Chen", "Singh",
    "Nguyen", "Santos", "Mueller", "Tanaka", "Okafor", "Kowalski", "Silva", "Johansson",
    "Ali", "Fernandez", "Ivanov", "Nakamura", "Park", "Costa", "Bianchi", "Hoffman"
]

SKILLS_BY_DIVISION = {
    "IT": ["Python", "Java", "AWS", "Azure", "SQL", "Docker", "Kubernetes", "React", "TypeScript", "Terraform"],
    "Engineering": ["AutoCAD", "SolidWorks", "MATLAB", "PLC Programming", "Six Sigma", "GD&T", "FEA", "CFD"],
    "Finance": ["Excel Advanced", "SAP", "QuickBooks", "Power BI", "Financial Modeling", "GAAP", "SOX Compliance"],
    "Healthcare": ["EMR Systems", "HIPAA", "BLS Certified", "IV Therapy", "Phlebotomy", "Epic Systems", "Cerner"],
    "Administrative": ["MS Office Suite", "Salesforce", "SAP", "Scheduling", "Travel Coordination", "Event Planning"]
}

VMS_PLATFORMS = ["Fieldglass", "Beeline", "VNDLY", "Coupa", "IQNavigator"]


def random_date(start_year=2024, end_year=2026):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 3, 15)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


def random_recent_date():
    """Date within last 6 months"""
    start = datetime(2025, 9, 1)
    end = datetime(2026, 3, 15)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


def create_tables(conn):
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            division TEXT NOT NULL,
            region TEXT NOT NULL,
            client_name TEXT NOT NULL,
            bill_rate REAL NOT NULL,
            pay_rate REAL NOT NULL,
            status TEXT NOT NULL,
            vms_platform TEXT,
            openings INTEGER NOT NULL DEFAULT 1,
            date_posted TEXT NOT NULL,
            date_closed TEXT,
            required_skills TEXT,
            job_type TEXT NOT NULL DEFAULT 'Contract',
            duration_weeks INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            division TEXT NOT NULL,
            region TEXT NOT NULL,
            status TEXT NOT NULL,
            skills TEXT,
            years_experience INTEGER,
            date_registered TEXT NOT NULL,
            last_activity_date TEXT,
            source TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS placements (
            placement_id INTEGER PRIMARY KEY,
            job_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            region TEXT NOT NULL,
            division TEXT NOT NULL,
            client_name TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT,
            bill_rate REAL NOT NULL,
            pay_rate REAL NOT NULL,
            status TEXT NOT NULL,
            hours_per_week REAL DEFAULT 40.0,
            vms_platform TEXT,
            margin_pct REAL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id),
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        )
    """)

    conn.commit()


def seed_jobs(conn, count=400):
    cursor = conn.cursor()
    jobs = []

    for i in range(1, count + 1):
        division = random.choice(DIVISIONS)
        region = random.choice(REGIONS)
        title = random.choice(JOB_TITLES[division])
        client = random.choice(CLIENT_NAMES)
        bill_rate = round(random.uniform(35, 180), 2)
        pay_rate = round(bill_rate * random.uniform(0.55, 0.75), 2)
        status = random.choices(STATUSES_JOB, weights=[30, 40, 10, 20])[0]
        vms = random.choice(VMS_PLATFORMS) if random.random() > 0.3 else None
        openings = random.choices([1, 2, 3, 5], weights=[60, 25, 10, 5])[0]
        date_posted = random_date()
        date_closed = random_date(2025, 2026) if status in ("Filled", "Closed") else None
        skills = ", ".join(random.sample(SKILLS_BY_DIVISION[division], k=min(3, len(SKILLS_BY_DIVISION[division]))))
        job_type = random.choices(["Contract", "Contract-to-Hire", "Direct Hire"], weights=[60, 25, 15])[0]
        duration = random.choice([4, 8, 12, 13, 26, 52]) if job_type != "Direct Hire" else None

        jobs.append((
            i, title, division, region, client, bill_rate, pay_rate,
            status, vms, openings, date_posted, date_closed, skills,
            job_type, duration
        ))

    cursor.executemany("""
        INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, jobs)
    conn.commit()
    return jobs


def seed_candidates(conn, count=600):
    cursor = conn.cursor()
    candidates = []
    sources = ["Job Board", "Referral", "LinkedIn", "Internal Database", "Career Fair", "Agency"]

    for i in range(1, count + 1):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        email = f"{first.lower()}.{last.lower()}{random.randint(1,99)}@email.com"
        phone = f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
        division = random.choice(DIVISIONS)
        region = random.choice(REGIONS)
        status = random.choices(STATUSES_CANDIDATE, weights=[30, 35, 20, 15])[0]
        skills = ", ".join(random.sample(SKILLS_BY_DIVISION[division], k=min(3, len(SKILLS_BY_DIVISION[division]))))
        yoe = random.randint(0, 25)
        date_reg = random_date(2023, 2026)
        last_active = random_recent_date() if status != "Inactive" else random_date(2024, 2025)
        source = random.choice(sources)

        candidates.append((
            i, first, last, email, phone, division, region, status,
            skills, yoe, date_reg, last_active, source
        ))

    cursor.executemany("""
        INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, candidates)
    conn.commit()
    return candidates


def seed_placements(conn, jobs, candidates, count=350):
    cursor = conn.cursor()
    placements = []

    placed_candidates = [c for c in candidates if c[7] == "Placed"]
    filled_jobs = [j for j in jobs if j[7] in ("Filled", "Closed")]

    for i in range(1, count + 1):
        job = random.choice(filled_jobs) if filled_jobs else random.choice(jobs)
        cand = random.choice(placed_candidates) if placed_candidates else random.choice(candidates)

        region = job[3]
        division = job[2]
        client = job[4]
        bill_rate = job[5]
        pay_rate = job[6]
        start_date = random_date(2024, 2026)
        status = random.choices(STATUSES_PLACEMENT, weights=[35, 35, 10, 20])[0]
        end_date = None
        if status in ("Completed", "Terminated"):
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = start_dt + timedelta(weeks=random.randint(4, 52))
            if end_dt <= datetime(2026, 3, 15):
                end_date = end_dt.strftime("%Y-%m-%d")
        hours = random.choices([40.0, 32.0, 20.0, 37.5], weights=[60, 15, 10, 15])[0]
        vms = job[8]
        margin = round((bill_rate - pay_rate) / bill_rate * 100, 2) if bill_rate > 0 else 0

        placements.append((
            i, job[0], cand[0], region, division, client,
            start_date, end_date, bill_rate, pay_rate, status,
            hours, vms, margin
        ))

    cursor.executemany("""
        INSERT INTO placements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, placements)
    conn.commit()


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    print("Creating tables...")
    create_tables(conn)

    print("Seeding jobs (400 records)...")
    jobs = seed_jobs(conn, 400)

    print("Seeding candidates (600 records)...")
    candidates = seed_candidates(conn, 600)

    print("Seeding placements (350 records)...")
    seed_placements(conn, jobs, candidates, 350)

    # Print summary
    cursor = conn.cursor()
    for table in ["jobs", "candidates", "placements"]:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} records")

    # Print distribution
    print("\nJobs by region:")
    for row in cursor.execute("SELECT region, COUNT(*) FROM jobs GROUP BY region ORDER BY region"):
        print(f"  {row[0]}: {row[1]}")

    print("\nJobs by division:")
    for row in cursor.execute("SELECT division, COUNT(*) FROM jobs GROUP BY division ORDER BY division"):
        print(f"  {row[0]}: {row[1]}")

    print("\nPlacements by status:")
    for row in cursor.execute("SELECT status, COUNT(*) FROM placements GROUP BY status ORDER BY status"):
        print(f"  {row[0]}: {row[1]}")

    conn.close()
    print(f"\nDatabase created at: {DB_PATH}")


if __name__ == "__main__":
    main()
