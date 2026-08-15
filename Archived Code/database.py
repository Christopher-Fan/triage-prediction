import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Fallback to local postgres for local testing
# Docker-Compose will override this
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://triage_user:triage_pass@localhost:5432/triage_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency for FastAPI routing
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally: 
        db.close()