import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# DB_PATH configurable μέσω env var ώστε σε production (π.χ. Render) να μπορεί
# να δείχνει σε persistent disk (π.χ. /var/data/fruits_history.db) - χωρίς αυτό,
# η βάση θα χανόταν σε κάθε redeploy αφού το filesystem του container είναι εφήμερο.
# Τοπικά, χωρίς το env var, συνεχίζει να δουλεύει όπως πριν.
DB_PATH = os.environ.get("DB_PATH", "./fruits_history.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()