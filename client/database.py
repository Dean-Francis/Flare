import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime, timezone

engine = create_engine(DATABASE_URL, connect_args = {"check_same_thread": False})
SessionLocal = sessionmaker(bind = engine)

class Base(DeclarativeBase):
    pass

class FlaggedEmail(Base):
    __tablename__ = "flagged_emails"

    id = Column(Integer, primary_key = True, autoincrement = True)
    user_id = Column(String, nullable = False)
    body = Column(String, nullable = False)
    label = Column(Integer, nullable = False)
    timestamp = Column(DateTime, default = lambda: datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(engine)

def insert_flagged_email(user_id: str, body: str, label: int):
    with SessionLocal() as session:
        flagged_email = FlaggedEmail(user_id=user_id, body=body, label=label)
        session.add(flagged_email)
        session.commit()

def count_flagged_emails() -> int:
    with SessionLocal() as session:
        return session.query(FlaggedEmail).count()

if __name__ == "__main__":
    init_db()
    print("Database Initialized")
