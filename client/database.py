from config import DATABASE_URL
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from datetime import datetime, timezone
from dataclasses import dataclass

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class FlaggedEmail(Base):
    __tablename__ = "flagged_emails"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    body = Column(String, nullable=False)
    label = Column(Integer, nullable=False)
    trained = Column(Boolean, default=False, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


@dataclass
class FlaggedEmailData:
    id: int
    user_id: str
    body: str
    label: int


def init_db():
    Base.metadata.create_all(engine)


def insert_flagged_email(user_id: str, body: str, label: int):
    with SessionLocal() as session:
        session.add(FlaggedEmail(user_id=user_id, body=body, label=label))
        session.commit()


def count_flagged_emails() -> int:
    with SessionLocal() as session:
        return session.query(FlaggedEmail).filter(FlaggedEmail.trained == False).count()


def get_untrained_emails() -> list[FlaggedEmailData]:
    with SessionLocal() as session:
        rows = session.query(FlaggedEmail).filter(FlaggedEmail.trained == False).all()
        return [FlaggedEmailData(id=r.id, user_id=r.user_id, body=r.body, label=r.label) for r in rows]


def mark_emails_trained(email_ids: list[int]):
    with SessionLocal() as session:
        session.query(FlaggedEmail).filter(FlaggedEmail.id.in_(email_ids)).update(
            {"trained": True}, synchronize_session=False
        )
        session.commit()


if __name__ == "__main__":
    init_db()
    print("Database Initialized")
