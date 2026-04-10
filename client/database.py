from config import DATABASE_URL
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, func, distinct
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
    timestamp: str = ""


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
        return [FlaggedEmailData(id=r.id, user_id=r.user_id, body=r.body, label=r.label, timestamp=r.timestamp.isoformat() if r.timestamp else "") for r in rows]


def get_flagged_emails_by_user(user_id: str) -> list[FlaggedEmailData]:
    with SessionLocal() as session:
        rows = session.query(FlaggedEmail).filter(FlaggedEmail.user_id == user_id).order_by(FlaggedEmail.timestamp.desc()).all()
        return [FlaggedEmailData(id=r.id, user_id=r.user_id, body=r.body, label=r.label, timestamp=r.timestamp.isoformat() if r.timestamp else "") for r in rows]


def get_all_flagged_emails(user_id: str | None = None) -> list[FlaggedEmailData]:
    with SessionLocal() as session:
        q = session.query(FlaggedEmail)
        if user_id:
            q = q.filter(FlaggedEmail.user_id == user_id)
        rows = q.order_by(FlaggedEmail.timestamp.desc()).all()
        return [FlaggedEmailData(id=r.id, user_id=r.user_id, body=r.body, label=r.label, timestamp=r.timestamp.isoformat() if r.timestamp else "") for r in rows]


def get_distinct_user_ids() -> list[str]:
    with SessionLocal() as session:
        rows = session.query(distinct(FlaggedEmail.user_id)).all()
        return [r[0] for r in rows]


def get_flag_stats(user_id: str | None = None, start: str | None = None, end: str | None = None) -> list[dict]:
    """Returns per-date counts of false positives (label=0) and false negatives (label=1)."""
    with SessionLocal() as session:
        q = session.query(
            func.date(FlaggedEmail.timestamp).label("date"),
            FlaggedEmail.label,
            func.count().label("count"),
        )
        if user_id:
            q = q.filter(FlaggedEmail.user_id == user_id)
        if start:
            q = q.filter(FlaggedEmail.timestamp >= start)
        if end:
            q = q.filter(FlaggedEmail.timestamp <= end)
        q = q.group_by(func.date(FlaggedEmail.timestamp), FlaggedEmail.label)
        q = q.order_by(func.date(FlaggedEmail.timestamp))
        rows = q.all()

        # Merge into per-date dicts
        by_date: dict[str, dict] = {}
        for date_str, label, count in rows:
            d = str(date_str)
            if d not in by_date:
                by_date[d] = {"date": d, "false_positives": 0, "false_negatives": 0}
            if label == 0:
                by_date[d]["false_positives"] = count
            else:
                by_date[d]["false_negatives"] = count

        return list(by_date.values())


def mark_emails_trained(email_ids: list[int]):
    with SessionLocal() as session:
        session.query(FlaggedEmail).filter(FlaggedEmail.id.in_(email_ids)).update(
            {"trained": True}, synchronize_session=False
        )
        session.commit()


if __name__ == "__main__":
    init_db()
    print("Database Initialized")
