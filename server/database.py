import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from config import SERVER_DATABASE_URL, ROUND_TIMEOUT

engine = create_engine(SERVER_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String, nullable=False, default="open")
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    deadline = Column(DateTime, nullable=False)


class WeightUpdate(Base):
    __tablename__ = "weight_updates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    round_id = Column(Integer, nullable=False)
    user_id = Column(String, nullable=False)
    weights = Column(LargeBinary, nullable=False)
    num_samples = Column(Integer, nullable=False)
    submitted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(engine)


def create_round() -> Round:
    with SessionLocal() as session:
        round = Round(deadline=datetime.now(timezone.utc) + timedelta(seconds=ROUND_TIMEOUT))
        session.add(round)
        session.commit()
        session.refresh(round)
        return round


def get_open_round() -> Round | None:
    with SessionLocal() as session:
        return session.query(Round).filter(Round.status == "open").first()


def close_round(round_id: int):
    with SessionLocal() as session:
        round = session.query(Round).filter(Round.id == round_id).first()
        round.status = "complete"
        session.commit()


def insert_weight_update(round_id: int, user_id: str, weights: bytes, num_samples: int):
    with SessionLocal() as session:
        update = WeightUpdate(round_id=round_id, user_id=user_id, weights=weights, num_samples=num_samples)
        session.add(update)
        session.commit()


def get_updates_for_round(round_id: int) -> list[WeightUpdate]:
    with SessionLocal() as session:
        return session.query(WeightUpdate).filter(WeightUpdate.round_id == round_id).all()


def count_updates_for_round(round_id: int) -> int:
    with SessionLocal() as session:
        return session.query(WeightUpdate).filter(WeightUpdate.round_id == round_id).count()


def has_user_submitted(round_id: int, user_id: str) -> bool:
    with SessionLocal() as session:
        return session.query(WeightUpdate).filter(
            WeightUpdate.round_id == round_id,
            WeightUpdate.user_id == user_id
        ).first() is not None


def delete_updates_for_round(round_id: int):
    with SessionLocal() as session:
        session.query(WeightUpdate).filter(WeightUpdate.round_id == round_id).delete()
        session.commit()


if __name__ == "__main__":
    init_db()
    print("Server database initialized")
