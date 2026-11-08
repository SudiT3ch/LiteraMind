"""SQLAlchemy engine/session management (SQLite default; set DATABASE_URL for PostgreSQL)."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from config import get_settings

_engine = None
_Session = None


def init_db(url: str | None = None):
    """(Re)initialise engine and create tables. Passing url is used by tests."""
    global _engine, _Session
    from database.models import Base
    url = url or get_settings().database_url
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    _engine = create_engine(url, future=True, **kwargs)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def is_initialised() -> bool:
    return _engine is not None


@contextmanager
def session_scope() -> Iterator[Session]:
    if _Session is None:
        init_db()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
