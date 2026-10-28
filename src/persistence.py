"""Checkpointer factory: where conversation state lives between turns.

  memory     in-process only (tests)
  sqlite     a local file (local development); NOT durable on Cloud Run
  firestore  external, survives Cloud Run instance replacement (deployment)
"""

from contextlib import asynccontextmanager

from src.config import get_settings


@asynccontextmanager
async def checkpointer():
    s = get_settings()
    if s.state_backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
    elif s.state_backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(s.sqlite_checkpoint_path) as saver:
            yield saver
    elif s.state_backend == "firestore":
        import os

        from langgraph_checkpoint_firestore import FirestoreSaver

        os.environ.setdefault("AGENTSTATE_QUIET", "1")  # silence the package's "unbounded history" hint
        # Firestore's (default) database; on Cloud Run the client authenticates as the service account.
        yield FirestoreSaver(s.firestore_project_id, s.firestore_collection)
    else:
        raise ValueError(f"unknown STATE_BACKEND {s.state_backend!r}")
