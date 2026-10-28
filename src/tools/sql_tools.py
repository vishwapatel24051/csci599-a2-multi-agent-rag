"""Read-only SQL access to data/agentlab.db for the Data agent.

LLM-written SQL is untrusted input, so writes are blocked three ways:
  1. the connection is opened read-only (`mode=ro`),
  2. a SQLite authorizer denies every action except reads and function calls,
  3. only a single SELECT / WITH statement is accepted.
A progress-handler budget aborts runaway queries, and results are capped at MAX_ROWS.
"""

import sqlite3

from src.config import get_settings

MAX_ROWS = 50
MAX_VM_STEPS = 5_000_000  # ~ a few hundred ms of SQLite work

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


class SQLToolError(ValueError):
    pass


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_settings().sqlite_db_path
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    return con


def list_tables(con: sqlite3.Connection) -> str:
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    lines = [f"- {n} ({con.execute(f'SELECT COUNT(*) FROM \"{n}\"').fetchone()[0]} rows)" for n in names]
    return "Tables:\n" + "\n".join(lines)


def describe_table(con: sqlite3.Connection, table: str) -> str:
    row = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if row is None:
        raise SQLToolError(f"no table named {table!r}; call list_tables first")
    sample = run_sql(con, f'SELECT * FROM "{table}" LIMIT 3')
    return f"{row[0].strip()}\n\nSample rows:\n{sample}"


def _validate(sql: str) -> str:
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        raise SQLToolError("only one statement is allowed")
    first = stripped.split(None, 1)[0].upper() if stripped else ""
    if first not in ("SELECT", "WITH"):
        raise SQLToolError("only SELECT (or WITH … SELECT) queries are allowed")
    return stripped


def run_sql(con: sqlite3.Connection, sql: str) -> str:
    """Execute one read-only query; return a markdown table (at most MAX_ROWS rows)."""
    query = _validate(sql)
    steps = 0

    def budget():
        nonlocal steps
        steps += 1000
        return 1 if steps > MAX_VM_STEPS else 0  # non-zero aborts the query

    con.set_authorizer(_authorizer)
    con.set_progress_handler(budget, 1000)
    try:
        cur = con.execute(query)
        rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.DatabaseError as e:
        raise SQLToolError(f"SQL error: {e}") from e
    finally:
        con.set_authorizer(None)
        con.set_progress_handler(None, 0)

    cols = [d[0] for d in cur.description]
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    if not rows:
        return "(no rows)"

    def cell(v) -> str:
        s = "NULL" if v is None else str(v)
        s = s.replace("|", "\\|").replace("\n", " ")
        return s if len(s) <= 120 else s[:117] + "…"

    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    table += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    if truncated:
        table.append(f"(truncated to {MAX_ROWS} rows; add LIMIT or aggregate)")
    return "\n".join(table)
