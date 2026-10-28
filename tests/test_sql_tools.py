import sqlite3

import pytest

from src.tools.sql_tools import MAX_ROWS, SQLToolError, connect, describe_table, list_tables, run_sql


@pytest.fixture
def con(tmp_path):
    path = tmp_path / "t.db"
    setup = sqlite3.connect(path)
    setup.executescript(
        "CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, -- the id\n title TEXT, year INTEGER);"
        + "".join(f"INSERT INTO papers VALUES ('{i}', 'Paper {i}', {2020 + i % 5});" for i in range(80))
    )
    setup.commit()
    setup.close()
    return connect(str(path))


def test_select_returns_markdown_table(con):
    out = run_sql(con, "SELECT year, COUNT(*) AS n FROM papers GROUP BY year ORDER BY year")
    assert out.splitlines()[0] == "| year | n |"
    assert "| 2020 | 16 |" in out


def test_rows_are_capped(con):
    out = run_sql(con, "SELECT * FROM papers")
    assert f"truncated to {MAX_ROWS} rows" in out


@pytest.mark.parametrize("sql", [
    "DELETE FROM papers",
    "UPDATE papers SET title='x'",
    "DROP TABLE papers",
    "SELECT 1; DROP TABLE papers",
    "PRAGMA table_info(papers)",
    "ATTACH DATABASE 'x.db' AS x",
])
def test_writes_and_multi_statements_rejected(con, sql):
    with pytest.raises(SQLToolError):
        run_sql(con, sql)
    assert run_sql(con, "SELECT COUNT(*) FROM papers") == "| COUNT(*) |\n|---|\n| 80 |"


def test_with_cte_cannot_smuggle_a_write(con):
    with pytest.raises(SQLToolError):
        run_sql(con, "WITH x AS (SELECT 1) DELETE FROM papers")


def test_runaway_query_is_aborted(con):
    with pytest.raises(SQLToolError):
        run_sql(con, "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c) SELECT MAX(n) FROM c")


def test_describe_and_list(con):
    assert "papers (80 rows)" in list_tables(con)
    desc = describe_table(con, "papers")
    assert "-- the id" in desc and "Sample rows" in desc
    with pytest.raises(SQLToolError):
        describe_table(con, "nope")
