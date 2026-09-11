from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_FILE = ROOT / "data" / "reviews" / "rakshak_reviews.db"


def main() -> None:
    if not DB_FILE.exists():
        print(f"No review database found: {DB_FILE}")
        return

    with sqlite3.connect(DB_FILE) as conn:
        before = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        conn.execute("DELETE FROM reviews")
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]

    print(f"Review database: {DB_FILE}")
    print(f"Deleted review rows: {before - after}")
    print(f"Remaining review rows: {after}")


if __name__ == "__main__":
    main()
