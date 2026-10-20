"""Cross-document SQLite signature cache.

Persists verified (sig_hash → proven_chain, proven_params) mappings in
~/.cache/wmclean/signatures.sqlite so previously encountered watermark
generators route in O(1) with zero LLM calls.

SAFETY: Only verified (non-FLAGGED) chains are written to the persistent library.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from wmclean.schemas import ClusterSignature, RemoverMethod

_DEFAULT_DB_PATH = Path.home() / ".cache" / "wmclean" / "signatures.sqlite"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS signatures (
    sig_hash TEXT PRIMARY KEY,
    page_kind TEXT NOT NULL,
    carrier_flags TEXT NOT NULL,       -- JSON array
    proven_chain TEXT NOT NULL,         -- JSON array of RemoverMethod values
    proven_params TEXT NOT NULL,        -- JSON dict
    historical_pass_rate REAL DEFAULT 1.0,
    hit_count INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


class SignatureLibrary:
    """Persistent SQLite cache of verified watermark signatures."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def lookup(self, sig_hash: str) -> Optional[ClusterSignature]:
        """Look up a signature by hash. Returns None if not found."""
        row = self._conn.execute(
            "SELECT sig_hash, page_kind, carrier_flags, proven_chain, "
            "proven_params, historical_pass_rate FROM signatures WHERE sig_hash = ?",
            (sig_hash,),
        ).fetchone()

        if row is None:
            return None

        # Increment hit count
        self._conn.execute(
            "UPDATE signatures SET hit_count = hit_count + 1, "
            "updated_at = datetime('now') WHERE sig_hash = ?",
            (sig_hash,),
        )
        self._conn.commit()

        from wmclean.schemas import CarrierFlag, PageKind

        return ClusterSignature(
            sig_hash=row[0],
            page_kind=PageKind(row[1]),
            carrier_flags=[CarrierFlag(f) for f in json.loads(row[2])],
            quant_geom_aspect=0.0,  # not stored — only chain/params matter for routing
            proven_chain=[RemoverMethod(m) for m in json.loads(row[3])],
            proven_params=json.loads(row[4]),
            historical_pass_rate=row[5],
        )

    def store(self, signature: ClusterSignature) -> None:
        """Store a verified signature.

        Raises ValueError if the signature has no proven_chain (i.e., unverified).
        """
        if not signature.proven_chain:
            raise ValueError("Cannot store unverified signature (no proven_chain)")

        self._conn.execute(
            "INSERT OR REPLACE INTO signatures "
            "(sig_hash, page_kind, carrier_flags, proven_chain, proven_params, "
            "historical_pass_rate, updated_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                signature.sig_hash,
                signature.page_kind.value,
                json.dumps([f.value for f in signature.carrier_flags]),
                json.dumps([m.value for m in signature.proven_chain]),
                json.dumps(signature.proven_params or {}),
                signature.historical_pass_rate or 1.0,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
