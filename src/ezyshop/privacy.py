"""GDPR-oriented privacy controls (Assessment 2, section 6).

* Pseudonymisation: direct identifiers (customer_id) are replaced by a keyed
  HMAC-SHA256 token in Silver/Gold. Names, e-mails and phone numbers never leave
  Bronze (data minimisation). Without the secret salt the token cannot be
  linked back - but note pseudonymised data is still personal data under GDPR
  (Irti, 2021), so access controls still apply.
* Right to erasure: an erasure request records the token in a suppression list,
  deletes it from the serving store immediately, and every subsequent Silver
  rebuild filters it out, which propagates the deletion to Gold and the models.
"""
from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone

import pandas as pd

from .config import Config
from .serving_store import ServingStore


def pseudonymise(value, salt: str) -> str | None:
    if value is None or (isinstance(value, float) and value != value) or value == "":
        return None
    return hmac.new(salt.encode(), str(value).encode(), hashlib.sha256).hexdigest()[:16]


def pseudonymise_series(s: pd.Series, salt: str) -> pd.Series:
    uniq = s.dropna().unique()
    mapping = {u: pseudonymise(u, salt) for u in uniq}
    return s.map(mapping)


def suppressed_keys(cfg: Config) -> set[str]:
    if not cfg.serving_db.exists():
        return set()
    store = ServingStore(cfg.serving_db)
    try:
        return {r["customer_key"] for r in store.query("SELECT customer_key FROM gdpr_suppression")}
    finally:
        store.close()


def erase_customer(cfg: Config, customer_id: str) -> str:
    """Handle a GDPR Art. 17 erasure request for a raw customer_id."""
    key = pseudonymise(customer_id, cfg.pii_salt)
    store = ServingStore(cfg.serving_db)
    try:
        store.upsert_many("gdpr_suppression", [{"customer_key": key,
                                                "requested_at": datetime.now(timezone.utc).isoformat()}])
        store.conn.execute("DELETE FROM recommendations WHERE customer_key=?", (key,))
        store.conn.execute("DELETE FROM customer_profile WHERE customer_key=?", (key,))
        store.conn.commit()
    finally:
        store.close()
    return key
