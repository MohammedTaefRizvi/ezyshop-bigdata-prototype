"""Serving API (stand-in for the recommendation micro-service behind the
website). Reads only from the low-latency serving store and the ALS item
factors - never from the lakehouse - so it stays fast under load.

Endpoints (JSON)::

    GET /health
    GET /recommendations/<customer_key>?viewed=P00012,P00007&k=10
        -> personalised top-k; if 'viewed' (live session) is given, results
           are re-ranked toward items similar to what is being viewed now
           (Assessment 2: 'coupled with live session data')
    GET /customers/<customer_key>       -> churn risk profile + next best action
    GET /metrics/live?limit=30          -> latest streaming windows
    GET /alerts?limit=50                -> latest real-time alerts

Run:  python -m ezyshop.api --port 8000
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .config import Config
from .serving_store import ServingStore


class Recommender:
    def __init__(self, cfg: Config):
        path = cfg.models_dir / "als_model.npz"
        self.ok = path.exists()
        if self.ok:
            m = np.load(path, allow_pickle=False)
            self.V = m["V"]
            self.items = list(m["items"])
            self.index = {p: i for i, p in enumerate(self.items)}
            norms = np.linalg.norm(self.V, axis=1, keepdims=True)
            self.Vn = self.V / np.where(norms == 0, 1, norms)

    def rerank(self, base: list[str], viewed: list[str], k: int) -> list[dict]:
        vi = [self.index[v] for v in viewed if v in self.index]
        if not self.ok or not vi:
            return [{"product_id": p, "score": None, "reason": "personalised (ALS)"} for p in base[:k]]
        # candidates = long-term ALS list + items most similar to the live session
        sim = self.Vn @ self.Vn[vi].mean(axis=0)
        long_term = {p: 1.0 - r / max(len(base), 1) for r, p in enumerate(base)}
        candidates = set(base) | {self.items[j] for j in np.argsort(-sim)[: 3 * k]}
        scored = [(p, 0.5 * long_term.get(p, 0.0) + 0.5 * float(sim[self.index[p]]))
                  for p in candidates if p not in viewed and p in self.index]
        scored.sort(key=lambda t: -t[1])
        return [{"product_id": p, "score": round(s, 4), "reason": "personalised + live session"} for p, s in scored[:k]]


def make_handler(cfg: Config):
    store = ServingStore(cfg.serving_db)
    rec = Recommender(cfg)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):  # quieter console
            pass

        def do_GET(self):  # noqa: N802
            t0 = time.perf_counter()
            u = urlparse(self.path)
            q = parse_qs(u.query)
            parts = [p for p in u.path.split("/") if p]
            try:
                if parts == ["health"]:
                    body = {"status": "ok", "model_loaded": rec.ok}
                elif len(parts) == 2 and parts[0] == "recommendations":
                    k = int(q.get("k", ["10"])[0])
                    rows = store.query("SELECT items, model_version FROM recommendations WHERE customer_key=?", (parts[1],))
                    if not rows:
                        return self._send(404, {"error": "unknown customer_key (or erased under GDPR)"})
                    viewed = [v for v in q.get("viewed", [""])[0].split(",") if v]
                    body = {"customer_key": parts[1], "model_version": rows[0]["model_version"],
                            "recommendations": rec.rerank(json.loads(rows[0]["items"]), viewed, k)}
                elif len(parts) == 2 and parts[0] == "customers":
                    rows = store.query("SELECT profile FROM customer_profile WHERE customer_key=?", (parts[1],))
                    if not rows:
                        return self._send(404, {"error": "unknown customer_key"})
                    body = {"customer_key": parts[1], **json.loads(rows[0]["profile"])}
                elif parts == ["metrics", "live"]:
                    n = int(q.get("limit", ["30"])[0])
                    body = {"windows": store.query("SELECT * FROM live_metrics ORDER BY window_start DESC LIMIT ?", (n,)),
                            "summary": store.get_kv("stream_summary")}
                elif parts == ["alerts"]:
                    n = int(q.get("limit", ["50"])[0])
                    body = {"alerts": store.query("SELECT * FROM alerts ORDER BY window_start DESC LIMIT ?", (n,))}
                else:
                    return self._send(404, {"error": "not found"})
                body["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                self._send(200, body)
            except Exception as e:  # pragma: no cover - defensive
                self._send(500, {"error": str(e)})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="EzyShop serving API")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    cfg = Config.from_env()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(cfg))
    print(f"EzyShop API on http://127.0.0.1:{a.port}  (try /health, /alerts, /metrics/live)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
