"""
test_api.py - exercises every endpoint of the deployment mock-up in-process
(FastAPI TestClient, no server needed) and saves the responses as evidence.

Usage (from part3_machine_learning):
    python deployment/test_api.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PART3_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PART3_DIR), str(PART3_DIR / "notebooks")]
import common  # noqa: E402
from deployment.app import app  # noqa: E402

logger = logging.getLogger(__name__)
OUT = PART3_DIR / "deployment" / "sample_api_responses.json"

CALLS = [
    ("GET", "/health", None, 200),
    ("POST", "/predict/traffic", {"datetime": "2018-10-02 08:00", "weather": "Clear", "temp_c": 10.0}, 200),
    ("POST", "/predict/traffic", {"datetime": "2018-10-06 03:00", "weather": "Clouds", "temp_c": 6.0}, 200),
    ("POST", "/predict/risk", {"datetime": "2018-01-16 07:00", "weather": "Fog", "temp_c": -5.0}, 200),
    ("POST", "/predict/risk", {"datetime": "2018-01-20 02:00", "weather": "Fog", "temp_c": -5.0}, 200),
    ("POST", "/recommend", {"day": "Friday", "weather": "Snow", "month": 1, "earliest": 14, "latest": 20}, 200),
    ("POST", "/predict/traffic", {"datetime": "2018-13-45 25:00", "weather": "Clear", "temp_c": 10.0}, 400),
    ("POST", "/predict/traffic", {"datetime": "2018-10-02 08:00", "weather": "Tornado", "temp_c": 10.0}, 400),
    ("POST", "/predict/traffic", {"datetime": "2018-10-02 08:00", "temp_c": 99}, 422),
]


def main() -> int:
    client = TestClient(app)
    evidence, failures = [], 0
    for method, path, body, expected in CALLS:
        response = client.request(method, path, json=body)
        ok = response.status_code == expected
        failures += not ok
        log = logger.info if ok else logger.error
        log("%s %s -> %d (expected %d)%s", method, path, response.status_code, expected,
            "" if ok else " FAILED")
        evidence.append({"request": {"method": method, "path": path, "body": body},
                         "status": response.status_code, "response": response.json()})
    OUT.write_text(json.dumps(evidence, indent=2))
    logger.info("Saved %d API calls to %s; %d failure(s)", len(evidence), OUT.name, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    common.configure_logging()
    sys.exit(main())
