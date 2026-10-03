"""
Client for pulling CVE records from the NVD REST API (v2.0), filtered to
the project's in-scope CWE categories (SQL Injection, Command Injection).

NVD API docs: https://nvd.nist.gov/developers/vulnerabilities

Note: NVD enforces a public rate limit of 5 requests / 30s without an API
key, or 50 requests / 30s with one (free, request at nvd.nist.gov). This
client paces requests automatically either way.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

from config import (
    NVD_API_BASE,
    NVD_API_KEY,
    NVD_RATE_LIMIT_DELAY_NO_KEY,
    NVD_RATE_LIMIT_DELAY_WITH_KEY,
    NVD_RESULTS_PER_PAGE,
    RAW_DIR,
    TARGET_CWE_IDS,
)

logger = logging.getLogger(__name__)


def _get_api_key() -> str | None:
    return os.environ.get("NVD_API_KEY", NVD_API_KEY)


def _request_with_retry(params: dict[str, Any], max_retries: int = 5) -> dict:
    """Single NVD API request with exponential backoff on failure/rate-limit."""
    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["apiKey"] = api_key

    delay = 2.0
    for attempt in range(1, max_retries + 1):
        resp = requests.get(NVD_API_BASE, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (403, 429):
            logger.warning(
                "NVD rate-limited (status %s), attempt %s/%s, backing off %.1fs",
                resp.status_code, attempt, max_retries, delay,
            )
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
    raise RuntimeError(f"NVD API request failed after {max_retries} retries: {params}")


def fetch_cves_for_cwe(cwe_id: str, max_results: int | None = None) -> list[dict]:
    """
    Pull all CVE records tagged with the given CWE ID from NVD, handling
    pagination. Returns a list of raw NVD 'vulnerabilities' entries.
    """
    results: list[dict] = []
    start_index = 0
    per_page = NVD_RESULTS_PER_PAGE
    rate_delay = NVD_RATE_LIMIT_DELAY_WITH_KEY if _get_api_key() else NVD_RATE_LIMIT_DELAY_NO_KEY

    while True:
        params = {
            "cweId": cwe_id,
            "resultsPerPage": per_page,
            "startIndex": start_index,
        }
        logger.info("Fetching %s CVEs, startIndex=%s", cwe_id, start_index)
        payload = _request_with_retry(params)

        vulns = payload.get("vulnerabilities", [])
        results.extend(vulns)

        total = payload.get("totalResults", len(vulns))
        start_index += per_page
        time.sleep(rate_delay)  # respect NVD rate limits between calls

        if start_index >= total:
            break
        if max_results and len(results) >= max_results:
            results = results[:max_results]
            break

    logger.info("Fetched %d CVE records for %s", len(results), cwe_id)
    return results


def collect_all_target_cves(max_results_per_cwe: int | None = None) -> dict[str, list[dict]]:
    """Fetch CVEs for every CWE in TARGET_CWE_IDS and save raw JSON to disk."""
    all_results: dict[str, list[dict]] = {}
    for cwe_id in TARGET_CWE_IDS:
        records = fetch_cves_for_cwe(cwe_id, max_results=max_results_per_cwe)
        all_results[cwe_id] = records

        out_path = Path(RAW_DIR) / f"nvd_{cwe_id.replace('CWE-', 'cwe')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        logger.info("Saved %d raw records to %s", len(records), out_path)

    return all_results


def extract_clean_records(raw_vulns: list[dict]) -> list[dict]:
    """
    Flatten raw NVD API records into the minimal fields the rest of the
    pipeline needs: cve_id, description text, cwe ids, affected products
    (CPE match strings), published date, cvss severity.
    """
    cleaned = []
    for entry in raw_vulns:
        cve = entry.get("cve", {})
        cve_id = cve.get("id")

        descriptions = cve.get("descriptions", [])
        description_text = next(
            (d["value"] for d in descriptions if d.get("lang") == "en"), ""
        )

        weaknesses = cve.get("weaknesses", [])
        cwe_ids = [
            d["value"]
            for w in weaknesses
            for d in w.get("description", [])
            if d.get("lang") == "en"
        ]

        configurations = cve.get("configurations", [])
        cpe_strings = [
            match.get("criteria")
            for config in configurations
            for node in config.get("nodes", [])
            for match in node.get("cpeMatch", [])
            if match.get("criteria")
        ]

        metrics = cve.get("metrics", {})
        cvss_severity = None
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics and metrics[key]:
                metric_entry = metrics[key][0]
                cvss_severity = metric_entry.get("cvssData", {}).get("baseSeverity") or metric_entry.get("baseSeverity")
                break

        cleaned.append({
            "cve_id": cve_id,
            "description": description_text,
            "cwe_ids": cwe_ids,
            "affected_cpe": cpe_strings,
            "published": cve.get("published"),
            "cvss_severity": cvss_severity,
        })
    return cleaned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw = collect_all_target_cves(max_results_per_cwe=200)  # cap for a quick first test run
    for cwe_id, records in raw.items():
        cleaned = extract_clean_records(records)
        out_path = Path(RAW_DIR) / f"clean_{cwe_id.replace('CWE-', 'cwe')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)
        logger.info("Wrote %d cleaned records to %s", len(cleaned), out_path)
