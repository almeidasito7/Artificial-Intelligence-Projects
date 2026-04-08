#!/usr/bin/env python3
"""
eval.py — runs the golden set against the live /guardrail endpoint and reports accuracy.

Usage:
    python eval.py                        # uses http://localhost:8000
    python eval.py --base-url http://...  # custom host
    python eval.py --strategy keyword     # override STRATEGY env var
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import httpx

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval script for Citation Guardrail Engine")
    p.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the service")
    p.add_argument("--strategy", default=None, help="Override STRATEGY env var (keyword|semantic|hybrid)")
    return p.parse_args()


def run_eval(base_url: str, strategy: str | None) -> None:
    with open(GOLDEN_SET_PATH) as f:
        cases = json.load(f)

    if strategy:
        os.environ["STRATEGY"] = strategy

    endpoint = f"{base_url}/guardrail"
    results = []

    print(f"\n{BOLD}Citation Guardrail Engine - Eval{RESET}")
    print(f"Endpoint : {endpoint}")
    print(f"Strategy : {os.getenv('STRATEGY', 'semantic')}")
    print(f"Cases    : {len(cases)}\n")
    print(f"{'ID':<45} {'EXPECTED':<22} {'ACTUAL':<22} {'RESULT'}")
    print("-" * 110)

    for case in cases:
        case_id = case["id"]
        expected = case["expected"]
        payload = case["input"]

        try:
            resp = httpx.post(endpoint, json=payload, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            decision = data["citation_decision"]
            actual_status = decision["status"]
            actual_label = decision.get("matched_label")
        except Exception as exc:
            print(f"{case_id:<45} {'':22} {'ERROR':<22} {RED}FAIL{RESET} ({exc})")
            results.append({"id": case_id, "pass": False})
            continue

        status_ok = actual_status == expected["status"]
        label_ok = actual_label == expected["matched_label"]
        passed = status_ok and label_ok

        tag = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        expected_str = f"{expected['status']} / {expected['matched_label']}"
        actual_str = f"{actual_status} / {actual_label}"
        print(f"{case_id:<45} {expected_str:<22} {actual_str:<22} {tag}")

        if not passed:
            if not status_ok:
                print(f"  {YELLOW}↳ status mismatch: expected '{expected['status']}', got '{actual_status}'{RESET}")
            if not label_ok:
                print(f"  {YELLOW}↳ label mismatch: expected '{expected['matched_label']}', got '{actual_label}'{RESET}")

        results.append({"id": case_id, "pass": passed})

    total = len(results)
    passed_count = sum(1 for r in results if r["pass"])
    accuracy = passed_count / total * 100 if total else 0

    print("\n" + "-" * 110)
    print(f"{BOLD}Results: {passed_count}/{total} passed — accuracy: {accuracy:.1f}%{RESET}")

    failed = [r["id"] for r in results if not r["pass"]]
    if failed:
        print(f"\n{RED}Failed cases:{RESET}")
        for f_id in failed:
            print(f"  • {f_id}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}All cases passed!{RESET}")


if __name__ == "__main__":
    args = parse_args()
    run_eval(args.base_url, args.strategy)
