#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a "full-context" cipher dataset by joining:
  - sample.jsonl (instances): idx, question, answer, rule_id, category, needle...
  - rule.jsonl   (rules):     idx, title, rule_content, tag...

Output JSONL per line:
{
  "idx": "...",
  "rule_id": "...",
  "category": "...",
  "question": "Title + Rules + Task (full context)",
  "expected_answer": "...",
  "explanation": "..."
}

Notes:
- By default, we map sample.rule_id -> rule.idx (string match).
- explanation is built mainly from sample.needle when available; otherwise it
  references rule_content at a high level.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Iterable, Optional


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at {path}:{line_no}: {e}") from e


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s.strip()


def build_rule_map(rules_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Supports rule records keyed by either:
      - {"idx": "..."}  (as in your example)
      - {"rule_id": "..."} (in case the file uses a different key)
    """
    mp: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(rules_path):
        rid = r.get("idx", None)
        if rid is None:
            rid = r.get("rule_id", None)
        if rid is None:
            continue
        mp[str(rid)] = r
    return mp


def build_full_question(rule: Optional[Dict[str, Any]], sample: Dict[str, Any]) -> str:
    title = (rule or {}).get("title", "")
    rule_content = (rule or {}).get("rule_content", "")
    raw_q = str(sample.get("question", ""))

    chunks = []
    if title:
        chunks.append(f"Title: {title}")
    if rule_content:
        chunks.append("Rules:\n" + str(rule_content))
    chunks.append("Task:\n" + raw_q)

    return normalize_whitespace("\n\n".join(chunks))


def build_explanation(rule: Optional[Dict[str, Any]], sample: Dict[str, Any]) -> str:
    """
    Explanation policy:
    - If sample has `needle` (list of step strings), we include them as the key reasoning steps.
    - Otherwise, we give a short generic instruction referencing the rule title.
    - We DO NOT compute/verify anything; we just explain what procedure yields the given answer.
    """
    rule_id = str(sample.get("rule_id", ""))
    title = (rule or {}).get("title", "")
    needle = sample.get("needle", None)

    parts = []
    if title:
        parts.append(f"Use the procedure defined in “{title}” (rule_id={rule_id}).")
    else:
        parts.append(f"Use the procedure defined by rule_id={rule_id}.")

    # Give a tiny anchor to the input
    q = str(sample.get("question", "")).strip()
    if q:
        first = q.splitlines()[0].strip()
        if first:
            parts.append(f"Input statement: {first}")

    if isinstance(needle, list) and needle:
        parts.append("Key steps:")
        parts.extend([f"- {str(x)}" for x in needle])
        parts.append("Apply the steps sequentially to each character in the plaintext/ciphertext to obtain the final output.")
    else:
        parts.append("Apply the rule’s steps sequentially to each character to obtain the final output.")

    parts.append(f"Expected answer (gold): {sample.get('answer','')}")
    return normalize_whitespace("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="sample.jsonl", help="Path to sample.jsonl")
    ap.add_argument("--rules", default="rule.jsonl", help="Path to rule.jsonl")
    ap.add_argument("--out", default="cipher_dataset.jsonl", help="Output path for full-context dataset")
    args = ap.parse_args()

    sample_path = Path(args.sample)
    rules_path = Path(args.rules)
    out_path = Path(args.out)

    rule_map = build_rule_map(rules_path)

    rows = []
    missing_rules = 0

    for s in read_jsonl(sample_path):
        rule_id = str(s.get("rule_id", ""))
        rule = rule_map.get(rule_id)

        if rule is None:
            missing_rules += 1

        row = {
            "idx": str(s.get("idx", "")),
            "rule_id": rule_id,
            "category": s.get("category", ""),
            "question": build_full_question(rule, s),
            "expected_answer": s.get("answer", ""),
            "explanation": build_explanation(rule, s),
        }
        rows.append(row)

    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    if missing_rules:
        print(f"Warning: {missing_rules} samples had no matching rule in rule.jsonl (rule_id not found).")


if __name__ == "__main__":
    main()
