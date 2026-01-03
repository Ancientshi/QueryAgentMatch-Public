#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute LLM/tool frequencies and agent popularity, cache the results, and plot
the popularity distribution for downstream analysis.

Popularity for an agent is defined as:
    popularity(agent) = LLM frequency + mean(tool frequencies)

Tool frequencies are winsorized at the 99.5th percentile to handle extreme
outliers before contributing to the popularity score.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from agent_rec.data import collect_data, ensure_cache_dir
from analysis.regime_shift_analysis import (
    PartStats,
    _apply_nips_style,
    _merge_stats_for_frequency,
    collect_part_stats,
)

UNKNOWN_LLM = "<UNK_LLM>"


def _agent_llm_and_tools(agent: Dict | None) -> Tuple[str, list[str]]:
    """Return (llm_id, tools[]) for an agent object, with safe defaults."""
    agent = agent or {}
    model = agent.get("M") or {}
    tool_info = agent.get("T") or {}
    llm_id = (model.get("id") or "").strip() or UNKNOWN_LLM
    tools = list(tool_info.get("tools") or [])
    return llm_id, tools


def _winsorize_counter(counter: Counter[str], percentile: float) -> Tuple[Counter[str], float]:
    """Clip counter values to the given percentile threshold."""
    if not counter:
        return Counter(), 0.0
    values = np.array(list(counter.values()), dtype=float)
    threshold = float(np.percentile(values, percentile))
    clipped = Counter({k: float(min(v, threshold)) for k, v in counter.items()})
    return clipped, threshold


def compute_llm_and_tool_frequencies(
    agent_degree: Counter[str], all_agents: Dict[str, dict], *, tool_percentile: float = 99.5
) -> Tuple[Counter[str], Counter[str], float]:
    """Aggregate positive assignment counts by LLM ID and tool, winsorizing tools."""
    llm_freq = Counter()
    tool_freq = Counter()
    for aid, cnt in agent_degree.items():
        llm_id, tools = _agent_llm_and_tools(all_agents.get(aid))
        llm_freq[llm_id] += cnt
        for tool in tools:
            tool_freq[tool] += cnt

    clipped_tool_freq, threshold = _winsorize_counter(tool_freq, tool_percentile)
    return llm_freq, clipped_tool_freq, threshold


def compute_agent_popularity(
    agent_degree: Counter[str],
    all_agents: Dict[str, dict],
    llm_frequency: Mapping[str, float],
    tool_frequency: Mapping[str, float],
) -> Dict[str, float]:
    """
    Popularity(agent) = LLM frequency + mean(tool frequencies).
    Uses winsorized tool frequencies to soften outliers.
    """
    popularity: Dict[str, float] = {}
    for aid in agent_degree.keys():
        llm_id, tools = _agent_llm_and_tools(all_agents.get(aid))
        llm_part = float(llm_frequency.get(llm_id, 0.0))
        if tools:
            tool_values = [float(tool_frequency.get(t, 0.0)) for t in tools]
            tool_part = float(np.mean(tool_values)) if tool_values else 0.0
        else:
            tool_part = 0.0
        popularity[aid] = llm_part + tool_part
    return popularity


def cache_popularity(
    cache_dir: Path,
    *,
    parts: Sequence[str],
    llm_frequency: Mapping[str, float],
    tool_frequency: Mapping[str, float],
    tool_clip_threshold: float,
    tool_percentile: float,
    agent_popularity: Mapping[str, float],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "parts": list(parts),
        "tool_percentile": tool_percentile,
        "tool_clip_threshold": tool_clip_threshold,
        "llm_frequency": dict(llm_frequency),
        "tool_frequency": dict(tool_frequency),
        "agent_popularity": dict(agent_popularity),
    }
    cache_path = cache_dir / "popularity_stats.json"
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return cache_path


def _plot_agent_popularity_curve(agent_popularity: Mapping[str, float], output_path: Path) -> None:
    if not agent_popularity:
        raise ValueError("No agent popularity values to plot.")

    _apply_nips_style()
    values = sorted(agent_popularity.values(), reverse=True)
    xs = np.arange(1, len(values) + 1, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(xs, values, color="#9467bd", linewidth=2.5)
    ax.set_title("Agent popularity curve")
    ax.set_xlabel("Agent rank by popularity")
    ax.set_ylabel("Popularity (LLM freq + mean tool freq)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute and plot agent popularity from rankings.")
    ap.add_argument("--data_root", required=True, help="Path to dataset root containing PartI/PartII/PartIII.")
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis") / "artifacts",
        help="Directory to save the popularity plot.",
    )
    ap.add_argument(
        "--parts",
        default="PartI,PartII,PartIII",
        help="Comma separated list of parts to include (default: all parts).",
    )
    ap.add_argument(
        "--tool_percentile",
        type=float,
        default=99.5,
        help="Upper percentile for winsorizing tool frequencies (default: 99.5).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    if not parts:
        raise ValueError("No parts specified for analysis.")

    bundle = collect_data(args.data_root, parts=parts)

    per_part_stats: list[PartStats] = []
    for part in ["PartI", "PartII", "PartIII"]:
        if part in parts:
            per_part_stats.append(collect_part_stats(bundle, [part]))

    if not per_part_stats:
        raise ValueError(f"No part statistics collected. Check parts filter: {parts}")

    merged = _merge_stats_for_frequency(per_part_stats)

    llm_freq, tool_freq, clip_threshold = compute_llm_and_tool_frequencies(
        merged.agent_degree, bundle.all_agents, tool_percentile=args.tool_percentile
    )
    agent_popularity = compute_agent_popularity(merged.agent_degree, bundle.all_agents, llm_freq, tool_freq)

    cache_root = Path(ensure_cache_dir(args.data_root, f"popularity_analysis_{'_'.join(parts)}"))
    cache_path = cache_popularity(
        cache_root,
        parts=parts,
        llm_frequency=llm_freq,
        tool_frequency=tool_freq,
        tool_clip_threshold=clip_threshold,
        tool_percentile=args.tool_percentile,
        agent_popularity=agent_popularity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "agent_popularity_curve.png"
    _plot_agent_popularity_curve(agent_popularity, figure_path)

    print(f"[OK] Popularity cache saved to: {cache_path}")
    print(f"[OK] Agent popularity curve saved to: {figure_path}")


if __name__ == "__main__":
    main()
