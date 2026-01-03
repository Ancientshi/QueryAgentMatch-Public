#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute regime-shift evidence between dense (Part I) and one-off (Part II/III)
interaction topologies. Generates one table with summary metrics and one figure
with frequency curves + degree histograms.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from agent_rec.config import pos_topk_for_part
from agent_rec.data import DatasetBundle, collect_data


@dataclass
class PartStats:
    label: str
    parts: List[str]
    question_count: int
    positive_interactions: int
    agent_degree: Counter[str]
    unique_agents: set[str]

    @property
    def unique_agent_ratio(self) -> float:
        if self.positive_interactions == 0:
            return 0.0
        return len(self.unique_agents) / self.positive_interactions

    @property
    def entropy(self) -> float:
        total = self.positive_interactions
        if total == 0 or not self.agent_degree:
            return 0.0
        return -sum((cnt / total) * math.log(cnt / total) for cnt in self.agent_degree.values())

    @property
    def effective_size(self) -> float:
        return math.exp(self.entropy)

    def head_coverage(self, percentile: float) -> float:
        if self.positive_interactions == 0 or not self.agent_degree:
            return 0.0
        if percentile <= 0:
            return 0.0
        top_n = max(1, math.ceil(len(self.agent_degree) * percentile / 100.0))
        covered = sum(cnt for _, cnt in self.agent_degree.most_common(top_n))
        return covered / self.positive_interactions

    def degree_values(self) -> List[int]:
        return list(self.agent_degree.values())


def collect_part_stats(bundle: DatasetBundle, parts: Iterable[str]) -> PartStats:
    part_set = set(parts)
    degree = Counter()
    unique_agents: set[str] = set()
    question_count = 0
    positive_interactions = 0

    for qid, part in bundle.qid_to_part.items():
        if part not in part_set:
            continue
        ranking = bundle.all_rankings.get(qid, [])
        if not ranking:
            continue

        question_count += 1
        top_k = pos_topk_for_part(part)
        positives = [aid for aid in ranking[:top_k] if aid in bundle.all_agents]
        if not positives:
            continue

        degree.update(positives)
        unique_agents.update(positives)
        positive_interactions += len(positives)

    label = "/".join(sorted(part_set))
    return PartStats(
        label=label,
        parts=sorted(part_set),
        question_count=question_count,
        positive_interactions=positive_interactions,
        agent_degree=degree,
        unique_agents=unique_agents,
    )


def export_metrics_table(stats_list: Sequence[PartStats], output_csv: Path) -> None:
    fieldnames = [
        "label",
        "parts",
        "questions",
        "positive_interactions",
        "unique_agents",
        "unique_agent_ratio",
        "entropy",
        "effective_size",
        "top1pct_coverage",
        "top5pct_coverage",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_list:
            writer.writerow(
                {
                    "label": stats.label,
                    "parts": ",".join(stats.parts),
                    "questions": stats.question_count,
                    "positive_interactions": stats.positive_interactions,
                    "unique_agents": len(stats.unique_agents),
                    "unique_agent_ratio": f"{stats.unique_agent_ratio:.4f}",
                    "entropy": f"{stats.entropy:.4f}",
                    "effective_size": f"{stats.effective_size:.2f}",
                    "top1pct_coverage": f"{stats.head_coverage(1):.4f}",
                    "top5pct_coverage": f"{stats.head_coverage(5):.4f}",
                }
            )


def _apply_nips_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
        }
    )
    plt.style.use("seaborn-v0_8-whitegrid")


def _plot_frequency(ax, stats: PartStats, color: str, title: str, log_x: bool = False) -> None:
    freqs = [cnt for _, cnt in stats.agent_degree.most_common()]
    xs = range(1, len(freqs) + 1)
    ax.plot(xs, freqs, color=color, linewidth=2, label=title.split(":")[0])
    ax.set_title(title)
    ax.set_xlabel("Agent rank by positives")
    ax.set_ylabel("#Positive assignments")
    ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=False)


def plot_regime_shift(per_part: Sequence[PartStats], output_path: Path) -> None:
    _apply_nips_style()
    color_map = {
        "PartI": "#1f77b4",  # blue
        "PartII": "#d62728",  # red
        "PartIII": "#9467bd",  # purple
    }

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5), gridspec_kw={"wspace": 0.25})

    part_lookup = {"/".join(s.parts): s for s in per_part}
    part_i = part_lookup.get("PartI")
    part_ii = part_lookup.get("PartII")
    part_iii = part_lookup.get("PartIII")

    if part_i:
        _plot_frequency(axes[0], part_i, color_map["PartI"], "Part I: Agent frequency curve", log_x=False)
    if part_ii:
        _plot_frequency(axes[1], part_ii, color_map["PartII"], "Part II: Agent frequency curve", log_x=True)
    if part_iii:
        _plot_frequency(axes[2], part_iii, color_map["PartIII"], "Part III: Agent frequency curve", log_x=True)

    bins = 40
    percentile_cap = 99.5
    for stats in per_part:
        degrees = [v for v in stats.degree_values() if v > 0]
        if not degrees:
            continue
        cap = float(np.percentile(degrees, percentile_cap)) if len(degrees) > 1 else degrees[0]
        clipped = [min(v, cap) for v in degrees]
        axes[3].hist(
            clipped,
            bins=bins,
            alpha=0.45,
            label=stats.label,
            log=True,
            color=color_map.get(stats.label, "#333333"),
            edgecolor="white",
            linewidth=0.5,
        )
    axes[3].set_title("Agent-degree distribution (log-scaled, p99.5 capped)")
    axes[3].set_xlabel("#Positive assignments per agent")
    axes[3].set_ylabel("Agent count (log)")
    axes[3].grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    axes[3].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Regime shift evidence: dense vs one-off interactions.")
    ap.add_argument("--data_root", required=True, help="Path to dataset root containing PartI/PartII/PartIII.")
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis") / "artifacts",
        help="Directory to save the figure and metrics table.",
    )
    ap.add_argument(
        "--parts",
        default="PartI,PartII,PartIII",
        help="Comma separated list of parts to include (default: all parts).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = collect_data(args.data_root, parts=parts)

    part_stats: list[PartStats] = []
    for part in ["PartI", "PartII", "PartIII"]:
        if part in parts:
            part_stats.append(collect_part_stats(bundle, [part]))

    if not part_stats:
        raise ValueError(f"No part statistics collected. Check parts filter: {parts}")

    metrics_path = args.output_dir / "regime_shift_metrics.csv"
    export_metrics_table(part_stats, metrics_path)

    figure_path = args.output_dir / "regime_shift_frequency.png"
    plot_regime_shift(per_part=part_stats, output_path=figure_path)

    print(f"[OK] Metrics table saved to: {metrics_path}")
    print(f"[OK] Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
