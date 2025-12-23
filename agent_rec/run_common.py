import json
import os
import random
from typing import Callable, Iterable, Tuple

import numpy as np
import torch

from agent_rec.config import EVAL_TOPK
from agent_rec.data import collect_data, load_tools, qids_with_rankings

DEFAULT_PARTS = ("PartI", "PartII", "PartIII")


def set_global_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def warn_if_topk_diff(topk: int, expected: int = EVAL_TOPK) -> None:
    if topk != expected:
        print(f"[warn] You set --topk={topk}, but protocol suggests fixed top10. Proceeding.")


def load_data_bundle(
    data_root: str,
    parts: Iterable[str] = DEFAULT_PARTS,
    *,
    with_tools: bool = False,
):
    bundle = collect_data(data_root, parts=list(parts))
    tools = load_tools(data_root) if with_tools else None
    return bundle, tools


def summarize_bundle(bundle, tools=None) -> None:
    if tools is None:
        print(
            f"Loaded {len(bundle.all_agents)} agents, {len(bundle.all_questions)} questions, "
            f"{len(bundle.all_rankings)} ranked entries."
        )
        return
    print(
        f"Loaded {len(bundle.all_agents)} agents, {len(bundle.all_questions)} questions, "
        f"{len(bundle.all_rankings)} ranked entries, {len(tools)} tools."
    )


def build_id_maps(all_questions, all_agents):
    q_ids = list(all_questions.keys())
    a_ids = list(all_agents.keys())
    qid2idx = {qid: i for i, qid in enumerate(q_ids)}
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}
    return q_ids, a_ids, qid2idx, aid2idx


def qids_with_rankings_and_log(q_ids, all_rankings):
    qids_in_rank = qids_with_rankings(q_ids, all_rankings)
    print(f"Questions with rankings: {len(qids_in_rank)} / {len(q_ids)}")
    return qids_in_rank


def training_cache_paths(cache_dir: str) -> Tuple[str, str, str, str]:
    return (
        os.path.join(cache_dir, "train_qids.json"),
        os.path.join(cache_dir, "valid_qids.json"),
        os.path.join(cache_dir, "pairs_train.npy"),
        os.path.join(cache_dir, "train_cache_meta.json"),
    )


def load_or_build_training_cache(
    cache_dir: str,
    rebuild: int,
    want_meta: dict,
    build_cache_fn: Callable[[], Tuple[list, list, np.ndarray]],
    *,
    save_message: str = "train/valid/pairs",
):
    train_qids_path, valid_qids_path, pairs_path, meta_path = training_cache_paths(cache_dir)
    use_cache = (
        os.path.exists(train_qids_path)
        and os.path.exists(valid_qids_path)
        and os.path.exists(pairs_path)
        and os.path.exists(meta_path)
        and rebuild == 0
    )
    if use_cache:
        with open(train_qids_path, "r", encoding="utf-8") as f:
            train_qids = json.load(f)
        with open(valid_qids_path, "r", encoding="utf-8") as f:
            valid_qids = json.load(f)
        pairs_idx_np = np.load(pairs_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta != want_meta:
            print("[cache] training cache meta mismatch, rebuilding...")
            use_cache = False

    if not use_cache:
        train_qids, valid_qids, pairs_idx_np = build_cache_fn()
        with open(train_qids_path, "w", encoding="utf-8") as f:
            json.dump(train_qids, f, ensure_ascii=False)
        with open(valid_qids_path, "w", encoding="utf-8") as f:
            json.dump(valid_qids, f, ensure_ascii=False)
        np.save(pairs_path, pairs_idx_np)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(want_meta, f, ensure_ascii=False, sort_keys=True)
        print(f"[cache] saved {save_message} to {cache_dir}")

    return train_qids, valid_qids, pairs_idx_np
