#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import random
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm

from agent_rec.config import POS_TOPK, EVAL_TOPK
from agent_rec.data import (
    ensure_cache_dir,
    dataset_signature,
    stratified_train_valid_split,
    build_training_pairs,
)
from agent_rec.knn import build_knn_cache, load_knn_cache
from agent_rec.eval import evaluate_sampled_knn_top10, split_eval_qids_by_part
from agent_rec.models.bpr_mf import BPRMF, bpr_loss
from agent_rec.run_common import (
    build_id_maps,
    cache_key_from_meta,
    load_data_bundle,
    load_or_build_training_cache,
    qids_with_rankings_and_log,
    shared_cache_dir,
    set_global_seed,
    summarize_bundle,
    warn_if_topk_diff,
)

from utils import print_metrics_table  # 依赖你现有 utils.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="bpr_mf_knn", help="Cache folder name under .cache/")
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rebuild_training_cache", type=int, default=0)

    parser.add_argument("--knn_N", type=int, default=8)
    parser.add_argument("--eval_cand_size", type=int, default=100)
    parser.add_argument("--score_mode", type=str, default="dot", choices=["dot", "cosine"])
    parser.add_argument("--topk", type=int, default=EVAL_TOPK, help="Fixed to 10 by default")

    args = parser.parse_args()
    warn_if_topk_diff(args.topk)

    set_global_seed(1234)

    bundle, _ = load_data_bundle(args.data_root, with_tools=False)
    all_agents = bundle.all_agents
    all_questions = bundle.all_questions
    all_rankings = bundle.all_rankings
    qid_to_part = bundle.qid_to_part

    summarize_bundle(bundle)

    q_ids, a_ids, qid2idx, aid2idx = build_id_maps(all_questions, all_agents)
    qids_in_rank = qids_with_rankings_and_log(q_ids, all_rankings)

    data_sig = dataset_signature(qids_in_rank, a_ids, {k: all_rankings[k] for k in qids_in_rank})
    exp_cache_dir = ensure_cache_dir(args.data_root, args.exp_name)

    want_meta = {
        "data_sig": data_sig,
        "pos_topk": int(POS_TOPK),
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
    }
    training_cache_dir = shared_cache_dir(args.data_root, "training", f"{data_sig}_{cache_key_from_meta(want_meta)}")

    def build_cache():
        train_qids, valid_qids = stratified_train_valid_split(
            qids_in_rank, qid_to_part=qid_to_part, valid_ratio=args.valid_ratio, seed=args.split_seed
        )
        print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")

        rankings_train = {qid: all_rankings[qid] for qid in train_qids}
        pairs = build_training_pairs(
            rankings_train, a_ids, pos_topk=POS_TOPK, neg_per_pos=args.neg_per_pos, rng_seed=args.rng_seed_pairs
        )
        pairs_idx = [(qid2idx[q], aid2idx[p], aid2idx[n]) for (q, p, n) in pairs]
        pairs_idx_np = np.array(pairs_idx, dtype=np.int64)
        return train_qids, valid_qids, pairs_idx_np

    train_qids, valid_qids, pairs_idx_np = load_or_build_training_cache(
        training_cache_dir,
        args.rebuild_training_cache,
        want_meta,
        build_cache,
    )

    device = torch.device(args.device)
    model = BPRMF(num_q=len(q_ids), num_a=len(a_ids), factors=args.factors, add_bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pairs = pairs_idx_np.tolist()
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", leave=True, dynamic_ncols=True)
        model.train()
        for b in pbar:
            batch = pairs[b * args.batch_size:(b + 1) * args.batch_size]
            if not batch:
                continue
            q_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            pos_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            neg_idx = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            pos, neg = model(q_idx, pos_idx, neg_idx)
            loss = bpr_loss(pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total_loss / (b + 1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total_loss / num_batches if num_batches else 0.0):.4f}")

    model_dir = os.path.join(exp_cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    data_sig = want_meta["data_sig"]

    ckpt_path = os.path.join(model_dir, f"{args.exp_name}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{args.exp_name}_{data_sig}.json")

    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dims": {"num_q": len(q_ids), "num_a": len(a_ids), "factors": args.factors},
        "mappings": {"q_ids": q_ids, "a_ids": a_ids},
        "args": vars(args),
        "model_extra": model.extra_state_dict(),
    }
    torch.save(ckpt, ckpt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data_sig": data_sig, "q_ids": q_ids, "a_ids": a_ids}, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {ckpt_path}")
    print(f"[save] meta  -> {meta_path}")

    knn_path = os.path.join(exp_cache_dir, "knn_copy.pkl")
    build_knn_cache(
        train_qids=train_qids,
        all_questions=all_questions,
        qid2idx=qid2idx,
        model=model,
        cache_path=knn_path,
    )
    print("[cache] saved KNN-copy cache")
    knn_cache = load_knn_cache(knn_path)

    model.eval()
    topk = int(args.topk)

    overall_metrics = evaluate_sampled_knn_top10(
        model=model,
        aid2idx=aid2idx,
        all_rankings=all_rankings,
        all_questions=all_questions,
        eval_qids=valid_qids,
        knn_cache=knn_cache,
        cand_size=args.eval_cand_size,
        knn_N=args.knn_N,
        pos_topk=POS_TOPK,
        topk=topk,
        score_mode=args.score_mode,
        seed=123,
        desc=f"Valid Overall (KNN q-vector, top{topk})",
    )
    print_metrics_table("Validation Overall (KNN q-vector)", overall_metrics, ks=(topk,), filename=args.exp_name)

    part_splits = split_eval_qids_by_part(valid_qids, qid_to_part=qid_to_part)
    for part in ["PartI", "PartII", "PartIII"]:
        qids_part = part_splits.get(part, [])
        if not qids_part:
            continue
        m_part = evaluate_sampled_knn_top10(
            model=model,
            aid2idx=aid2idx,
            all_rankings=all_rankings,
            all_questions=all_questions,
            eval_qids=qids_part,
            knn_cache=knn_cache,
            cand_size=args.eval_cand_size,
            knn_N=args.knn_N,
            pos_topk=POS_TOPK,
            topk=topk,
            score_mode=args.score_mode,
            seed=123,
            desc=f"Valid {part} (KNN q-vector, top{topk})",
        )
        print_metrics_table(f"Validation {part} (KNN q-vector)", m_part, ks=(topk,), filename=args.exp_name)


if __name__ == "__main__":
    main()
