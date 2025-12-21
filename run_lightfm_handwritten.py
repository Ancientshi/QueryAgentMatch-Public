#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import random
import pickle
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from agent_rec.config import POS_TOPK, EVAL_TOPK, TFIDF_MAX_FEATURES
from agent_rec.data import (
    collect_data,
    ensure_cache_dir,
    dataset_signature,
    qids_with_rankings,
    stratified_train_valid_split,
    build_training_pairs,
    load_tools,
)
from agent_rec.features import (
    build_feature_cache,
    build_text_corpora,
    feature_cache_exists,
    load_feature_cache,
    load_q_vectorizer,
    normalize_features,
    save_feature_cache,
    save_q_vectorizer,
)
from agent_rec.knn import build_knn_cache, build_knn_cache_with_vectorizer, load_knn_cache
from agent_rec.eval import evaluate_sampled_knn_top10, split_eval_qids_by_part
from agent_rec.models.lightfm_handwritten import LightFMHandwritten, bpr_loss, csr_to_bag_lists

from utils import print_metrics_table  # 依赖你现有 utils.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="lightfm_handwritten", help="Cache folder name under .cache/")
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

    parser.add_argument("--alpha_id", type=float, default=1.0)
    parser.add_argument("--alpha_feat", type=float, default=1.0)
    parser.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)
    parser.add_argument("--rebuild_feature_cache", type=int, default=0)

    parser.add_argument("--user_features_path", type=str, default="")
    parser.add_argument("--item_features_path", type=str, default="")

    parser.add_argument("--knn_N", type=int, default=8)
    parser.add_argument("--eval_cand_size", type=int, default=100)
    parser.add_argument("--topk", type=int, default=EVAL_TOPK, help="Fixed to 10 by default")

    args = parser.parse_args()
    if args.topk != EVAL_TOPK:
        print(f"[warn] You set --topk={args.topk}, but protocol suggests fixed top10. Proceeding.")

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    bundle = collect_data(args.data_root, parts=["PartI", "PartII", "PartIII"])
    all_agents = bundle.all_agents
    all_questions = bundle.all_questions
    all_rankings = bundle.all_rankings
    qid_to_part = bundle.qid_to_part

    tools = load_tools(args.data_root)

    print(
        f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, "
        f"{len(all_rankings)} ranked entries, {len(tools)} tools."
    )

    q_ids = list(all_questions.keys())
    a_ids = list(all_agents.keys())
    qid2idx = {qid: i for i, qid in enumerate(q_ids)}
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}

    qids_in_rank = qids_with_rankings(q_ids, all_rankings)
    print(f"Questions with rankings: {len(qids_in_rank)} / {len(q_ids)}")

    cache_dir = ensure_cache_dir(args.data_root, args.exp_name)

    q_vectorizer_runtime = None
    if (not args.user_features_path) and (not args.item_features_path):
        if feature_cache_exists(cache_dir) and args.rebuild_feature_cache == 0:
            feature_cache = load_feature_cache(cache_dir)
            q_vectorizer_runtime = load_q_vectorizer(cache_dir)
            if q_vectorizer_runtime is None:
                _, q_texts, _, _, _, _, _ = build_text_corpora(all_agents, all_questions, tools)
                q_vectorizer_runtime = TfidfVectorizer(
                    max_features=args.max_features, lowercase=True
                ).fit(q_texts)
                save_q_vectorizer(cache_dir, q_vectorizer_runtime)
        else:
            feature_cache, q_vectorizer_runtime = build_feature_cache(
                all_agents, all_questions, tools, max_features=args.max_features
            )
            save_feature_cache(cache_dir, feature_cache)
            save_q_vectorizer(cache_dir, q_vectorizer_runtime)
            print(f"[cache] saved features to {cache_dir}")
    else:
        with open(args.user_features_path, "rb") as f:
            U = pickle.load(f)
        with open(args.item_features_path, "rb") as f:
            V = pickle.load(f)
        if U.shape[0] != len(q_ids) or V.shape[0] != len(a_ids):
            raise ValueError("CSR rows mismatch between provided features and data.")
        feature_cache = None
        q_vectorizer_runtime = None

    if feature_cache is None:
        Q_np = U.toarray().astype(np.float32)
        A_text_full_np = V.toarray().astype(np.float32)
    else:
        Q_np = feature_cache.Q.astype(np.float32)
        A_text_full_np = feature_cache.A_text_full.astype(np.float32)

    U_csr, V_csr = normalize_features(Q_np, A_text_full_np)
    print(f"[features] user_features: {U_csr.shape}, item_features: {V_csr.shape}")

    u_feats_per_row, u_vals_per_row, num_user_feats = csr_to_bag_lists(U_csr)
    i_feats_per_row, i_vals_per_row, num_item_feats = csr_to_bag_lists(V_csr)

    split_paths = (
        os.path.join(cache_dir, "train_qids.json"),
        os.path.join(cache_dir, "valid_qids.json"),
        os.path.join(cache_dir, "pairs_train.npy"),
        os.path.join(cache_dir, "train_cache_meta.json"),
    )

    want_meta = {
        "data_sig": dataset_signature(qids_in_rank, a_ids, {k: all_rankings[k] for k in qids_in_rank}),
        "pos_topk": int(POS_TOPK),
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
    }

    use_cache = all(os.path.exists(p) for p in split_paths) and (args.rebuild_training_cache == 0)
    if use_cache:
        with open(split_paths[0], "r", encoding="utf-8") as f:
            train_qids = json.load(f)
        with open(split_paths[1], "r", encoding="utf-8") as f:
            valid_qids = json.load(f)
        pairs_idx_np = np.load(split_paths[2])
        with open(split_paths[3], "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta != want_meta:
            print("[cache] training cache meta mismatch, rebuilding...")
            use_cache = False

    if not use_cache:
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

        with open(split_paths[0], "w", encoding="utf-8") as f:
            json.dump(train_qids, f, ensure_ascii=False)
        with open(split_paths[1], "w", encoding="utf-8") as f:
            json.dump(valid_qids, f, ensure_ascii=False)
        np.save(split_paths[2], pairs_idx_np)
        with open(split_paths[3], "w", encoding="utf-8") as f:
            json.dump(want_meta, f, ensure_ascii=False, sort_keys=True)

        print(f"[cache] saved train/valid/pairs to {cache_dir}")

    device = torch.device(args.device)
    model = LightFMHandwritten(
        num_q=len(q_ids),
        num_a=len(a_ids),
        num_user_feats=num_user_feats,
        num_item_feats=num_item_feats,
        factors=args.factors,
        add_bias=True,
        alpha_id=args.alpha_id,
        alpha_feat=args.alpha_feat,
        device=device,
    ).to(device)
    model.set_user_feat_lists(u_feats_per_row, u_vals_per_row)
    model.set_item_feat_lists(i_feats_per_row, i_vals_per_row)

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

    model_dir = os.path.join(cache_dir, "models")
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

    knn_path = os.path.join(cache_dir, "knn_copy.pkl")
    if q_vectorizer_runtime is None:
        build_knn_cache(
            train_qids=train_qids,
            all_questions=all_questions,
            qid2idx=qid2idx,
            model=model,
            cache_path=knn_path,
        )
    else:
        build_knn_cache_with_vectorizer(
            train_qids=train_qids,
            all_questions=all_questions,
            qid2idx=qid2idx,
            model=model,
            cache_path=knn_path,
            tfidf=q_vectorizer_runtime,
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
        score_mode="dot",
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
            score_mode="dot",
            seed=123,
            desc=f"Valid {part} (KNN q-vector, top{topk})",
        )
        print_metrics_table(f"Validation {part} (KNN q-vector)", m_part, ks=(topk,), filename=args.exp_name)


if __name__ == "__main__":
    main()
