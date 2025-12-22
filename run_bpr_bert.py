#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from agent_rec.config import EVAL_TOPK, POS_TOPK
from agent_rec.data import (
    collect_data,
    dataset_signature,
    ensure_cache_dir,
    qids_with_rankings,
    stratified_train_valid_split,
    build_training_pairs,
    load_tools,
)
from agent_rec.eval import evaluate_sampled_embedding_topk, split_eval_qids_by_part
from agent_rec.features import build_agent_tool_id_buffers, build_transformer_corpora
from agent_rec.models.dnn import SimpleBPRDNN, bpr_loss

from utils import print_metrics_table


def ensure_transformer_cache_dir(cache_dir: str) -> str:
    d = os.path.join(cache_dir, "transformer_cache")
    os.makedirs(d, exist_ok=True)
    return d


def transformer_cache_exists(cache_dir: str) -> bool:
    needed = [
        "q_ids.json",
        "a_ids.json",
        "tool_names.json",
        "Q_emb.npy",
        "A_emb.npy",
        "agent_tool_idx_padded.npy",
        "agent_tool_mask.npy",
        "enc_meta.json",
    ]
    return all(os.path.exists(os.path.join(cache_dir, name)) for name in needed)


def save_transformer_cache(
    cache_dir: str,
    q_ids,
    a_ids,
    tool_names,
    Q_emb,
    A_emb,
    agent_tool_idx_padded,
    agent_tool_mask,
    enc_meta,
):
    with open(os.path.join(cache_dir, "q_ids.json"), "w", encoding="utf-8") as f:
        json.dump(q_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "a_ids.json"), "w", encoding="utf-8") as f:
        json.dump(a_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "tool_names.json"), "w", encoding="utf-8") as f:
        json.dump(tool_names, f, ensure_ascii=False)
    np.save(os.path.join(cache_dir, "Q_emb.npy"), Q_emb.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_emb.npy"), A_emb.astype(np.float32))
    np.save(os.path.join(cache_dir, "agent_tool_idx_padded.npy"), agent_tool_idx_padded.astype(np.int64))
    np.save(os.path.join(cache_dir, "agent_tool_mask.npy"), agent_tool_mask.astype(np.float32))
    with open(os.path.join(cache_dir, "enc_meta.json"), "w", encoding="utf-8") as f:
        json.dump(enc_meta, f, ensure_ascii=False)


def load_transformer_cache(cache_dir: str):
    with open(os.path.join(cache_dir, "q_ids.json"), "r", encoding="utf-8") as f:
        q_ids = json.load(f)
    with open(os.path.join(cache_dir, "a_ids.json"), "r", encoding="utf-8") as f:
        a_ids = json.load(f)
    with open(os.path.join(cache_dir, "tool_names.json"), "r", encoding="utf-8") as f:
        tool_names = json.load(f)
    with open(os.path.join(cache_dir, "enc_meta.json"), "r", encoding="utf-8") as f:
        enc_meta = json.load(f)
    Q_emb = np.load(os.path.join(cache_dir, "Q_emb.npy"))
    A_emb = np.load(os.path.join(cache_dir, "A_emb.npy"))
    agent_tool_idx_padded = np.load(os.path.join(cache_dir, "agent_tool_idx_padded.npy"))
    agent_tool_mask = np.load(os.path.join(cache_dir, "agent_tool_mask.npy"))
    return (
        q_ids,
        a_ids,
        tool_names,
        Q_emb,
        A_emb,
        agent_tool_idx_padded,
        agent_tool_mask,
        enc_meta,
    )


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    encoder,
    device,
    max_len: int = 128,
    batch_size: int = 256,
    pooling: str = "cls",
):
    embs = []
    use_cls = pooling == "cls"
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with Transformer", dynamic_ncols=True):
        batch = texts[i : i + batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out = encoder(**toks)
        if hasattr(out, "last_hidden_state"):
            if use_cls:
                vec = out.last_hidden_state[:, 0, :]
            else:
                attn = toks["attention_mask"].unsqueeze(-1)
                sum_h = (out.last_hidden_state * attn).sum(1)
                vec = sum_h / attn.sum(1).clamp(min=1)
        else:
            vec = out.pooler_output
        embs.append(vec.detach().cpu())
    return torch.cat(embs, dim=0).numpy()


def encode_batch(tokenizer, encoder, texts: List[str], device, max_len: int = 128, pooling: str = "cls"):
    toks = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    out = encoder(**toks)
    if hasattr(out, "last_hidden_state"):
        if pooling == "cls":
            vec = out.last_hidden_state[:, 0, :]
        else:
            attn = toks["attention_mask"].unsqueeze(-1)
            sum_h = (out.last_hidden_state * attn).sum(1)
            vec = sum_h / attn.sum(1).clamp(min=1)
    else:
        vec = out.pooler_output
    return vec


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        for p in (self.weight, self.bias):
            if p is not None:
                p.requires_grad = False
        self.A = nn.Parameter(torch.zeros((r, self.in_features)))
        self.B = nn.Parameter(torch.zeros((self.out_features, r)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = (self.B @ (self.A @ x.transpose(-2, -1))).transpose(-2, -1)
        return base + self.dropout(lora) * self.scaling


def apply_lora_to_encoder(encoder: nn.Module, target_keywords: List[str], r: int, alpha: int, dropout: float):
    repl = 0
    for _, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(k in child_name.lower() for k in target_keywords):
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, wrapped)
                repl += 1
    print(
        f"[LoRA] injected into {repl} Linear layers "
        f"(targets={target_keywords}, r={r}, alpha={alpha}, dropout={dropout})"
    )


def _get_transformer_layers(model: nn.Module):
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder") and hasattr(model.roberta.encoder, "layer"):
        return model.roberta.encoder.layer
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return model.transformer.layer
    return None


def set_finetune_scope(encoder: nn.Module, unfreeze_last_n: int, unfreeze_emb: bool) -> None:
    for p in encoder.parameters():
        p.requires_grad = False

    layers = _get_transformer_layers(encoder)
    if layers is None:
        print("[warn] could not locate transformer layers; keeping encoder frozen")
        return

    if unfreeze_last_n <= 0 or unfreeze_last_n >= len(layers):
        for p in encoder.parameters():
            p.requires_grad = True
    else:
        for block in layers[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

    if unfreeze_emb:
        if hasattr(encoder, "embeddings"):
            for p in encoder.embeddings.parameters():
                p.requires_grad = True
        if hasattr(encoder, "roberta") and hasattr(encoder.roberta, "embeddings"):
            for p in encoder.roberta.embeddings.parameters():
                p.requires_grad = True
        if hasattr(encoder, "distilbert") and hasattr(encoder.distilbert, "embeddings"):
            for p in encoder.distilbert.embeddings.parameters():
                p.requires_grad = True
        for m in encoder.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="bpr_bert")
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--text_hidden", type=int, default=256)
    parser.add_argument("--id_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for BPR head / embeddings")
    parser.add_argument("--encoder_lr", type=float, default=5e-5)
    parser.add_argument("--encoder_weight_decay", type=float, default=0.0)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=EVAL_TOPK)
    parser.add_argument("--eval_cand_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rebuild_training_cache", type=int, default=0)
    parser.add_argument("--rebuild_embedding_cache", type=int, default=0)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument(
        "--tune_mode",
        type=str,
        choices=["frozen", "full", "lora"],
        default="frozen",
        help="frozen: offline cache; full/lora: online encoding and finetune",
    )
    parser.add_argument("--unfreeze_last_n", type=int, default=0)
    parser.add_argument("--unfreeze_emb", type=int, default=0)
    parser.add_argument("--grad_ckpt", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", type=str, default="query,key,value,dense")
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

    (
        q_ids,
        q_texts,
        tool_names,
        a_ids,
        a_texts,
        a_tool_lists,
    ) = build_transformer_corpora(all_agents, all_questions, tools)

    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(
        a_ids, a_tool_lists, tool_names
    )

    qid2idx = {qid: i for i, qid in enumerate(q_ids)}
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}

    qids_in_rank = qids_with_rankings(q_ids, all_rankings)
    print(f"Questions with rankings: {len(qids_in_rank)} / {len(q_ids)}")

    cache_dir = ensure_cache_dir(args.data_root, args.exp_name)
    transformer_cache_dir = ensure_transformer_cache_dir(cache_dir)

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

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    encoder = AutoModel.from_pretrained(args.pretrained_model).to(device)

    if args.grad_ckpt:
        try:
            encoder.gradient_checkpointing_enable()
            print("[encoder] gradient checkpointing enabled")
        except Exception as exc:
            print(f"[encoder] gradient checkpointing not supported: {exc}")

    if args.tune_mode == "lora":
        targets = [s.strip().lower() for s in args.lora_targets.split(",") if s.strip()]
        apply_lora_to_encoder(
            encoder,
            targets,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    if args.tune_mode == "full":
        set_finetune_scope(encoder, unfreeze_last_n=args.unfreeze_last_n, unfreeze_emb=bool(args.unfreeze_emb))
    elif args.tune_mode == "frozen":
        for p in encoder.parameters():
            p.requires_grad = False

    use_embedding_cache = args.tune_mode == "frozen"
    Q_emb = A_emb = None

    if use_embedding_cache:
        if transformer_cache_exists(transformer_cache_dir) and args.rebuild_embedding_cache == 0:
            (
                q_ids_c,
                a_ids_c,
                tool_names_c,
                Q_emb,
                A_emb,
                agent_tool_idx_padded_np,
                agent_tool_mask_np,
                enc_meta,
            ) = load_transformer_cache(transformer_cache_dir)
            if (
                q_ids_c == q_ids
                and a_ids_c == a_ids
                and tool_names_c == tool_names
                and enc_meta.get("pretrained_model") == args.pretrained_model
                and enc_meta.get("max_len") == args.max_len
                and enc_meta.get("pooling", "cls") == args.pooling
            ):
                print(f"[cache] loaded transformer embeddings from {transformer_cache_dir}")
                agent_tool_idx_padded = torch.from_numpy(agent_tool_idx_padded_np)
                agent_tool_mask = torch.from_numpy(agent_tool_mask_np)
            else:
                print("[cache] transformer cache mismatch; rebuilding embeddings...")
                Q_emb = A_emb = None

        if Q_emb is None or A_emb is None:
            encoder.eval()
            with torch.no_grad():
                Q_emb = encode_texts(
                    q_texts,
                    tokenizer,
                    encoder,
                    device,
                    max_len=args.max_len,
                    batch_size=256,
                    pooling=args.pooling,
                )
                A_emb = encode_texts(
                    a_texts,
                    tokenizer,
                    encoder,
                    device,
                    max_len=args.max_len,
                    batch_size=256,
                    pooling=args.pooling,
                )
            save_transformer_cache(
                transformer_cache_dir,
                q_ids,
                a_ids,
                tool_names,
                Q_emb,
                A_emb,
                agent_tool_idx_padded.cpu().numpy()
                if torch.is_tensor(agent_tool_idx_padded)
                else agent_tool_idx_padded,
                agent_tool_mask.cpu().numpy() if torch.is_tensor(agent_tool_mask) else agent_tool_mask,
                enc_meta={
                    "pretrained_model": args.pretrained_model,
                    "max_len": args.max_len,
                    "pooling": args.pooling,
                },
            )
            print(f"[cache] saved transformer embeddings to {transformer_cache_dir}")
    else:
        encoder.train()

    if use_embedding_cache:
        d_text = int(Q_emb.shape[1])
    else:
        encoder.eval()
        with torch.no_grad():
            tmp = encode_batch(tokenizer, encoder, ["hello"], device, max_len=args.max_len, pooling=args.pooling)
            d_text = int(tmp.shape[1])
        encoder.train()

    model = SimpleBPRDNN(
        d_q=d_text,
        d_a=d_text,
        num_agents=len(a_ids),
        num_tools=len(tool_names),
        agent_tool_indices_padded=agent_tool_idx_padded.to(device),
        agent_tool_mask=agent_tool_mask.to(device),
        text_hidden=args.text_hidden,
        id_dim=args.id_dim,
    ).to(device)

    head_params = list(model.parameters())
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": args.lr}]
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": args.encoder_lr,
                "weight_decay": args.encoder_weight_decay,
            }
        )
    optimizer = torch.optim.Adam(param_groups)

    pairs = pairs_idx_np.tolist()
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    if use_embedding_cache:
        Q_t = torch.tensor(Q_emb, dtype=torch.float32, device=device)
        A_t = torch.tensor(A_emb, dtype=torch.float32, device=device)
    else:
        Q_t = A_t = None

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", leave=True, dynamic_ncols=True)
        model.train()
        for b in pbar:
            batch = pairs[b * args.batch_size : (b + 1) * args.batch_size]
            if not batch:
                continue
            q_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            pos_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            neg_idx = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            if use_embedding_cache:
                q_vec = Q_t[q_idx]
                pos_vec = A_t[pos_idx]
                neg_vec = A_t[neg_idx]
            else:
                uniq_q, inv_q = torch.unique(q_idx, sorted=True, return_inverse=True)
                uniq_pos, inv_pos = torch.unique(pos_idx, sorted=True, return_inverse=True)
                uniq_neg, inv_neg = torch.unique(neg_idx, sorted=True, return_inverse=True)

                q_text_batch = [q_texts[i] for i in uniq_q.tolist()]
                pos_text_batch = [a_texts[i] for i in uniq_pos.tolist()]
                neg_text_batch = [a_texts[i] for i in uniq_neg.tolist()]

                q_vec_uniq = encode_batch(
                    tokenizer, encoder, q_text_batch, device, max_len=args.max_len, pooling=args.pooling
                )
                pos_vec_uniq = encode_batch(
                    tokenizer, encoder, pos_text_batch, device, max_len=args.max_len, pooling=args.pooling
                )
                neg_vec_uniq = encode_batch(
                    tokenizer, encoder, neg_text_batch, device, max_len=args.max_len, pooling=args.pooling
                )

                q_vec = q_vec_uniq[inv_q]
                pos_vec = pos_vec_uniq[inv_pos]
                neg_vec = neg_vec_uniq[inv_neg]

            pos, neg = model(q_vec, pos_vec, neg_vec, pos_idx, neg_idx)
            loss = bpr_loss(pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total_loss / (b + 1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total_loss / num_batches if num_batches else 0.0):.4f}")

    if not use_embedding_cache:
        encoder.eval()
        with torch.no_grad():
            Q_emb_eval = encode_texts(
                q_texts,
                tokenizer,
                encoder,
                device,
                max_len=args.max_len,
                batch_size=256,
                pooling=args.pooling,
            )
            A_emb_eval = encode_texts(
                a_texts,
                tokenizer,
                encoder,
                device,
                max_len=args.max_len,
                batch_size=256,
                pooling=args.pooling,
            )
        Q_t = torch.from_numpy(Q_emb_eval).to(device)
        A_t = torch.from_numpy(A_emb_eval).to(device)

    model_dir = os.path.join(cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    data_sig = want_meta["data_sig"]

    ckpt_path = os.path.join(model_dir, f"{args.exp_name}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{args.exp_name}_{data_sig}.json")

    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dims": {
            "d_text": int(Q_t.shape[1]),
            "num_agents": len(a_ids),
            "num_tools": len(tool_names),
            "text_hidden": args.text_hidden,
            "id_dim": args.id_dim,
        },
        "mappings": {"q_ids": q_ids, "a_ids": a_ids, "tool_names": tool_names},
        "args": vars(args),
        "pretrained_model": args.pretrained_model,
        "tune_mode": args.tune_mode,
    }
    torch.save(ckpt, ckpt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data_sig": data_sig, "q_ids": q_ids, "a_ids": a_ids}, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {ckpt_path}")
    print(f"[save] meta  -> {meta_path}")

    model.eval()

    overall_metrics = evaluate_sampled_embedding_topk(
        model=model,
        qid2idx=qid2idx,
        aid2idx=aid2idx,
        all_rankings=all_rankings,
        eval_qids=valid_qids,
        Q_t=Q_t,
        A_t=A_t,
        cand_size=args.eval_cand_size,
        pos_topk=POS_TOPK,
        topk=int(args.topk),
        seed=123,
        desc=f"Valid Overall (transformer, top{int(args.topk)})",
    )
    print_metrics_table("Validation Overall (transformer)", overall_metrics, ks=(int(args.topk),), filename=args.exp_name)

    part_splits = split_eval_qids_by_part(valid_qids, qid_to_part=qid_to_part)
    for part in ["PartI", "PartII", "PartIII"]:
        qids_part = part_splits.get(part, [])
        if not qids_part:
            continue
        m_part = evaluate_sampled_embedding_topk(
            model=model,
            qid2idx=qid2idx,
            aid2idx=aid2idx,
            all_rankings=all_rankings,
            eval_qids=qids_part,
            Q_t=Q_t,
            A_t=A_t,
            cand_size=args.eval_cand_size,
            pos_topk=POS_TOPK,
            topk=int(args.topk),
            seed=123,
            desc=f"Valid {part} (transformer, top{int(args.topk)})",
        )
        print_metrics_table(
            f"Validation {part} (transformer)", m_part, ks=(int(args.topk),), filename=args.exp_name
        )


if __name__ == "__main__":
    main()
