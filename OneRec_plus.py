#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OneRec++ (fixed): Candidate-only logits + CPU TF-IDF retrieval + Session-aware + Lite-DPO

Key fixes (for huge vocab / huge agent count):
  1) NEVER build full (B,L,V) logits. Use candidate-only logits (B,L,C).
  2) Candidate retrieval (train_mask) moved to CPU sklearn NearestNeighbors on sparse CSR.
  3) Evaluation / RM sampling / DPO logprob all use candidate-only sets.
  4) Proper AMP via GradScaler for stability.

Dataset:
  {data_root}/PartI|PartII|PartIII/{agents,questions,rankings}/merge.json + Tools/merge.json
Optional:
  {data_root}/sessions.json  # [{"qid": qid, "history": [prev_qid1,...]}]
"""

from agent_rec.features import load_feature_cache, build_agent_content_view
from agent_rec.config import (
    EVAL_TOPK,
    TFIDF_MAX_FEATURES,
    pos_topk_for_qid,
)
from agent_rec.data import stratified_train_valid_split
from agent_rec.run_common import set_global_seed, warn_if_topk_diff

import os, json, math, argparse, random, zlib, copy
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from utils import print_metrics_table

filename = os.path.splitext(os.path.basename(__file__))[0]



from collections import defaultdict

def stratified_split_by_part(
    qids: List[str],
    qid_to_part: Dict[str, str],
    valid_ratio: float,
    seed: int
) -> Tuple[List[str], List[str]]:
    # Delegate to shared splitter to align with other pipelines (ensures per-part minimums).
    return stratified_train_valid_split(qids, qid_to_part=qid_to_part, valid_ratio=valid_ratio, seed=seed)


def sample_qids_by_part(
    qids: List[str],
    qid_to_part: Dict[str, str],
    per_part: int,
    seed: int,
    parts: Optional[List[str]] = None
) -> List[str]:
    """
    per_part=200：每个 part 最多取 200；不足则全取。
    parts=None：自动用出现过的 part；你也可以传 ["PartI","PartII","PartIII"].
    """
    if per_part <= 0:
        return list(qids)

    rng = random.Random(seed)
    buckets = defaultdict(list)
    for q in qids:
        buckets[qid_to_part.get(q, "Unknown")].append(q)

    if parts is None:
        parts = sorted(buckets.keys())

    out = []
    for p in parts:
        lst = buckets.get(p, [])
        rng.shuffle(lst)
        out.extend(lst[: min(per_part, len(lst))])

    # 最后再 shuffle 一下，避免 part 顺序偏置
    rng.shuffle(out)
    return out

# ---------------------- I/O helpers ----------------------

def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}")
    os.makedirs(d, exist_ok=True)
    return d

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_data(data_root: str):
    parts = ["PartI", "PartII", "PartIII"]
    all_agents: Dict[str, dict] = {}
    all_questions: Dict[str, dict] = {}
    all_rankings: Dict[str, List[str]] = {}
    qid_to_part: Dict[str, str] = {}

    for part in parts:
        agents_path = os.path.join(data_root, part, "agents", "merge.json")
        questions_path = os.path.join(data_root, part, "questions", "merge.json")
        rankings_path = os.path.join(data_root, part, "rankings", "merge.json")

        if os.path.exists(agents_path):
            all_agents.update(load_json(agents_path))

        if os.path.exists(questions_path):
            qd = load_json(questions_path)
            all_questions.update(qd)
            # 标注 qid 来源 part（后写的不覆盖先写的也行；你也可以反过来）
            for qid in qd.keys():
                qid_to_part.setdefault(qid, part)

        if os.path.exists(rankings_path):
            r = load_json(rankings_path)
            rr = r.get("rankings", {})
            all_rankings.update(rr)
            # rankings 里的 qid 也补标（有些数据可能 questions/merge 不全）
            for qid in rr.keys():
                qid_to_part.setdefault(qid, part)

    tools_path = os.path.join(data_root, "Tools", "merge.json")
    tools = load_json(tools_path) if os.path.exists(tools_path) else {}
    return all_agents, all_questions, all_rankings, tools, qid_to_part


# ---------------------- Text & TF-IDF ----------------------

def build_text_corpora(all_agents, all_questions, tools):
    q_ids = sorted(all_questions.keys())
    q_texts = [all_questions[qid].get("input", "") for qid in q_ids]

    tool_names = sorted(tools.keys())
    def _tool_text(tn: str) -> str:
        t = tools.get(tn, {})
        desc = t.get("description", "")
        return f"{tn} {desc}".strip()

    a_ids = sorted(all_agents.keys())
    a_texts = []
    a_tool_lists = []
    for aid in a_ids:
        a = all_agents[aid]
        mname = a.get("M", {}).get("name", "")
        tool_list = a.get("T", {}).get("tools", []) or []
        a_tool_lists.append(tool_list)
        concat_tool_desc = " ".join([_tool_text(tn) for tn in tool_list])
        text = f"{mname} {concat_tool_desc}".strip()
        a_texts.append(text)

    return q_ids, q_texts, a_ids, a_texts, a_tool_lists

def build_q_vectorizer(q_texts, max_features: int):
    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    Q_csr = q_vec.fit_transform(q_texts)
    return q_vec, Q_csr

# ---------------------- Sessions ----------------------

def load_sessions_or_build(
    data_root: str,
    q_ids: List[str],
    Q_csr,
    use_sessions: bool,
    session_len: int,
    train_qids: List[str],
    nn_extra: int = 8,
    nn_batch: int = 10000,
    n_jobs: int = 4,
) -> Dict[str, List[str]]:
    """
    If sessions.json exists: use it.
    Else: build pseudo sessions by NN among train questions in TF-IDF space.
    """
    sess_path = os.path.join(data_root, "sessions.json")
    if use_sessions and os.path.exists(sess_path):
        try:
            raw = load_json(sess_path)
            out = {}
            for it in raw:
                qid = it.get("qid"); hist = it.get("history", [])
                if qid in q_ids:
                    out[qid] = hist[-session_len:]
            return out
        except Exception:
            pass

    if not train_qids or session_len <= 0:
        return {qid: [] for qid in q_ids}

    from scipy.sparse import issparse
    if not issparse(Q_csr):
        raise ValueError("Q_csr must be scipy.sparse CSR/CSC.")

    Qn = normalize(Q_csr.tocsr().astype(np.float32), norm="l2", axis=1, copy=True)

    qid2idx = {qid: i for i, qid in enumerate(q_ids)}
    train_idx = np.array([qid2idx[q] for q in train_qids if q in qid2idx], dtype=np.int64)
    if train_idx.size == 0:
        return {qid: [] for qid in q_ids}

    k_query = min(session_len + nn_extra, train_idx.size)
    nbrs = NearestNeighbors(
        n_neighbors=k_query,
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs
    ).fit(Qn[train_idx])

    out = {}
    N = len(q_ids)
    for s in range(0, N, nn_batch):
        e = min(s + nn_batch, N)
        dist, idx_local = nbrs.kneighbors(Qn[s:e], return_distance=True)
        idx_global = train_idx[idx_local]

        for i, qrow in enumerate(range(s, e)):
            qid = q_ids[qrow]
            cand = idx_global[i].tolist()
            hist = []
            seen = set()
            for gidx in cand:
                if gidx == qrow:
                    continue
                qh = q_ids[gidx]
                if qh in seen:
                    continue
                seen.add(qh)
                hist.append(qh)
                if len(hist) >= session_len:
                    break
            out[qid] = hist

    for qid in q_ids:
        out.setdefault(qid, [])
    return out

# ---------------------- Metrics ----------------------

def _dcg_at_k(binary_hits, k):
    dcg = 0.0
    for i, h in enumerate(binary_hits[:k]):
        if h:
            dcg += 1.0 / math.log2(i + 2.0)
    return dcg

@torch.no_grad()
def evaluate_sampled(pred_ids_topk: List[str], rel_set: set, ks=(5, 10, 50)):
    bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids_topk]
    out = {}
    for k in ks:
        Hk = sum(bin_hits[:k])
        P = Hk / float(k)
        R = Hk / float(len(rel_set)) if len(rel_set) > 0 else 0.0
        F1 = (2*P*R)/(P+R) if (P+R) > 0 else 0.0
        Hit = 1.0 if Hk > 0 else 0.0
        dcg = _dcg_at_k(bin_hits, k)
        ideal = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal > 0 else 0.0
        nDCG = (dcg / idcg) if idcg > 0 else 0.0
        rr = 0.0
        for i in range(k):
            if bin_hits[i]:
                rr = 1.0 / float(i+1)
                break
        out[k] = {"P":P, "R":R, "F1":F1, "Hit":Hit, "nDCG":nDCG, "MRR":rr}
    return out

# ---------------------- Vocab ----------------------

class AgentVocab:
    def __init__(self, a_ids: List[str]):
        self.PAD = 0; self.BOS = 1; self.offset = 2
        self.a_ids = a_ids
        self.vocab_size = len(a_ids) + self.offset
        self.aid2tok = {aid: i + self.offset for i, aid in enumerate(a_ids)}
        self.tok2aid = {i + self.offset: aid for i, aid in enumerate(a_ids)}
    def aid_to_token(self, aid: str) -> int:
        return self.aid2tok[aid]
    def token_to_aid(self, tok: int) -> Optional[str]:
        if tok < self.offset: return None
        return self.tok2aid.get(tok, None)

# ---------------------- Remap CE targets ----------------------

def remap_targets_to_cand(
    tgt_tok: torch.Tensor,      # (B,L) full-vocab token ids
    cand_tok: torch.Tensor,     # (B,C) candidate token ids
    pad_tok: int = 0,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Return local indices in [0..C-1], else ignore_index.
    This uses broadcasting match (B,L,C) but is safe for typical B<=1024, L<=64, C<=2k.
    """
    # (B,L,C)
    match = (tgt_tok.unsqueeze(-1) == cand_tok.unsqueeze(1))
    has = match.any(dim=-1)  # (B,L)
    local = match.float().argmax(dim=-1)  # (B,L), arbitrary if no match
    local = torch.where(has, local, torch.full_like(local, ignore_index))
    local = torch.where(tgt_tok == pad_tok, torch.full_like(local, ignore_index), local)
    return local

# ---------------------- Models ----------------------

class SessionEncoder(nn.Module):
    """
    Encodes (B, Lq, Dq) TF-IDF vectors into (B, H).
    """
    def __init__(self, d_q: int, vocab_size: int, tok_dim: int = 256, hidden: int = 512, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.q_proj = nn.Linear(d_q, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.tok_emb = nn.Embedding(vocab_size, tok_dim)  # optional, currently unused unless you pass a_hist_tok
        self.mix = nn.Linear(hidden + tok_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        for m in [self.q_proj, self.mix]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, q_hist_vec: torch.Tensor, a_hist_tok: Optional[torch.Tensor] = None):
        hq = self.q_proj(q_hist_vec)   # (B,L,H)
        h = self.enc(hq)               # (B,L,H)
        h_last = h[:, -1]              # (B,H)
        if a_hist_tok is not None:
            ae = self.tok_emb(a_hist_tok).mean(dim=1)  # (B,E)
            h_cat = torch.cat([h_last, ae], dim=1)
            h_last = self.norm(torch.relu(self.mix(h_cat)))
        return h_last

class OneRecPlus(nn.Module):
    """
    Generator with tied token embeddings. Supports candidate-only logits.
    """
    def __init__(self, enc_dim: int, vocab_size: int, hidden: int = 512, num_layers: int = 2, tok_dim: int = 256):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, tok_dim)
        self.enc_to_h = nn.Linear(enc_dim, hidden)
        self.dec = nn.GRU(input_size=tok_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.bridge = nn.Linear(hidden, tok_dim, bias=False)

        nn.init.xavier_uniform_(self.tok_emb.weight)
        nn.init.xavier_uniform_(self.enc_to_h.weight); nn.init.zeros_(self.enc_to_h.bias)
        nn.init.xavier_uniform_(self.bridge.weight)

        with torch.no_grad():
            self.tok_emb.weight.data[0].zero_()
            self.tok_emb.weight.data[1].zero_()

    def forward(self, enc_vec: torch.Tensor, tgt_inp_tok: torch.Tensor, cand_tok: torch.Tensor):
        """
        cand_tok: (B,C) candidate token ids
        returns logits: (B,L,C)
        """
        B, L = tgt_inp_tok.shape
        h0 = self.enc_to_h(enc_vec).unsqueeze(0).repeat(self.num_layers, 1, 1)
        x = self.tok_emb(tgt_inp_tok)   # (B,L,E)
        out, _ = self.dec(x, h0)        # (B,L,H)
        z = self.bridge(out)            # (B,L,E)
        Wc = self.tok_emb(cand_tok)     # (B,C,E)
        logits_c = torch.einsum("ble,bce->blc", z, Wc)
        return logits_c

    @torch.no_grad()
    def generate(self, enc_vec: torch.Tensor, topk: int, cand_tok: torch.Tensor, temperature: float = 0.0):
        """
        Candidate-only generation: only rank/select within cand_tok (B,C).
        """
        was_training = self.training
        self.eval()
        try:
            B = enc_vec.size(0)
            h = self.enc_to_h(enc_vec).unsqueeze(0).repeat(self.num_layers, 1, 1)
            prev = torch.full((B, 1), 1, dtype=torch.long, device=enc_vec.device)  # BOS

            C = cand_tok.size(1)
            used_c = torch.zeros((B, C), dtype=torch.bool, device=enc_vec.device)
            used_c |= (cand_tok < 2)  # mask PAD/BOS if present

            # precompute candidate embeddings once
            Wc = self.tok_emb(cand_tok)  # (B,C,E)

            out_tokens = []
            for _ in range(topk):
                x = self.tok_emb(prev)
                o, h = self.dec(x, h)
                z = self.bridge(o[:, -1])                 # (B,E)
                logits = torch.einsum("be,bce->bc", z, Wc)  # (B,C)
                logits = logits.masked_fill(used_c, float("-inf"))

                if temperature and temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_pos = torch.multinomial(probs, 1).squeeze(1)
                else:
                    next_pos = torch.argmax(logits, dim=-1)

                next_tok = cand_tok.gather(1, next_pos.unsqueeze(1)).squeeze(1)
                out_tokens.append(next_tok)
                prev = next_tok.unsqueeze(1)
                used_c.scatter_(1, next_pos.unsqueeze(1), True)

            return torch.stack(out_tokens, dim=1)
        finally:
            self.train(was_training)

# ---------------------- Agent token init (optional) ----------------------

class AgentTokenComposer(nn.Module):
    """
    Build agent token embeddings from content vector + (llm_id emb + mean(tool_id emb)).
    """
    def __init__(self, d_content: int, num_tools: int, num_llm: int, tok_dim: int, id_dim: int = 256):
        super().__init__()
        self.emb_tool = nn.Embedding(num_tools, id_dim)
        self.emb_llm  = nn.Embedding(num_llm, id_dim)
        self.proj_content = nn.Linear(d_content, tok_dim, bias=False)
        self.proj_ids     = nn.Linear(id_dim * 2, tok_dim, bias=False)
        nn.init.xavier_uniform_(self.emb_tool.weight)
        nn.init.xavier_uniform_(self.emb_llm.weight)
        nn.init.xavier_uniform_(self.proj_content.weight)
        nn.init.xavier_uniform_(self.proj_ids.weight)

    def forward(self, A_content: torch.Tensor, tool_idx_pad: torch.LongTensor, tool_mask: torch.FloatTensor, llm_idx: torch.LongTensor):
        te = self.emb_tool(tool_idx_pad)                 # (Na,T,id_dim)
        m  = tool_mask.unsqueeze(-1)                     # (Na,T,1)
        tool_mean = (te * m).sum(1) / (m.sum(1) + 1e-8)  # (Na,id_dim)
        llm_e = self.emb_llm(llm_idx)                    # (Na,id_dim)
        ids = torch.cat([llm_e, tool_mean], dim=-1)      # (Na,2*id_dim)
        e = self.proj_content(A_content) + self.proj_ids(ids)
        e = F.normalize(e, dim=-1)
        return e

def init_tok_emb_from_bge_plus_ids(
    gen,
    vocab,
    a_ids,
    feature_dir,
    device,
    freeze_tok=False,
    max_tool_per_agent: int = 8,
    init_chunk: int = 50000,
    use_fp16: bool = True,
):
    """
    Fixes:
      - tool pad length fixed to max_tool_per_agent (default 8)
      - chunked initialization to avoid OOM (no full Na tensors on GPU)
    """
    cache = load_feature_cache(feature_dir)

    # content vectors (CPU numpy)
    A_content_np = build_agent_content_view(
        cache=cache,
        use_model_content_vector=True,
        use_tool_content_vector=True,
    ).astype(np.float32)  # (Na_cache, d_content)

    cache_a_ids = cache.a_ids
    idx_map = {aid: i for i, aid in enumerate(cache_a_ids)}

    # raw padded tool arrays from cache (CPU numpy / memmap likely)
    tool_pad = cache.agent_tool_idx_padded    # (Na_cache, T_cache)
    tool_msk = cache.agent_tool_mask          # (Na_cache, T_cache)
    llm_idx  = cache.agent_llm_idx            # (Na_cache,)

    # ---- align to current a_ids on CPU ----
    Na = len(a_ids)
    d_content = A_content_np.shape[1]

    # IMPORTANT: do NOT allocate huge GPU tensors here.
    A_aligned = np.zeros((Na, d_content), dtype=np.float32)
    llm_idx_aligned = np.zeros((Na,), dtype=np.int64)

    # fixed tool pad length = 8
    T = int(max_tool_per_agent)
    tool_pad_aligned = np.zeros((Na, T), dtype=np.int64)
    tool_msk_aligned = np.zeros((Na, T), dtype=np.float32)

    missing = 0
    for i, aid in enumerate(a_ids):
        j = idx_map.get(aid, None)
        if j is None:
            missing += 1
            continue
        A_aligned[i] = A_content_np[j]
        llm_idx_aligned[i] = int(llm_idx[j])

        # truncate/pad to T=8 (keep order)
        tp = np.asarray(tool_pad[j], dtype=np.int64)
        tm = np.asarray(tool_msk[j], dtype=np.float32)

        if tp.ndim != 1:
            tp = tp.reshape(-1)
        if tm.ndim != 1:
            tm = tm.reshape(-1)

        tp = tp[:T]
        tm = tm[:T]
        tool_pad_aligned[i, :len(tp)] = tp
        tool_msk_aligned[i, :len(tm)] = tm

    if missing:
        print(f"[warn] {missing} agents missing in cache alignment.")

    num_tools = len(cache.tool_id_vocab)
    num_llm   = len(cache.llm_vocab)
    tok_dim   = gen.tok_emb.embedding_dim

    composer = AgentTokenComposer(
        d_content=d_content, num_tools=num_tools, num_llm=num_llm, tok_dim=tok_dim, id_dim=256
    ).to(device)

    # optional freeze tok embedding after init
    if freeze_tok:
        for p in gen.tok_emb.parameters():
            p.requires_grad = False

    # ---- chunked init on GPU ----
    was_training_gen = gen.training
    gen.eval()
    composer.eval()
    dtype_ctx = torch.cuda.amp.autocast(enabled=(use_fp16 and device.type == "cuda"))

    try:
        with torch.no_grad():
            for s in tqdm(range(0, Na, init_chunk), desc="[tok-init] chunked", total=(Na + init_chunk - 1)//init_chunk):
                e = min(Na, s + init_chunk)

                A_t = torch.from_numpy(A_aligned[s:e]).to(device, non_blocking=True)
                tool_pad_t = torch.from_numpy(tool_pad_aligned[s:e]).to(device, non_blocking=True)
                tool_msk_t = torch.from_numpy(tool_msk_aligned[s:e]).to(device, non_blocking=True)
                llm_idx_t  = torch.from_numpy(llm_idx_aligned[s:e]).to(device, non_blocking=True)

                with dtype_ctx:
                    E_agents = composer(A_t, tool_pad_t, tool_msk_t, llm_idx_t)  # (chunk, tok_dim)

                # write into gen.tok_emb.weight
                gen.tok_emb.weight.data[vocab.offset + s : vocab.offset + e] = E_agents.to(gen.tok_emb.weight.dtype)

                # keep PAD/BOS zeros
                gen.tok_emb.weight.data[vocab.PAD].zero_()
                gen.tok_emb.weight.data[vocab.BOS].zero_()

                # explicit free
                del A_t, tool_pad_t, tool_msk_t, llm_idx_t, E_agents
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        gen.train(was_training_gen)
        composer.train()

    print(f"[tok-init] done. tool_pad=T={T}, chunk={init_chunk}, fp16={use_fp16}")
    return composer


# ---------------------- Reward Model & Lite-DPO ----------------------

class RewardModel(nn.Module):
    def __init__(self, enc_dim: int, tok_emb: nn.Embedding, hidden: int = 512, freeze_tok: bool = True):
        super().__init__()
        if freeze_tok:
            # snapshot the embedding space used to pretrain RM to avoid drift during DPO finetune
            self.tok_emb = nn.Embedding.from_pretrained(tok_emb.weight.detach().clone(), freeze=True)
        else:
            self.tok_emb = tok_emb
        self.ff = nn.Sequential(
            nn.Linear(enc_dim + tok_emb.embedding_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        for m in self.ff:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, enc_vec: torch.Tensor, tok_seq: torch.Tensor):
        te = self.tok_emb(tok_seq)  # (B,L,E)
        mask = (tok_seq >= 2).float().unsqueeze(-1)
        te_sum = (te * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        te_mean = te_sum / denom
        x = torch.cat([enc_vec, te_mean], dim=1)
        return self.ff(x).squeeze(1)

class DPOTrainer:
    def __init__(self, model: OneRecPlus, ref_model: OneRecPlus, rm: RewardModel, beta: float = 0.1, pad_tok: int = 0):
        self.model = model
        self.ref_model = ref_model
        self.rm = rm
        self.beta = beta
        self.pad_tok = pad_tok

    def seq_logprob_candidate_only(self, enc_vec: torch.Tensor, seq: torch.Tensor, cand_tok: torch.Tensor):
        """
        seq: (B,L) full-vocab token ids (agents)
        cand_tok: (B,C)
        """
        B, L = seq.shape
        bos = torch.full((B, 1), 1, dtype=torch.long, device=seq.device)
        inp = torch.cat([bos, seq[:, :-1]], dim=1)  # teacher forcing input

        logits_c = self.model(enc_vec, inp, cand_tok=cand_tok)      # (B,L,C)
        lp = F.log_softmax(logits_c, dim=-1)                        # (B,L,C)

        local = remap_targets_to_cand(seq, cand_tok, pad_tok=self.pad_tok, ignore_index=-100)  # (B,L)
        local_safe = local.clamp_min(0)
        gather = lp.gather(-1, local_safe.unsqueeze(-1)).squeeze(-1)  # (B,L)
        mask = (local != -100).float()
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        lp_sum = (gather * mask).sum(dim=1)
        return lp_sum / token_counts

    def dpo_loss(self, enc_vec: torch.Tensor, pref_seq: torch.Tensor, nonpref_seq: torch.Tensor, cand_tok: torch.Tensor):
        # policy log-prob
        lp_pref = self.seq_logprob_candidate_only(enc_vec, pref_seq, cand_tok)
        lp_nonp = self.seq_logprob_candidate_only(enc_vec, nonpref_seq, cand_tok)
        # frozen reference log-prob
        with torch.no_grad():
            lp_pref_ref = self.seq_logprob_candidate_only(self.ref_model, enc_vec, pref_seq, cand_tok)
            lp_nonp_ref = self.seq_logprob_candidate_only(self.ref_model, enc_vec, nonpref_seq, cand_tok)

        logratio = (lp_pref - lp_pref_ref) - (lp_nonp - lp_nonp_ref)
        return -F.logsigmoid(self.beta * logratio).mean()

# ---------------------- Split / signature ----------------------

def train_valid_split(qids_in_rank, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    q = list(qids_in_rank); rng.shuffle(q)
    n_valid = int(len(q) * valid_ratio)
    return q[n_valid:], q[:n_valid]

def dataset_signature(a_ids: List[str], all_rankings: Dict[str, List[str]]) -> str:
    payload = {"a_ids": a_ids, "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sig = zlib.crc32(blob) & 0xFFFFFFFF
    return f"{sig:08x}"

# ---------------------- Evaluation (candidate-only) ----------------------

@torch.no_grad()
def evaluate_model_gen(
    gen_model: OneRecPlus,
    enc_vecs_cpu: torch.Tensor,         # (Nq,H) on CPU
    qid2idx: Dict[str,int],
    a_ids: List[str],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    device: torch.device,
    ks=(5, 10, 50),
    cand_size: int = 1000,
    rng_seed: int = 123,
    ref_k: Optional[int] = None,
    bar_update_every: int = 1,
    qid_to_part: Optional[Dict[str, str]] = None
):
    if ref_k is None:
        ref_k = max(ks)

    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt = 0
    skipped = 0

    all_agent_set = set(a_ids)
    rnd = random.Random(rng_seed)
    vocab = AgentVocab(a_ids)

    pbar = tqdm(eval_qids, desc="Evaluating (gen, sampled)", total=len(eval_qids))
    for i, qid in enumerate(pbar, start=1):
        k_pos = pos_topk_for_qid(qid, qid_to_part)
        gt_list = [aid for aid in all_rankings.get(qid, [])[:k_pos] if aid in all_agent_set]
        if not gt_list:
            skipped += 1
            if (i % bar_update_every) == 0:
                done = cnt
                if cnt > 0:
                    ref = agg[ref_k]
                    pbar.set_postfix({
                        "done": done, "skipped": skipped,
                        f"P@{ref_k}": f"{(ref['P']/cnt):.4f}",
                        f"nDCG@{ref_k}": f"{(ref['nDCG']/cnt):.4f}",
                        f"MRR@{ref_k}": f"{(ref['MRR']/cnt):.4f}",
                        "Ncand": 0
                    })
                else:
                    pbar.set_postfix({"done": done, "skipped": skipped, f"P@{ref_k}": "0.0000",
                                      f"nDCG@{ref_k}": "0.0000", f"MRR@{ref_k}": "0.0000",
                                      "Ncand": 0})
            continue

        rel_set = set(gt_list)
        neg_pool = list(all_agent_set - rel_set)
        need_neg = max(0, cand_size - len(gt_list))
        if need_neg > 0 and len(neg_pool) > 0:
            k = min(need_neg, len(neg_pool))
            sampled_negs = rnd.sample(neg_pool, k)
            cand_ids = gt_list + sampled_negs
        else:
            cand_ids = gt_list

        # cand_tok: (1,C)  pad with PAD if needed
        cand_tok_list = [vocab.aid_to_token(a) for a in cand_ids]
        # de-dup preserving order
        seen = set()
        cand_tok_u = []
        for t in cand_tok_list:
            if t in seen:
                continue
            seen.add(t)
            cand_tok_u.append(t)
        # ensure at least topk size
        while len(cand_tok_u) < max(ks):
            cand_tok_u.append(vocab.PAD)
        cand_tok_u = cand_tok_u[:cand_size] if len(cand_tok_u) > cand_size else cand_tok_u
        while len(cand_tok_u) < cand_size:
            cand_tok_u.append(vocab.PAD)
        cand_tok = torch.tensor([cand_tok_u], dtype=torch.long, device=device)

        qi = qid2idx[qid]
        enc = enc_vecs_cpu[qi:qi+1].to(device, non_blocking=True)
        pred_tok = gen_model.generate(enc, topk=max(ks), cand_tok=cand_tok, temperature=0.0)

        pred_ids = []
        for t in pred_tok[0].tolist():
            aid = vocab.token_to_aid(t)
            if aid is not None:
                pred_ids.append(aid)

        md = evaluate_sampled(pred_ids, rel_set, ks)
        for k in ks:
            for m in md[k]:
                agg[k][m] += md[k][m]
        cnt += 1

        if (i % bar_update_every) == 0:
            ref = agg[ref_k]
            pbar.set_postfix({
                "done": cnt,
                "skipped": skipped,
                f"P@{ref_k}": f"{(ref['P']/cnt):.4f}",
                f"nDCG@{ref_k}": f"{(ref['nDCG']/cnt):.4f}",
                f"MRR@{ref_k}": f"{(ref['MRR']/cnt):.4f}",
                "Ncand": len(cand_ids)
            })

    if cnt == 0:
        return {k:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}
    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)

    ap.add_argument("--use_sessions", type=int, default=1)
    ap.add_argument("--session_len", type=int, default=4)

    ap.add_argument("--train_mask", type=int, default=0,
                    help="1: use CPU TF-IDF NN to build per-sample candidate set; 0: random negatives")
    ap.add_argument("--candidate_size", type=int, default=200,
                    help="candidate set size C (must be >= topk)")
    ap.add_argument("--cand_extra", type=int, default=32,
                    help="extra neighbors retrieved to allow dedup + GT insertion")
    ap.add_argument("--eval_candidate_size", type=int, default=200,
                    help="candidate set size for evaluation (sampled metrics)")

    ap.add_argument("--mode", choices=["gen","dpo"], default="gen")
    ap.add_argument("--dpo_steps", type=int, default=1000)
    ap.add_argument("--dpo_batch", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--dpo_lr", type=float, default=None, help="Override LR for DPO; default = lr * 0.1")

    ap.add_argument("--nn_extra", type=int, default=8)
    ap.add_argument("--nn_batch", type=int, default=10000)
    ap.add_argument("--nn_jobs", type=int, default=4)

    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--enc_chunk", type=int, default=512)

    ap.add_argument("--bge_feature_dir", type=str, default=None)
    ap.add_argument("--init_tok_from_bge", type=int, default=1)
    ap.add_argument("--freeze_tok_emb", type=int, default=0)
    ap.add_argument("--eval_per_part", type=int, default=200, help="Per-part eval cap. 0 = no per-part cap.")
    ap.add_argument("--eval_parts", type=str, default="PartI,PartII,PartIII", help="Comma-separated part names to eval.")
    
    ap.add_argument("--max_tool_per_agent", type=int, default=8)
    ap.add_argument("--tok_init_chunk", type=int, default=50000)
    ap.add_argument("--tok_init_fp16", type=int, default=1)
    
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip_eval", type=int, default=0, help="1 to skip validation (saves time)")


    args = ap.parse_args()
    eval_parts = [x.strip() for x in args.eval_parts.split(",") if x.strip()]

    warn_if_topk_diff(args.topk, expected=EVAL_TOPK)
    set_global_seed(args.seed)
    device = torch.device(args.device)
    use_cuda = (device.type == "cuda")
    use_amp = bool(args.amp) and use_cuda

    if args.candidate_size < args.topk:
        raise ValueError(f"candidate_size ({args.candidate_size}) must be >= topk ({args.topk}).")

    # ---------------- load data ----------------
    all_agents, all_questions, all_rankings, tools, qid_to_part = collect_data(args.data_root)
    q_ids, q_texts, a_ids, a_texts, a_tool_lists = build_text_corpora(all_agents, all_questions, tools)
    qid2idx = {qid:i for i,qid in enumerate(q_ids)}
    vocab = AgentVocab(a_ids)

    # ---------------- TF-IDF ----------------
    q_vec, Q_csr = build_q_vectorizer(q_texts, args.max_features)

    # For candidate retrieval: represent each agent in the SAME Q-space via q_vec.transform
    A_in_Q_csr = q_vec.transform(a_texts)  # (Na, Dq) sparse

    # normalize on CPU once
    from scipy.sparse import issparse
    if not issparse(Q_csr) or not issparse(A_in_Q_csr):
        raise ValueError("Q_csr / A_in_Q_csr must be sparse matrices.")

    Qn = normalize(Q_csr.tocsr().astype(np.float32), norm="l2", axis=1, copy=True)
    Aq = normalize(A_in_Q_csr.tocsr().astype(np.float32), norm="l2", axis=1, copy=True)

    # build agent retriever (CPU)
    # NOTE: brute+cosine on CSR is stable; speed depends on Na.
    print("[retriever] fitting NearestNeighbors on agents (CPU)...")
    nbrsA = NearestNeighbors(
        n_neighbors=min(args.candidate_size + args.cand_extra, Aq.shape[0]),
        metric="cosine",
        algorithm="brute",
        n_jobs=args.nn_jobs
    ).fit(Aq)
    print("[retriever] done.")

    def retrieve_agent_topN_indices(qid_batch: List[str], k: int) -> np.ndarray:
        idx = np.array([qid2idx[q] for q in qid_batch], dtype=np.int64)
        k = min(k, Aq.shape[0])
        dist, ind = nbrsA.kneighbors(Qn[idx], n_neighbors=k, return_distance=True)
        return ind  # (B,k) agent indices in [0..Na-1]

    # ---------------- split ----------------
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
    train_qids, valid_qids = stratified_split_by_part(
        qids_in_rank,
        qid_to_part=qid_to_part,
        valid_ratio=args.valid_ratio,
        seed=args.split_seed
    )


    # ---------------- sessions ----------------
    sessions = load_sessions_or_build(
        args.data_root, q_ids, Q_csr,
        bool(args.use_sessions), args.session_len, train_qids,
        nn_extra=args.nn_extra, nn_batch=args.nn_batch, n_jobs=args.nn_jobs
    )

    # dense Q tensor for session encoder input (CPU)
    # if Q is huge for your setting, you can change this to on-demand densify rows.
    Q_dense = Q_csr.toarray().astype(np.float32)
    Q_t_cpu = torch.from_numpy(Q_dense).pin_memory()

    # ---------------- build session tensors ----------------
    def build_session_tensor(qid_batch: List[str]) -> torch.Tensor:
        L = args.session_len if args.use_sessions else 1
        B = len(qid_batch)
        Dq = Q_t_cpu.shape[1]
        out = torch.zeros((B, L, Dq), dtype=Q_t_cpu.dtype)
        for i, qid in enumerate(qid_batch):
            hist = sessions.get(qid, [])[-L:]
            if not hist:
                qi = qid2idx[qid]
                out[i, -1] = Q_t_cpu[qi]
            else:
                start = L - len(hist)
                for j, hq in enumerate(hist):
                    if hq in qid2idx:
                        out[i, start+j] = Q_t_cpu[qid2idx[hq]]
                out[i, -1] = Q_t_cpu[qid2idx[qid]]
        return out

    def build_agent_hist_tokens(qid_batch: List[str]) -> Optional[torch.Tensor]:
        return None

    @torch.no_grad()
    def encode_sessions_in_chunks(qid_batch: List[str], chunk: int = 512) -> torch.Tensor:
        outs = []
        sess_enc.eval()
        for i in range(0, len(qid_batch), chunk):
            qids_s = qid_batch[i:i+chunk]
            h_cpu = build_session_tensor(qids_s)  # CPU
            h = h_cpu.to(device, non_blocking=True)
            enc = sess_enc(h, None).cpu()
            outs.append(enc)
        return torch.cat(outs, dim=0)

    # ---------------- models ----------------
    sess_enc = SessionEncoder(
        d_q=Q_t_cpu.shape[1],
        vocab_size=vocab.vocab_size,
        tok_dim=256, hidden=512, n_heads=8, n_layers=2
    ).to(device)

    gen = OneRecPlus(
        enc_dim=512,
        vocab_size=vocab.vocab_size,
        hidden=512, num_layers=2, tok_dim=256
    ).to(device)

    composer = None
    if args.init_tok_from_bge and args.bge_feature_dir:
        composer = init_tok_emb_from_bge_plus_ids(
            gen, vocab, a_ids, args.bge_feature_dir, device,
            freeze_tok=bool(args.freeze_tok_emb),
            max_tool_per_agent=args.max_tool_per_agent,
            init_chunk=args.tok_init_chunk,
            use_fp16=bool(args.tok_init_fp16),
        )


    if args.freeze_tok_emb:
        for p in gen.tok_emb.parameters():
            p.requires_grad = False
        print("[tok-init] frozen gen.tok_emb")

    # ---------------- candidate builder ----------------
    def build_cand_tok_batch(
        qid_batch: List[str],
        tgt_tok: torch.Tensor,              # (B,topk) full-vocab targets
        use_retrieval: bool
    ) -> torch.Tensor:
        """
        Return cand_tok: (B,C) full-vocab token ids (agent tokens), padded with PAD.
        Always inserts GT tokens (non-PAD) into candidate set.
        """
        B = tgt_tok.size(0)
        C = args.candidate_size
        cand_list = []

        if use_retrieval:
            k_ret = min(C + args.cand_extra, len(a_ids))
            top_idx = retrieve_agent_topN_indices(qid_batch, k=k_ret)  # (B,k_ret)

        for i in range(B):
            gt = [t for t in tgt_tok[i].tolist() if t >= vocab.offset]  # filter PAD/BOS
            gt_set = set(gt)

            cand = []
            # put GT first (important for CE remap)
            for t in gt:
                if t not in cand:
                    cand.append(t)
                    if len(cand) >= C:
                        break

            if len(cand) < C:
                if use_retrieval:
                    # add retrieved
                    for aidx in top_idx[i].tolist():
                        t = vocab.offset + int(aidx)
                        if t in gt_set:
                            # allowed but will dedup anyway
                            pass
                        if t not in cand:
                            cand.append(t)
                            if len(cand) >= C:
                                break
                else:
                    # random negatives (avoid GT agent indices if possible)
                    need = C - len(cand)
                    forbid_agent = set([t - vocab.offset for t in gt if t >= vocab.offset])
                    # rejection sampling
                    tries = 0
                    while need > 0 and tries < need * 20:
                        aidx = random.randrange(len(a_ids))
                        tries += 1
                        if aidx in forbid_agent:
                            continue
                        t = vocab.offset + aidx
                        if t in cand:
                            continue
                        cand.append(t)
                        need -= 1

            # pad
            if len(cand) < C:
                cand += [vocab.PAD] * (C - len(cand))
            else:
                cand = cand[:C]
            cand_list.append(cand)

        return torch.tensor(cand_list, dtype=torch.long, device=device)

    # ---------------- training targets ----------------
    def build_targets(qids: List[str]) -> Tuple[List[str], List[List[int]]]:
        in_q = []; tgt_tok = []
        for qid in qids:
            k_pos = min(args.topk, pos_topk_for_qid(qid, qid_to_part))
            ranked = [aid for aid in all_rankings.get(qid, [])[:k_pos] if aid in vocab.aid2tok]
            if not ranked:
                continue
            toks = [vocab.aid_to_token(a) for a in ranked]
            if len(toks) < args.topk:
                toks += [vocab.PAD]*(args.topk - len(toks))
            in_q.append(qid); tgt_tok.append(toks)
        return in_q, tgt_tok

    train_q, train_targets = build_targets(train_qids)
    valid_q, valid_targets = build_targets(valid_qids)
    print(f"[gen] train sequences={len(train_q)}  valid sequences={len(valid_q)}  topk={args.topk}")

    # ---------------- checkpoints ----------------
    data_sig = dataset_signature(a_ids, all_rankings)
    cache_dir = ensure_cache_dir(args.data_root)
    model_dir = os.path.join(cache_dir, "models"); os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{filename}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{data_sig}.json")

    # ---------------- optim / scaler ----------------
    if args.mode == "gen":
        params = list(sess_enc.parameters()) + list(gen.parameters())
        if composer is not None:
            # NOTE: composer currently only initializes tok_emb; it is not used in forward.
            # We keep it out of optimizer to avoid "training dead params" confusion.
            pass

        opt = torch.optim.Adam(params, lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        nb = math.ceil(len(train_q) / args.batch_size)
        for epoch in range(1, args.epochs+1):
            order = list(range(len(train_q))); random.shuffle(order)
            total = 0.0
            pbar = tqdm(range(nb), desc=f"Epoch {epoch}/{args.epochs} [GEN]")
            for b in pbar:
                sl = order[b*args.batch_size:(b+1)*args.batch_size]
                if not sl:
                    continue

                qid_batch = [train_q[i] for i in sl]
                tgt = torch.tensor([train_targets[i] for i in sl], dtype=torch.long, device=device)  # (B,topk)
                B = tgt.size(0)

                bos = torch.full((B,1), vocab.BOS, dtype=torch.long, device=device)
                inp = torch.cat([bos, tgt[:, :-1]], dim=1)  # (B,topk)

                # session tensor (CPU->GPU)
                q_hist = build_session_tensor(qid_batch).to(device, non_blocking=True)
                a_hist = build_agent_hist_tokens(qid_batch)

                # candidate set (B,C) in full-vocab token ids
                cand_tok = build_cand_tok_batch(qid_batch, tgt_tok=tgt, use_retrieval=bool(args.train_mask))

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    enc_vec = sess_enc(q_hist, a_hist)                 # (B,H)
                    logits_c = gen(enc_vec, inp, cand_tok=cand_tok)    # (B,L,C)
                    tgt_local = remap_targets_to_cand(tgt, cand_tok, pad_tok=vocab.PAD, ignore_index=-100)
                    loss = F.cross_entropy(
                        logits_c.reshape(-1, logits_c.size(-1)),
                        tgt_local.reshape(-1),
                        ignore_index=-100
                    )

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total += loss.item()
                pbar.set_postfix({"ce": f"{loss.item():.4f}", "avg": f"{total/(b+1):.4f}", "C": cand_tok.size(1)})

            print(f"Epoch {epoch}: avg CE={total/max(1,nb):.4f}")

        # save
        ckpt = {
            "mode": "gen",
            "sess_enc": sess_enc.state_dict(),
            "gen": gen.state_dict(),
            "data_sig": data_sig,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "dims": {"d_q": int(Q_t_cpu.shape[1]), "vocab_size": vocab.vocab_size}
        }
        torch.save(ckpt, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"a_ids": a_ids, "q_ids": q_ids, "data_sig": data_sig}, f, ensure_ascii=False, indent=2)
        print(f"[save] model -> {model_path}\n[save] meta  -> {meta_path}")

        # build full encodings (CPU)
        if not bool(args.skip_eval):
            enc_all = []
            bs = 2048
            sess_enc.eval()
            with torch.no_grad():
                for i in range(0, len(q_ids), bs):
                    qid_batch = q_ids[i:i+bs]
                    enc_vec = encode_sessions_in_chunks(qid_batch, chunk=getattr(args, "enc_chunk", 512))
                    enc_all.append(enc_vec.cpu())
            enc_all = torch.cat(enc_all, dim=0)  # CPU

            # --- eval qids: per-part cap ---
            eval_parts = [x.strip() for x in args.eval_parts.split(",") if x.strip()]
            eval_qids = sample_qids_by_part(
                valid_qids,
                qid_to_part=qid_to_part,
                per_part=args.eval_per_part,
                seed=args.seed,
                parts=eval_parts
            )
            print(f"[eval] valid_qids={len(valid_qids)} -> sampled_eval={len(eval_qids)} (per_part={args.eval_per_part})")

            # overall
            valid_metrics = evaluate_model_gen(
                gen_model=gen,
                enc_vecs_cpu=enc_all,
                qid2idx=qid2idx,
                a_ids=a_ids,
                all_rankings=all_rankings,
                eval_qids=eval_qids,
                device=device,
                ks=(5, 10, 50),
                cand_size=args.eval_candidate_size,
                rng_seed=args.seed,
                qid_to_part=qid_to_part,
            )
            print_metrics_table("Validation (GEN, per-part sampled)", valid_metrics, filename=filename)

            # per-part
            for part in eval_parts:
                part_qids = [q for q in eval_qids if qid_to_part.get(q, "Unknown") == part]
                if not part_qids:
                    continue
                m_part = evaluate_model_gen(
                    gen_model=gen,
                    enc_vecs_cpu=enc_all,
                    qid2idx=qid2idx,
                    a_ids=a_ids,
                    all_rankings=all_rankings,
                    eval_qids=part_qids,
                    device=device,
                    ks=(5, 10, 50),
                    cand_size=args.eval_candidate_size,
                    rng_seed=args.seed + 17,  # 给个小偏移，避免和 overall 完全一样的 neg sampling（可选）
                    qid_to_part=qid_to_part,
                )
                print_metrics_table(f"Validation (GEN) {part} (n={len(part_qids)})", m_part, filename=filename)


    # ---------------------- Lite-DPO finetune ----------------------
    if args.mode == "dpo":
        assert os.path.exists(model_path), f"CE checkpoint not found: {model_path}"
        ckpt = torch.load(model_path, map_location=device)
        sess_enc.load_state_dict(ckpt["sess_enc"])
        gen.load_state_dict(ckpt["gen"])

        # reference policy (frozen snapshot of CE model)
        gen_ref = copy.deepcopy(gen).to(device)
        gen_ref.eval()
        for p in gen_ref.parameters():
            p.requires_grad = False

        rm = RewardModel(enc_dim=512, tok_emb=gen.tok_emb, hidden=512, freeze_tok=True).to(device)
        dpo = DPOTrainer(model=gen, ref_model=gen_ref, rm=rm, beta=args.beta, pad_tok=vocab.PAD)

        scaler_rm = torch.cuda.amp.GradScaler(enabled=use_amp)
        scaler_dpo = torch.cuda.amp.GradScaler(enabled=use_amp)

        # RM pretrain: GT list > (generated lists & random lists)
        print("[RM] Pretraining reward model...")
        opt_rm = torch.optim.Adam(rm.parameters(), lr=args.lr)
        rm_steps = max(500, args.dpo_steps//4)
        rm_batch = args.dpo_batch
        train_pool = [qid for qid in train_qids if qid in qid2idx]

        for step in tqdm(range(rm_steps), desc="RM pretrain"):
            batch_q = random.sample(train_pool, min(rm_batch, len(train_pool)))
            # build GT token sequences
            gt_tok = []
            for q in batch_q:
                ranked = [aid for aid in all_rankings.get(q, [])[:args.topk] if aid in vocab.aid2tok]
                if not ranked:
                    # fallback
                    k = min(args.topk, len(a_ids))
                    ranked = random.sample(a_ids, k=k)
                gt = [vocab.aid_to_token(a) for a in ranked]
                if len(gt) < args.topk:
                    gt += [vocab.PAD] * (args.topk - len(gt))
                gt_tok.append(gt)
            gt_tok = torch.tensor(gt_tok, dtype=torch.long, device=device)

            # candidate set for generation (use retrieval if enabled, else random)
            cand_tok = build_cand_tok_batch(batch_q, tgt_tok=gt_tok, use_retrieval=bool(args.train_mask))

            q_hist = build_session_tensor(batch_q).to(device, non_blocking=True)
            opt_rm.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                enc = sess_enc(q_hist, build_agent_hist_tokens(batch_q))
                seq_a = gen.generate(enc, topk=args.topk, cand_tok=cand_tok, temperature=0.7)
                seq_b = gen.generate(enc, topk=args.topk, cand_tok=cand_tok, temperature=1.3)

                # random list
                rnd_tok = []
                for _ in batch_q:
                    # sample random agents, pad to topk
                    kk = args.topk
                    ids = [vocab.offset + random.randrange(len(a_ids)) for _ in range(kk)]
                    rnd_tok.append(ids)
                rnd_tok = torch.tensor(rnd_tok, dtype=torch.long, device=device)

                r_pos = rm(enc, gt_tok)
                r_rand = rm(enc, rnd_tok)
                r_a = rm(enc, seq_a)
                r_b = rm(enc, seq_b)

                loss_rm = (
                    F.softplus(-(r_pos - r_rand)) +
                    F.softplus(-(r_pos - r_a)) +
                    F.softplus(-(r_pos - r_b))
                ).mean()

            scaler_rm.scale(loss_rm).backward()
            scaler_rm.step(opt_rm)
            scaler_rm.update()

        print("[RM] done.")
        rm.eval()
        for p in rm.parameters():
            p.requires_grad = False

        # Freeze session encoder for stability (optional)
        for p in sess_enc.parameters():
            p.requires_grad = False

        lr_dpo = args.dpo_lr if args.dpo_lr is not None else args.lr * 0.1
        opt_dpo = torch.optim.Adam([p for p in gen.parameters() if p.requires_grad], lr=lr_dpo)

        dpo_steps = args.dpo_steps
        dpo_batch = args.dpo_batch

        for step in tqdm(range(dpo_steps), desc="Lite-DPO"):
            batch_q = random.sample(train_pool, min(dpo_batch, len(train_pool)))

            # build GT token sequences for candidate insertion
            gt_tok = []
            for q in batch_q:
                ranked = [aid for aid in all_rankings.get(q, [])[:args.topk] if aid in vocab.aid2tok]
                if not ranked:
                    ranked = random.sample(a_ids, k=min(args.topk, len(a_ids)))
                gt = [vocab.aid_to_token(a) for a in ranked]
                if len(gt) < args.topk:
                    gt += [vocab.PAD] * (args.topk - len(gt))
                gt_tok.append(gt)
            gt_tok = torch.tensor(gt_tok, dtype=torch.long, device=device)

            cand_tok = build_cand_tok_batch(batch_q, tgt_tok=gt_tok, use_retrieval=bool(args.train_mask))

            q_hist = build_session_tensor(batch_q).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                enc = sess_enc(q_hist, build_agent_hist_tokens(batch_q))
                seq_a = gen.generate(enc, topk=args.topk, cand_tok=cand_tok, temperature=0.7)
                seq_b = gen.generate(enc, topk=args.topk, cand_tok=cand_tok, temperature=1.3)

            with torch.no_grad():
                r_a = rm(enc, seq_a)
                r_b = rm(enc, seq_b)
                prefer_a = (r_a >= r_b)
                margin = 0.05
                keep = (r_a - r_b).abs() >= margin
                if keep.sum().item() == 0:
                    continue

            pref_seq = torch.where(prefer_a.unsqueeze(1), seq_a, seq_b)
            nonpref_seq = torch.where(prefer_a.unsqueeze(1), seq_b, seq_a)

            # keep filtered
            pref_seq = pref_seq[keep]
            nonpref_seq = nonpref_seq[keep]
            enc_keep = enc[keep]
            cand_keep = cand_tok[keep]

            gen.train()
            opt_dpo.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = dpo.dpo_loss(enc_keep, pref_seq, nonpref_seq, cand_keep)

            scaler_dpo.scale(loss).backward()
            scaler_dpo.unscale_(opt_dpo)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            scaler_dpo.step(opt_dpo)
            scaler_dpo.update()

            if (step+1) % 100 == 0:
                print(f"[DPO] step {step+1}/{dpo_steps} loss={loss.item():.4f}")

        ckpt_out = {
            "mode": "gen+dpo",
            "sess_enc": sess_enc.state_dict(),
            "gen": gen.state_dict(),
            "data_sig": data_sig,
            "saved_at": datetime.now().isoformat(timespec="seconds")
        }
        out_path = model_path.replace(".pt", "_dpo.pt")
        torch.save(ckpt_out, out_path)
        print(f"[save] dpo model -> {out_path}")

        # quick eval
        enc_all = []
        bs = 2048
        sess_enc.eval()
        with torch.no_grad():
            for i in range(0, len(q_ids), bs):
                qid_batch = q_ids[i:i+bs]
                enc_vec = encode_sessions_in_chunks(qid_batch, chunk=getattr(args, "enc_chunk", 512))
                enc_all.append(enc_vec.cpu())
        enc_all = torch.cat(enc_all, dim=0)

        # --- eval qids: per-part cap ---
        eval_parts = [x.strip() for x in args.eval_parts.split(",") if x.strip()]
        eval_qids = sample_qids_by_part(
            valid_qids,
            qid_to_part=qid_to_part,
            per_part=args.eval_per_part,
            seed=args.seed,
            parts=eval_parts
        )
        print(f"[eval] valid_qids={len(valid_qids)} -> sampled_eval={len(eval_qids)} (per_part={args.eval_per_part})")

        # overall
        valid_metrics = evaluate_model_gen(
            gen_model=gen,
            enc_vecs_cpu=enc_all,
            qid2idx=qid2idx,
            a_ids=a_ids,
            all_rankings=all_rankings,
            eval_qids=eval_qids,
            device=device,
            ks=(5, 10, 50),
            cand_size=args.eval_candidate_size,
            rng_seed=args.seed,
            qid_to_part=qid_to_part,
        )
        print_metrics_table("Validation (GEN, per-part sampled)", valid_metrics, filename=filename)

        # per-part
        for part in eval_parts:
            part_qids = [q for q in eval_qids if qid_to_part.get(q, "Unknown") == part]
            if not part_qids:
                continue
            m_part = evaluate_model_gen(
                gen_model=gen,
                enc_vecs_cpu=enc_all,
                qid2idx=qid2idx,
                a_ids=a_ids,
                all_rankings=all_rankings,
                eval_qids=part_qids,
                device=device,
                ks=(5, 10, 50),
                cand_size=args.eval_candidate_size,
                rng_seed=args.seed + 17,  # 给个小偏移，避免和 overall 完全一样的 neg sampling（可选）
                qid_to_part=qid_to_part,
            )
            print_metrics_table(f"Validation (GEN) {part} (n={len(part_qids)})", m_part, filename=filename)


if __name__ == "__main__":
    main()
