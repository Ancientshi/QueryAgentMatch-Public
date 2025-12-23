#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
import os
import tempfile
import uuid
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from numpy.lib.format import open_memmap

try:
    from scipy import sparse as sp
except Exception as e:  # pragma: no cover - import error is user environment
    raise RuntimeError("This module needs SciPy for CSR features. pip install scipy\n" + str(e))

from .config import TFIDF_MAX_FEATURES


@dataclass
class FeatureCache:
    q_ids: List[str]
    a_ids: List[str]
    tool_names: List[str]
    Q: np.ndarray
    A_text_full: np.ndarray
    agent_tool_idx_padded: np.ndarray
    agent_tool_mask: np.ndarray


def build_text_corpora(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[List[str]]]:
    """Build text corpora aligned with simple_lightfm_agent_rec.py."""
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid].get("input", "") for qid in q_ids]

    tool_names = list(tools.keys())

    def tool_text(name: str) -> str:
        t = tools.get(name, {}) or {}
        desc = t.get("description", "")
        return f"{name} {desc}".strip()

    tool_texts = [tool_text(name) for name in tool_names]

    a_ids = list(all_agents.keys())
    a_texts: List[str] = []
    a_tool_lists: List[List[str]] = []
    for aid in a_ids:
        a = all_agents.get(aid, {}) or {}
        mname = ((a.get("M") or {}).get("name") or "").strip()
        tool_list = ((a.get("T") or {}).get("tools") or [])
        a_tool_lists.append(tool_list)
        a_texts.append(mname)

    return q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists


def build_twotower_text_corpora(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[List[str]]]:
    """Build TF-IDF corpora for two-tower models with tool text appended to agent text."""
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid].get("input", "") for qid in q_ids]

    tool_names = list(tools.keys())

    def tool_text(name: str) -> str:
        t = tools.get(name, {}) or {}
        desc = t.get("description", "")
        return f"{name} {desc}".strip()

    tool_texts = [tool_text(name) for name in tool_names]

    a_ids = list(all_agents.keys())
    a_texts: List[str] = []
    a_tool_lists: List[List[str]] = []
    for aid in a_ids:
        a = all_agents.get(aid, {}) or {}
        mname = ((a.get("M") or {}).get("name") or "").strip()
        tool_list = ((a.get("T") or {}).get("tools") or [])
        a_tool_lists.append(tool_list)
        tools_concat = " ".join(tool_text(name) for name in tool_list)
        a_texts.append(f"{mname} {tools_concat}".strip())

    return q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists


def build_transformer_corpora(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[List[str]]]:
    """Build transformer corpora with tool descriptions embedded in agent text."""
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid].get("input", "") for qid in q_ids]

    tool_names = list(tools.keys())

    def tool_text(name: str) -> str:
        t = tools.get(name, {}) or {}
        desc = t.get("description", "")
        return f"{name} {desc}".strip()

    a_ids = list(all_agents.keys())
    a_texts: List[str] = []
    a_tool_lists: List[List[str]] = []
    for aid in a_ids:
        a = all_agents.get(aid, {}) or {}
        mname = ((a.get("M") or {}).get("name") or "").strip()
        tool_list = ((a.get("T") or {}).get("tools") or [])
        a_tool_lists.append(tool_list)
        tools_concat = " ".join(tool_text(name) for name in tool_list)
        a_texts.append(f"{mname} {tools_concat}".strip())

    return q_ids, q_texts, tool_names, a_ids, a_texts, a_tool_lists


def build_vectorizers(
    q_texts: List[str],
    tool_texts: List[str],
    a_texts: List[str],
    max_features: int,
) -> Tuple[TfidfVectorizer, TfidfVectorizer, TfidfVectorizer, "sp.csr_matrix", "sp.csr_matrix", "sp.csr_matrix"]:
    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    tool_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    a_vec = TfidfVectorizer(max_features=max_features, lowercase=True)

    Q = q_vec.fit_transform(q_texts)
    Tm = tool_vec.fit_transform(tool_texts)
    Am = a_vec.fit_transform(a_texts)
    return q_vec, tool_vec, a_vec, Q, Tm, Am


def agent_tool_text_matrix(
    agent_tool_lists: List[List[str]],
    tool_names: List[str],
    tool_matrix: "sp.csr_matrix",
) -> np.ndarray:
    name2idx = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(agent_tool_lists)
    dim = tool_matrix.shape[1]
    out = np.zeros((num_agents, dim), dtype=np.float32)
    for i, tool_list in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if not idxs:
            continue
        vecs = tool_matrix[idxs].toarray()
        out[i] = vecs.mean(axis=0).astype(np.float32)
    return out


def build_agent_tool_id_buffers(
    a_ids: List[str],
    agent_tool_lists: List[List[str]],
    tool_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    t_map = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(a_ids)
    max_t = max([len(lst) for lst in agent_tool_lists]) if num_agents > 0 else 0
    if max_t == 0:
        max_t = 1
    idx_pad = np.zeros((num_agents, max_t), dtype=np.int64)
    mask = np.zeros((num_agents, max_t), dtype=np.float32)
    for i, lst in enumerate(agent_tool_lists):
        for j, name in enumerate(lst[:max_t]):
            if name in t_map:
                idx_pad[i, j] = t_map[name]
                mask[i, j] = 1.0
    return idx_pad, mask


def build_twotower_feature_cache(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
    max_features: int = TFIDF_MAX_FEATURES,
) -> Tuple[FeatureCache, TfidfVectorizer]:
    (
        q_ids,
        q_texts,
        tool_names,
        tool_texts,
        a_ids,
        a_texts,
        a_tool_lists,
    ) = build_twotower_text_corpora(all_agents, all_questions, tools)

    q_vec, tool_vec, a_vec, Q_csr, Tm_csr, Am_csr = build_vectorizers(
        q_texts, tool_texts, a_texts, max_features
    )

    Atool = agent_tool_text_matrix(a_tool_lists, tool_names, Tm_csr)
    Am = Am_csr.toarray().astype(np.float32)
    A_text_full = np.concatenate([Am, Atool], axis=1).astype(np.float32)
    Q = Q_csr.toarray().astype(np.float32)

    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)
    return FeatureCache(
        q_ids=q_ids,
        a_ids=a_ids,
        tool_names=tool_names,
        Q=Q,
        A_text_full=A_text_full,
        agent_tool_idx_padded=agent_tool_idx_padded,
        agent_tool_mask=agent_tool_mask,
    ), q_vec


def build_feature_cache(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
    max_features: int = TFIDF_MAX_FEATURES,
) -> Tuple[FeatureCache, TfidfVectorizer]:
    q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists = build_text_corpora(
        all_agents, all_questions, tools
    )
    q_vec, tool_vec, a_vec, Q_csr, Tm_csr, Am_csr = build_vectorizers(
        q_texts, tool_texts, a_texts, max_features
    )

    Atool = agent_tool_text_matrix(a_tool_lists, tool_names, Tm_csr)
    Am = Am_csr.toarray().astype(np.float32)
    A_text_full = np.concatenate([Am, Atool], axis=1).astype(np.float32)
    Q = Q_csr.toarray().astype(np.float32)

    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)
    return FeatureCache(
        q_ids=q_ids,
        a_ids=a_ids,
        tool_names=tool_names,
        Q=Q,
        A_text_full=A_text_full,
        agent_tool_idx_padded=agent_tool_idx_padded,
        agent_tool_mask=agent_tool_mask,
    ), q_vec


def save_feature_cache(cache_dir: str, cache: FeatureCache) -> None:
    def dump_json(name: str, payload: object) -> None:
        with open(f"{cache_dir}/{name}", "w", encoding="utf-8") as f:
            import json

            json.dump(payload, f, ensure_ascii=False)

    dump_json("q_ids.json", cache.q_ids)
    dump_json("a_ids.json", cache.a_ids)
    dump_json("tool_names.json", cache.tool_names)

    np.save(f"{cache_dir}/Q.npy", cache.Q.astype(np.float32))
    np.save(f"{cache_dir}/A_text_full.npy", cache.A_text_full.astype(np.float32))
    np.save(f"{cache_dir}/agent_tool_idx_padded.npy", cache.agent_tool_idx_padded.astype(np.int64))
    np.save(f"{cache_dir}/agent_tool_mask.npy", cache.agent_tool_mask.astype(np.float32))


def save_q_vectorizer(cache_dir: str, q_vec: TfidfVectorizer) -> None:
    import pickle

    with open(f"{cache_dir}/q_vectorizer.pkl", "wb") as f:
        pickle.dump(q_vec, f)


def load_q_vectorizer(cache_dir: str) -> Optional[TfidfVectorizer]:
    import os
    import pickle

    path = f"{cache_dir}/q_vectorizer.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_cache(cache_dir: str) -> FeatureCache:
    import json

    with open(f"{cache_dir}/q_ids.json", "r", encoding="utf-8") as f:
        q_ids = json.load(f)
    with open(f"{cache_dir}/a_ids.json", "r", encoding="utf-8") as f:
        a_ids = json.load(f)
    with open(f"{cache_dir}/tool_names.json", "r", encoding="utf-8") as f:
        tool_names = json.load(f)

    Q = np.load(f"{cache_dir}/Q.npy")
    A_text_full = np.load(f"{cache_dir}/A_text_full.npy")
    agent_tool_idx_padded = np.load(f"{cache_dir}/agent_tool_idx_padded.npy")
    agent_tool_mask = np.load(f"{cache_dir}/agent_tool_mask.npy")

    return FeatureCache(
        q_ids=q_ids,
        a_ids=a_ids,
        tool_names=tool_names,
        Q=Q,
        A_text_full=A_text_full,
        agent_tool_idx_padded=agent_tool_idx_padded,
        agent_tool_mask=agent_tool_mask,
    )


def feature_cache_exists(cache_dir: str) -> bool:
    import os

    needed = [
        "q_ids.json",
        "a_ids.json",
        "tool_names.json",
        "Q.npy",
        "A_text_full.npy",
        "agent_tool_idx_padded.npy",
        "agent_tool_mask.npy",
    ]
    return all(os.path.exists(f"{cache_dir}/{name}") for name in needed)


def to_2d_float32(x: np.ndarray | list | tuple) -> np.ndarray:
    if isinstance(x, tuple) and len(x) == 3:
        arr, inv_order, _ = x
        x = arr if inv_order is None else arr[inv_order, :]
    x = np.asarray(x)
    if x.dtype == np.object_ or x.ndim != 2:
        try:
            x = np.vstack([np.asarray(row, dtype=np.float32) for row in x])
        except Exception as e:
            raise ValueError(
                f"Embedding batch is ragged or non-2D: {type(x)}, shape={getattr(x, 'shape', None)}"
            ) from e
    return x.astype(np.float32, copy=False)


def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mat = to_2d_float32(mat)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return mat / n


def _post_embed(embed_url: str, docs: List[str]) -> np.ndarray:
    if embed_url and embed_url.startswith("http"):
        import requests

        response = requests.post(embed_url, json={"documents": docs}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if "embeddings" not in payload:
            raise ValueError("Embedding service response missing `embeddings` field.")
        return np.asarray(payload["embeddings"], dtype=np.float32)

    from utils import load_BGEM3_model, get_embeddings

    load_BGEM3_model()
    embs = get_embeddings(docs)
    return np.asarray(embs, dtype=np.float32)


def batch_embed(
    texts: List[str],
    embed_url: str,
    batch_size: int = 64,
    desc: str = "Embedding",
    *,
    use_memmap: bool = True,
    memmap_path: str | None = None,
    return_mode: str = "array",
    sort_by_length: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, str]:
    num_texts = len(texts)
    if num_texts == 0:
        return np.zeros((0, 0), dtype=np.float32)

    order = np.arange(num_texts)
    if sort_by_length:
        order = np.argsort([len(t) for t in texts])
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(num_texts)
    texts_sorted = [texts[i] for i in order]

    out = None
    mm = None
    dim = None

    for start in range(0, num_texts, batch_size):
        end = min(start + batch_size, num_texts)
        chunk = texts_sorted[start:end]
        embs = _post_embed(embed_url, chunk)
        if dim is None:
            dim = int(embs.shape[1])
            if use_memmap:
                if memmap_path is None:
                    memmap_path = os.path.join(tempfile.gettempdir(), f"emb_{uuid.uuid4().hex}.npy")
                mm = open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=(num_texts, dim))
            else:
                out = np.empty((num_texts, dim), dtype=np.float32)

        if embs.shape[1] != dim:
            raise ValueError(f"Embedding dim changed: got {embs.shape[1]} vs {dim}")

        target = mm if use_memmap else out
        target[start:end, :] = embs

    if use_memmap:
        mm.flush()
        view = np.load(memmap_path, mmap_mode="r")
        if sort_by_length:
            view = view[inv_order, :]
        if return_mode == "mmap":
            return view, inv_order, memmap_path
        return np.array(view, copy=True)

    return out if not sort_by_length else out[inv_order, :]


def agent_tool_text_matrix_bge(
    agent_tool_lists: List[List[str]],
    tool_names: List[str],
    tool_embs: np.ndarray,
) -> np.ndarray:
    name2idx = {n: i for i, n in enumerate(tool_names)}
    dim = tool_embs.shape[1]
    num_agents = len(agent_tool_lists)
    out = np.zeros((num_agents, dim), dtype=np.float32)
    for i, tool_list in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if not idxs:
            continue
        vecs = tool_embs[idxs]
        out[i] = vecs.mean(axis=0).astype(np.float32)
    return out


def build_twotower_bge_feature_cache(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
    *,
    embed_url: str,
    embed_batch: int = 64,
    use_memmap: bool = True,
    sort_by_length: bool = False,
) -> FeatureCache:
    (
        q_ids,
        q_texts,
        tool_names,
        tool_texts,
        a_ids,
        a_texts,
        a_tool_lists,
    ) = build_twotower_text_corpora(all_agents, all_questions, tools)

    Q = batch_embed(q_texts, embed_url, embed_batch, desc="Embedding questions", use_memmap=use_memmap)
    Q = l2_normalize(Q)

    ToolE = batch_embed(tool_texts, embed_url, embed_batch, desc="Embedding tools", use_memmap=use_memmap)
    ToolE = l2_normalize(ToolE)

    A_text_emb = batch_embed(
        a_texts, embed_url, embed_batch, desc="Embedding agents (text)", use_memmap=use_memmap
    )
    A_text_emb = l2_normalize(A_text_emb)

    A_tool_emb = agent_tool_text_matrix_bge(a_tool_lists, tool_names, ToolE)
    if A_tool_emb.size > 0:
        A_tool_emb = l2_normalize(A_tool_emb)

    A_text_full = np.concatenate([A_text_emb, A_tool_emb], axis=1).astype(np.float32)

    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)

    return FeatureCache(
        q_ids=q_ids,
        a_ids=a_ids,
        tool_names=tool_names,
        Q=Q.astype(np.float32),
        A_text_full=A_text_full.astype(np.float32),
        agent_tool_idx_padded=agent_tool_idx_padded,
        agent_tool_mask=agent_tool_mask,
    )


def normalize_features(user_np: np.ndarray, item_np: np.ndarray) -> Tuple["sp.csr_matrix", "sp.csr_matrix"]:
    user_np = normalize(user_np, norm="l2", axis=1, copy=False)
    item_np = normalize(item_np, norm="l2", axis=1, copy=False)
    return sp.csr_matrix(user_np, dtype=np.float32), sp.csr_matrix(item_np, dtype=np.float32)
