#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Tower (TF-IDF) inference and Flask UI.

This script loads a trained TwoTowerTFIDF checkpoint, rebuilds the TF-IDF
feature cache (or uses the cached copy from training), and exposes a simple
web UI plus a JSON API for recommending agents for an ad-hoc query.

Usage examples:
  # Start the Flask UI (defaults to 0.0.0.0:8000)
  python infer_twotower_tfidf.py \
    --data_root /path/to/dataset_root \
    --model_path /path/to/models/latest_xxxx.pt \
    --device cpu

  # Run a single query without starting the server
  python infer_twotower_tfidf.py \
    --data_root /path/to/dataset_root \
    --model_path /path/to/models/latest_xxxx.pt \
    --query "如何构建一个天气查询Agent？" \
    --serve 0
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request

from agent_rec.config import TFIDF_MAX_FEATURES
from agent_rec.features import (
    UNK_LLM_TOKEN,
    UNK_TOOL_TOKEN,
    build_agent_content_view,
    feature_cache_exists,
    load_feature_cache,
    load_q_vectorizer,
)
from agent_rec.models.two_tower import TwoTowerTFIDF
from agent_rec.run_common import bootstrap_run, shared_cache_dir


HTML_TEMPLATE = """
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>TwoTower TF-IDF Agent 推荐</title>
  <style>
    body { font-family: "Helvetica Neue", Arial, sans-serif; background: #f5f6fa; }
    .container { max-width: 1200px; margin: 60px auto; background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 4px 30px rgba(0,0,0,0.08); }
    h1 { text-align: center; margin-bottom: 12px; }
    p.subtitle { text-align: center; color: #555; margin-top: 0; }
    form { display: flex; gap: 12px; justify-content: center; align-items: center; margin-bottom: 24px; }
    input[type=text] { width: 100%; max-width: 620px; padding: 14px 18px; font-size: 16px; border: 1px solid #dcdde1; border-radius: 10px; box-sizing: border-box; }
    button { padding: 14px 22px; font-size: 16px; background: #2d8cf0; color: #fff; border: none; border-radius: 10px; cursor: pointer; }
    button:hover { background: #1d7cd9; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
    .panel { border: 1px solid #ecf0f1; border-radius: 12px; padding: 14px 14px 4px; background: #fbfcfe; box-shadow: 0 1px 12px rgba(0,0,0,0.03); }
    .panel h2 { margin: 0 0 8px; font-size: 17px; display: flex; align-items: center; gap: 6px; }
    .panel small { color: #777; font-weight: 400; }
    .results ol, .results ul { padding-left: 18px; margin: 0; }
    .agent-card { border: 1px solid #ecf0f1; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; background: #fff; }
    .agent-header { display: flex; justify-content: space-between; align-items: center; }
    .agent-title { font-weight: 600; font-size: 16px; }
    .agent-id { color: #888; font-size: 12px; }
    .score { color: #2d8cf0; font-weight: 600; font-size: 14px; }
    .meta { color: #444; margin-top: 4px; line-height: 1.5; font-size: 13px; }
    .tools { margin-top: 6px; color: #555; font-size: 13px; }
    .candidate { border-left: 3px solid #2d8cf0; padding-left: 10px; margin-bottom: 10px; }
    .pill { display: inline-block; background: #eef5ff; color: #2d8cf0; border-radius: 999px; padding: 3px 10px; font-size: 12px; margin-right: 6px; }
    .error { color: #c0392b; text-align: center; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>TwoTower TF-IDF 推荐</h1>
    <p class="subtitle">输入需求，得到前 10 个推荐的 Agent（含配置详情）。</p>
    <form method="post">
      <input type="text" name="query" placeholder="请输入查询，例如：写一个天气查询助手" value="{{ query|e }}" required />
      <button type="submit">推荐</button>
    </form>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if results %}
      <div class="results grid">
        <div class="panel">
          <h2>LLM 推荐 <small>Top 10</small></h2>
          <ol>
            {% for item in llm_recs %}
            <li class="agent-card">
              <div class="agent-header">
                <div class="agent-title">{{ item.llm_name }}</div>
                <div class="score">{{ '%.4f'|format(item.score) }}</div>
              </div>
              <div class="meta">LLM ID: {{ item.llm_id or '未知' }}</div>
              <div class="meta">来源 Agent: {{ item.source_agent }}</div>
              {% if item.desc %}<div class="meta">{{ item.desc }}</div>{% endif %}
            </li>
            {% endfor %}
          </ol>
        </div>

        <div class="panel">
          <h2>Tool 推荐 <small>Top 10</small></h2>
          <ol>
            {% for item in tool_recs %}
            <li class="agent-card">
              <div class="agent-header">
                <div class="agent-title">{{ item.tool_name }}</div>
                <div class="score">{{ '%.4f'|format(item.score) }}</div>
              </div>
              <div class="meta">来源 Agent: {{ item.source_agent }}</div>
              {% if item.desc %}<div class="meta">{{ item.desc }}</div>{% endif %}
            </li>
            {% endfor %}
          </ol>
        </div>

        <div class="panel">
          <h2>Agent 候选（GPT 提示）</h2>
          <ol>
            {% for c in candidates %}
            <li class="agent-card candidate">
              <div class="agent-title">{{ c.title }}</div>
              <div class="meta">LLM: {{ c.llm_name }}</div>
              <div class="meta">Tools: {{ c.tools | join(', ') }}</div>
              <div class="meta">{{ c.reason }}</div>
            </li>
            {% endfor %}
          </ol>
        </div>

        <div class="panel">
          <h2>模型重排结果 <small>Top {{ results|length }}</small></h2>
          <ol>
            {% for r in results %}
            <li>
              <div class="agent-card">
                <div class="agent-header">
                  <div>
                    <div class="agent-title">{{ r.display_name }}</div>
                    <div class="agent-id">ID: {{ r.agent_id }}</div>
                  </div>
                  <div class="score">score={{ '%.4f'|format(r.score) }}</div>
                </div>
                <div class="meta">模型: {{ r.model_name or '未知模型' }} ({{ r.model_id or '未提供ID' }})</div>
                {% if r.model_desc %}<div class="meta">模型描述: {{ r.model_desc }}</div>{% endif %}
                {% if r.tools %}
                  <div class="tools"><strong>工具:</strong> {{ r.tools | join(', ') }}</div>
                {% endif %}
                {% if r.tool_details %}
                  <div class="tools"><strong>工具详情:</strong>
                    <ul>
                    {% for t in r.tool_details %}
                      <li>{{ t.name }}{% if t.description %}: {{ t.description }}{% endif %}</li>
                    {% endfor %}
                    </ul>
                  </div>
                {% endif %}
              </div>
            </li>
            {% endfor %}
          </ol>
        </div>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


def _device_from_arg(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] CUDA 不可用，回退到 CPU (请求: {device_str}).")
        return torch.device("cpu")
    return device


def _load_checkpoint(model_path: str, device: torch.device) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"模型文件不合法: {model_path}")
    return ckpt


def _resolve_feature_cache_dir(data_root: str, max_features: int, data_sig: str) -> str:
    return shared_cache_dir(data_root, "features", f"twotower_tfidf_{max_features}_{data_sig}")


def _build_encoder(
    *,
    ckpt: dict,
    feature_cache,
    device: torch.device,
) -> TwoTowerTFIDF:
    flags = ckpt.get("flags", {}) if isinstance(ckpt, dict) else {}
    dims = ckpt.get("dims", {}) if isinstance(ckpt, dict) else {}

    encoder = TwoTowerTFIDF(
        d_q=int(dims.get("d_q", feature_cache.Q.shape[1])),
        d_a=int(dims.get("d_a", feature_cache.A_text_full.shape[1] if hasattr(feature_cache, "A_text_full") else feature_cache.A_model_content.shape[1])),
        hid=int(dims.get("hid", 256)),
        num_tools=int(dims.get("num_tools", len(feature_cache.tool_id_vocab))),
        num_llm_ids=int(len(feature_cache.llm_vocab)),
        agent_tool_idx_padded=torch.tensor(feature_cache.agent_tool_idx_padded, dtype=torch.long, device=device),
        agent_tool_mask=torch.tensor(feature_cache.agent_tool_mask, dtype=torch.float32, device=device),
        agent_llm_idx=torch.tensor(feature_cache.agent_llm_idx, dtype=torch.long, device=device),
        use_tool_id_emb=bool(flags.get("use_tool_id_emb", True)),
        use_llm_id_emb=bool(flags.get("use_llm_id_emb", False)),
        num_agents=len(feature_cache.a_ids),
        num_queries=len(feature_cache.q_ids),
        use_query_id_emb=bool(flags.get("use_query_id_emb", False)),
    ).to(device)
    encoder.load_state_dict(ckpt["state_dict"], strict=False)
    encoder.eval()
    return encoder


class TwoTowerInference:
    def __init__(
        self,
        *,
        data_root: str,
        model_path: str,
        device: torch.device,
        max_features: int,
        topk: int,
    ) -> None:
        self.data_root = data_root
        self.model_path = model_path
        self.device = device
        self.max_features = max_features
        self.topk = topk

        boot = bootstrap_run(
            data_root=data_root,
            exp_name="infer_twotower_tfidf",
            topk=topk,
            seed=1234,
            with_tools=True,
        )
        self.bundle = boot.bundle
        self.tools = boot.tools or {}
        ckpt = _load_checkpoint(model_path, device)
        self.ckpt_data_sig = ckpt.get("data_sig", boot.data_sig)

        cache_dir = _resolve_feature_cache_dir(data_root, max_features, self.ckpt_data_sig)
        if not feature_cache_exists(cache_dir):
            raise RuntimeError(
                f"未找到特征缓存: {cache_dir}\n"
                "请确认使用相同的数据根目录与max_features训练过TwoTower TF-IDF模型。"
            )

        self.feature_cache = load_feature_cache(cache_dir)
        q_vec = load_q_vectorizer(cache_dir)
        if q_vec is None:
            raise RuntimeError(f"未找到查询向量化器: {cache_dir}/q_vectorizer.pkl")
        self.q_vectorizer = q_vec

        use_model_content_vector = bool(ckpt.get("flags", {}).get("use_model_content_vector", True))
        use_tool_content_vector = bool(ckpt.get("flags", {}).get("use_tool_content_vector", True))
        self.agent_content = build_agent_content_view(
            cache=self.feature_cache,
            use_model_content_vector=use_model_content_vector,
            use_tool_content_vector=use_tool_content_vector,
        )

        self.encoder = _build_encoder(ckpt=ckpt, feature_cache=self.feature_cache, device=self.device)
        self.encoder.set_agent_features(self.agent_content)
        self.agent_embeddings = self.encoder.export_agent_embeddings()
        self.agent_ids = list(self.feature_cache.a_ids)
        self.agent_id_to_index = {aid: i for i, aid in enumerate(self.agent_ids)}
        self.agent_llm_ids = list(getattr(self.feature_cache, "llm_ids", []))
        self.agent_tools: List[List[str]] = []
        self.agent_llm_names: List[str] = []
        for aid in self.agent_ids:
            agent = self.bundle.all_agents.get(aid, {}) or {}
            m = (agent.get("M") or {}) if isinstance(agent, dict) else {}
            t = (agent.get("T") or {}) if isinstance(agent, dict) else {}
            self.agent_llm_names.append((m.get("name") or m.get("id") or "").strip())
            self.agent_tools.append(list((t.get("tools") or [])))
        self.llm_vocab = [x for x in (self.feature_cache.llm_vocab or []) if x != UNK_LLM_TOKEN]
        self.tool_vocab = [x for x in (self.feature_cache.tool_id_vocab or []) if x != UNK_TOOL_TOKEN]
        self.tool_desc_map = {
            name: (self.tools.get(name, {}) or {}).get("description", "")
            for name in self.tool_vocab
        }

    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.q_vectorizer.transform([query]).toarray().astype(np.float32)
        q = torch.from_numpy(vec).to(self.device)
        q_idx = None
        if getattr(self.encoder, "use_query_id_emb", False):
            q_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            qe = self.encoder.encode_q(q, q_idx=q_idx).cpu().numpy()
        return qe

    def score_all_agents(self, query: str) -> np.ndarray:
        query = (query or "").strip()
        if not query:
            raise ValueError("查询不能为空。")
        qe = self._encode_query(query)
        scores = np.dot(qe, self.agent_embeddings.T).reshape(-1)
        return scores

    def _top_agents_from_scores(
        self, scores: np.ndarray, *, topk: int | None = None, candidate_ids: Sequence[str] | None = None
    ) -> List[Tuple[str, float]]:
        k = topk or self.topk
        if candidate_ids:
            candidate_idx = [self.agent_id_to_index[a] for a in candidate_ids if a in self.agent_id_to_index]
            if candidate_idx:
                sub_scores = np.array([scores[i] for i in candidate_idx], dtype=np.float32)
                k = min(k, len(sub_scores))
                idx = np.argpartition(-sub_scores, k - 1)[:k]
                ordered = idx[np.argsort(-sub_scores[idx])]
                return [(candidate_ids[i], float(sub_scores[i])) for i in ordered]
        k = min(k, len(scores))
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        ordered = idx[np.argsort(-scores[idx])]
        return [(self.agent_ids[i], float(scores[i])) for i in ordered]

    def recommend(
        self, query: str, topk: int | None = None, *, return_scores: bool = False
    ) -> List[Tuple[str, float]] | Tuple[List[Tuple[str, float]], np.ndarray]:
        scores = self.score_all_agents(query)
        recs = self._top_agents_from_scores(scores, topk=topk)
        if return_scores:
            return recs, scores
        return recs

    @lru_cache(maxsize=4096)
    def agent_detail(self, agent_id: str) -> Dict[str, object]:
        agent = self.bundle.all_agents.get(agent_id, {}) or {}
        model_info = agent.get("M", {}) if isinstance(agent, dict) else {}
        tool_info = agent.get("T", {}) if isinstance(agent, dict) else {}
        tools = tool_info.get("tools", []) if isinstance(tool_info, dict) else []
        tool_details = []
        for name in tools:
            t = self.tools.get(name, {}) if isinstance(self.tools, dict) else {}
            tool_details.append({
                "name": name,
                "description": (t or {}).get("description", ""),
            })
        return {
            "agent_id": agent_id,
            "display_name": model_info.get("name") or agent.get("name") or agent_id,
            "model_id": model_info.get("id") or "",
            "model_name": model_info.get("name") or "",
            "model_desc": model_info.get("desc")
            or model_info.get("description")
            or agent.get("desc")
            or agent.get("description")
            or "",
            "tools": tools,
            "tool_details": tool_details,
        }

    def recommend_llms(self, scores: np.ndarray, topk: int = 10) -> List[Dict[str, object]]:
        best: Dict[str, Dict[str, object]] = {}
        for idx, score in enumerate(scores):
            if idx >= len(self.agent_ids):
                continue
            aid = self.agent_ids[idx]
            detail = self.agent_detail(aid)
            llm_id = (self.agent_llm_ids[idx] if idx < len(self.agent_llm_ids) else "") or detail.get("model_id") or detail.get("model_name") or ""
            if not llm_id or llm_id == UNK_LLM_TOKEN:
                continue
            llm_name = detail.get("model_name") or self.agent_llm_names[idx] or llm_id
            if llm_id not in best or score > best[llm_id]["score"]:
                best[llm_id] = {
                    "llm_id": llm_id,
                    "llm_name": llm_name,
                    "score": score,
                    "source_agent": detail.get("display_name") or aid,
                    "desc": detail.get("model_desc") or "",
                }
        ordered = sorted(best.values(), key=lambda x: -x["score"])
        return ordered[:topk]

    def recommend_tools(self, scores: np.ndarray, topk: int = 10) -> List[Dict[str, object]]:
        best: Dict[str, Dict[str, object]] = {}
        for idx, score in enumerate(scores):
            if idx >= len(self.agent_ids):
                continue
            aid = self.agent_ids[idx]
            tools = self.agent_tools[idx] if idx < len(self.agent_tools) else []
            detail = self.agent_detail(aid)
            if not tools:
                tools = detail.get("tools", [])
            for name in tools:
                if not name or name == UNK_TOOL_TOKEN:
                    continue
                desc = self.tool_desc_map.get(name) or ""
                if name not in best or score > best[name]["score"]:
                    best[name] = {
                        "tool_name": name,
                        "score": score,
                        "source_agent": detail.get("display_name") or aid,
                        "desc": desc,
                    }
        ordered = sorted(best.values(), key=lambda x: -x["score"])
        return ordered[:topk]

    def compose_agent_candidates(
        self,
        query: str,
        llm_recs: List[Dict[str, object]],
        tool_recs: List[Dict[str, object]],
        topk: int = 10,
    ) -> List[Dict[str, object]]:
        """
        模拟 GPT 提示，利用 LLM/Tool 候选组合出可落地的 Agent 设想。
        """
        if not llm_recs or not tool_recs:
            return []

        suggestions: List[Dict[str, object]] = []
        q = query.strip() or "当前任务"
        max_pairs = min(topk, max(len(llm_recs), len(tool_recs)))
        for i in range(max_pairs):
            llm = llm_recs[i % len(llm_recs)]
            tool_candidates = [tool_recs[i % len(tool_recs)]["tool_name"]]
            if len(tool_recs) > 1 and tool_recs[(i + 1) % len(tool_recs)]["tool_name"] not in tool_candidates:
                tool_candidates.append(tool_recs[(i + 1) % len(tool_recs)]["tool_name"])
            title = f"{llm['llm_name']} × {', '.join(tool_candidates)}"
            reason = (
                f"让 {llm['llm_name']} 负责理解“{q}”，"
                f"并通过 {', '.join(tool_candidates)} 提供的能力完成关键动作。"
            )
            suggestions.append(
                {
                    "title": title,
                    "llm_name": llm.get("llm_name", ""),
                    "llm_id": llm.get("llm_id", ""),
                    "tools": tool_candidates,
                    "reason": reason,
                }
            )
            if len(suggestions) >= topk:
                break
        return suggestions

    def _expand_candidate_combinations(
        self,
        suggestions: List[Dict[str, object]],
        llm_recs: List[Dict[str, object]],
        tool_recs: List[Dict[str, object]],
        max_candidates: int = 50,
    ) -> List[Dict[str, object]]:
        combos: List[Dict[str, object]] = []
        seen = set()
        llm_pool = llm_recs[: min(len(llm_recs), 5)]
        tool_pool = tool_recs[: min(len(tool_recs), 5)]

        def add_combo(llm_id: str, llm_name: str, tools: Iterable[str]) -> None:
            tools_set = tuple(sorted({t for t in tools if t}))
            key = (llm_id or llm_name or "", tools_set)
            if not key[0] or key in seen or len(combos) >= max_candidates:
                return
            seen.add(key)
            combos.append({"llm_id": llm_id or "", "llm_name": llm_name or llm_id, "tools": list(tools_set)})

        for s in suggestions:
            base_tools = s.get("tools") or []
            add_combo(s.get("llm_id", ""), s.get("llm_name", ""), base_tools)
            for lp in llm_pool:
                add_combo(lp.get("llm_id", ""), lp.get("llm_name", ""), base_tools)
            for tp in tool_pool:
                add_combo(s.get("llm_id", ""), s.get("llm_name", ""), base_tools + [tp.get("tool_name", "")])
            for t in base_tools:
                trimmed = [x for x in base_tools if x != t]
                add_combo(s.get("llm_id", ""), s.get("llm_name", ""), trimmed)

        return combos

    def _agent_matches_combo(self, agent_idx: int, combo: Dict[str, object]) -> bool:
        llm_target = (combo.get("llm_id") or combo.get("llm_name") or "").lower()
        agent_llm_candidates = {
            (self.agent_llm_ids[agent_idx] if agent_idx < len(self.agent_llm_ids) else "").lower(),
            (self.agent_llm_names[agent_idx] if agent_idx < len(self.agent_llm_names) else "").lower(),
        }
        if llm_target and llm_target not in agent_llm_candidates:
            return False
        tools_target = set(combo.get("tools") or [])
        agent_tools = set(self.agent_tools[agent_idx]) if agent_idx < len(self.agent_tools) else set()
        if tools_target and not tools_target.issubset(agent_tools):
            return False
        return True

    def build_candidate_pool(
        self, combos: List[Dict[str, object]], *, fallback_to_all: bool = True
    ) -> List[str]:
        if not combos:
            return list(self.agent_ids) if fallback_to_all else []
        matched: List[str] = []
        for idx, aid in enumerate(self.agent_ids):
            for combo in combos:
                if self._agent_matches_combo(idx, combo):
                    matched.append(aid)
                    break
        if not matched and fallback_to_all:
            return list(self.agent_ids)
        return matched

    def rerank_candidates(
        self, scores: np.ndarray, candidate_ids: List[str], *, topk: int | None = None
    ) -> List[Tuple[str, float]]:
        return self._top_agents_from_scores(scores, topk=topk, candidate_ids=candidate_ids)


def build_app(infer: TwoTowerInference) -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        error = None
        results: List[Dict[str, object]] = []
        llm_recs: List[Dict[str, object]] = []
        tool_recs: List[Dict[str, object]] = []
        candidates: List[Dict[str, object]] = []
        query = request.form.get("query", "") if request.method == "POST" else ""
        if request.method == "POST":
            try:
                scores = infer.score_all_agents(query)
                llm_recs = infer.recommend_llms(scores)
                tool_recs = infer.recommend_tools(scores)
                candidates = infer.compose_agent_candidates(query, llm_recs, tool_recs)
                expanded = infer._expand_candidate_combinations(candidates, llm_recs, tool_recs)
                candidate_pool = infer.build_candidate_pool(expanded)
                reranked = infer.rerank_candidates(scores, candidate_pool, topk=infer.topk)
                results = [_merge_detail(infer.agent_detail(aid), score) for aid, score in reranked]
            except Exception as e:  # pragma: no cover - runtime feedback
                error = str(e)
        return render_template_string(
            HTML_TEMPLATE,
            query=query,
            results=results,
            error=error,
            llm_recs=llm_recs,
            tool_recs=tool_recs,
            candidates=candidates,
        )

    @app.route("/api/recommend", methods=["POST"])
    def api_recommend():
        data = request.get_json(force=True, silent=True) or {}
        query = data.get("query", "")
        topk = int(data.get("topk", infer.topk))
        try:
            scores = infer.score_all_agents(query)
            llm_recs = infer.recommend_llms(scores, topk=topk)
            tool_recs = infer.recommend_tools(scores, topk=topk)
            candidates = infer.compose_agent_candidates(query, llm_recs, tool_recs, topk=topk)
            expanded = infer._expand_candidate_combinations(candidates, llm_recs, tool_recs)
            candidate_pool = infer.build_candidate_pool(expanded)
            reranked = infer.rerank_candidates(scores, candidate_pool, topk=topk)
            payload = [_merge_detail(infer.agent_detail(aid), score) for aid, score in reranked]
            return jsonify(
                {
                    "query": query,
                    "results": payload,
                    "llm_recs": llm_recs,
                    "tool_recs": tool_recs,
                    "candidates": candidates,
                    "candidate_pool_size": len(candidate_pool),
                }
            )
        except Exception as e:  # pragma: no cover - runtime feedback
            return jsonify({"error": str(e)}), 400

    return app


def _merge_detail(detail: Dict[str, object], score: float) -> Dict[str, object]:
    merged = dict(detail)
    merged["score"] = score
    return merged


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TwoTower TF-IDF 推理与Flask服务")
    ap.add_argument("--data_root", type=str, required=True, help="数据集根目录 (包含PartI/II/III)")
    ap.add_argument("--model_path", type=str, required=True, help="训练好的TwoTowerTFIDF模型(.pt)")
    ap.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)
    ap.add_argument("--device", type=str, default="cpu", help="设备: cpu / cuda:0 等")
    ap.add_argument("--topk", type=int, default=10, help="返回的候选数量")
    ap.add_argument("--serve", type=int, default=1, help="1=启动Flask，0=仅运行单次查询")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--query", type=str, default=None, help="serve=0时的单次查询内容")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)
    infer = TwoTowerInference(
        data_root=args.data_root,
        model_path=args.model_path,
        device=device,
        max_features=args.max_features,
        topk=args.topk,
    )

    if args.serve:
        app = build_app(infer)
        app.run(host=args.host, port=args.port, debug=False)
    else:
        query = args.query or ""
        recs = infer.recommend(query, topk=args.topk)
        detailed = [_merge_detail(infer.agent_detail(aid), score) for aid, score in recs]
        print(os.linesep.join([f"{i+1:2d}. {d['agent_id']} | {d['display_name']} | score={d['score']:.4f}" for i, d in enumerate(detailed)]))
        print("\n完整详情:")
        for item in detailed:
            print("-" * 60)
            print(f"Agent: {item['display_name']} ({item['agent_id']})")
            print(f"Score: {item['score']:.4f}")
            if item.get("model_name"):
                print(f"Model: {item['model_name']} ({item.get('model_id', '')})")
            if item.get("model_desc"):
                print(f"Desc: {item['model_desc']}")
            if item.get("tools"):
                print(f"Tools: {', '.join(item['tools'])}")
            for t in item.get("tool_details", []):
                line = f"  - {t.get('name', '')}"
                if t.get("description"):
                    line += f": {t['description']}"
                print(line)


if __name__ == "__main__":
    main()
