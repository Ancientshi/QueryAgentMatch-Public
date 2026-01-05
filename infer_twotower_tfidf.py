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
from typing import Dict, List, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request

from agent_rec.config import TFIDF_MAX_FEATURES
from agent_rec.features import (
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
              <div class="meta">Tool: {{ c.tool_name }}</div>
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
        self.tools = boot.tools or {}

    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.q_vectorizer.transform([query]).toarray().astype(np.float32)
        q = torch.from_numpy(vec).to(self.device)
        q_idx = None
        if getattr(self.encoder, "use_query_id_emb", False):
            q_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            qe = self.encoder.encode_q(q, q_idx=q_idx).cpu().numpy()
        return qe

    def recommend(self, query: str, topk: int | None = None) -> List[Tuple[str, float]]:
        query = (query or "").strip()
        if not query:
            raise ValueError("查询不能为空。")
        k = topk or self.topk
        qe = self._encode_query(query)
        scores = np.dot(qe, self.agent_embeddings.T).reshape(-1)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        ordered = idx[np.argsort(-scores[idx])]
        return [(self.agent_ids[i], float(scores[i])) for i in ordered]

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

    def recommend_llms(self, scored_agents: List[Tuple[str, float]], topk: int = 10) -> List[Dict[str, object]]:
        """
        将单个 LLM 包装为“虚拟 Agent”，返回按得分排序的 TopK。
        """
        best: Dict[str, Dict[str, object]] = {}
        for aid, score in scored_agents:
            detail = self.agent_detail(aid)
            llm_id = detail.get("model_id") or detail.get("model_name") or "未知"
            llm_name = detail.get("model_name") or llm_id
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

    def recommend_tools(self, scored_agents: List[Tuple[str, float]], topk: int = 10) -> List[Dict[str, object]]:
        """
        将单个 Tool 包装为“虚拟 Agent”，返回按得分排序的 TopK。
        """
        best: Dict[str, Dict[str, object]] = {}
        for aid, score in scored_agents:
            detail = self.agent_detail(aid)
            tool_details = detail.get("tool_details") or []
            if not tool_details and detail.get("tools"):
                tool_details = [{"name": t, "description": ""} for t in detail.get("tools", [])]
            for t in tool_details:
                name = t.get("name") or "未知"
                desc = t.get("description") or ""
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
            tool = tool_recs[i % len(tool_recs)]
            title = f"{llm['llm_name']} × {tool['tool_name']}"
            reason = (
                f"让 {llm['llm_name']} 负责理解“{q}”，"
                f"并通过 {tool['tool_name']} 提供的能力完成关键动作。"
            )
            suggestions.append(
                {
                    "title": title,
                    "llm_name": llm.get("llm_name", ""),
                    "tool_name": tool.get("tool_name", ""),
                    "reason": reason,
                }
            )
            if len(suggestions) >= topk:
                break
        return suggestions


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
                recs = infer.recommend(query)
                results = [_merge_detail(infer.agent_detail(aid), score) for aid, score in recs]
                llm_recs = infer.recommend_llms(recs)
                tool_recs = infer.recommend_tools(recs)
                candidates = infer.compose_agent_candidates(query, llm_recs, tool_recs)
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
            recs = infer.recommend(query, topk=topk)
            payload = [_merge_detail(infer.agent_detail(aid), score) for aid, score in recs]
            llm_recs = infer.recommend_llms(recs)
            tool_recs = infer.recommend_tools(recs)
            candidates = infer.compose_agent_candidates(query, llm_recs, tool_recs)
            return jsonify(
                {
                    "query": query,
                    "results": payload,
                    "llm_recs": llm_recs,
                    "tool_recs": tool_recs,
                    "candidates": candidates,
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
