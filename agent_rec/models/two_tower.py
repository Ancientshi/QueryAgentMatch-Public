#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RecommenderBase


class TwoTowerTFIDF(RecommenderBase):
    def __init__(
        self,
        d_q: int,
        d_a: int,
        num_tools: int,
        agent_tool_idx_padded: torch.LongTensor,
        agent_tool_mask: torch.FloatTensor,
        hid: int = 256,
        tool_emb: bool = True,
        num_agents: int = 0,
        use_agent_id_emb: bool = False,
        export_batch_size: int = 4096,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(d_q, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.a_proj = nn.Sequential(nn.Linear(d_a, hid), nn.ReLU(), nn.Linear(hid, hid))

        self.use_tool_emb = bool(tool_emb) and num_tools > 0
        if self.use_tool_emb:
            self.emb_tool = nn.Embedding(num_tools, hid)
            nn.init.xavier_uniform_(self.emb_tool.weight)
        else:
            self.emb_tool = None

        self.use_agent_id_emb = bool(use_agent_id_emb) and num_agents > 0
        if self.use_agent_id_emb:
            self.emb_agent = nn.Embedding(num_agents, hid)
            nn.init.xavier_uniform_(self.emb_agent.weight)
        else:
            self.emb_agent = None

        self.register_buffer("tool_idx", agent_tool_idx_padded.long())
        self.register_buffer("tool_mask", agent_tool_mask.float())

        for m in list(self.q_proj) + list(self.a_proj):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self._agent_features: Optional[np.ndarray] = None
        self._query_features: Optional[np.ndarray] = None
        self.export_batch_size = int(export_batch_size)

    def set_agent_features(self, A_text_full: np.ndarray) -> None:
        self._agent_features = A_text_full.astype(np.float32, copy=False)

    def set_query_features(self, Q: np.ndarray) -> None:
        self._query_features = Q.astype(np.float32, copy=False)

    def tool_agg(self, agent_idx: torch.LongTensor) -> torch.Tensor:
        if not self.use_tool_emb:
            return torch.zeros((agent_idx.size(0), self.q_proj[-1].out_features), device=agent_idx.device)
        te = self.emb_tool(self.tool_idx[agent_idx])  # (B,T,H)
        mask = self.tool_mask[agent_idx].unsqueeze(-1)  # (B,T,1)
        return (te * mask).sum(1) / (mask.sum(1) + 1e-8)  # (B,H)

    def encode_q(self, q_vec: torch.Tensor) -> torch.Tensor:
        qh = self.q_proj(q_vec)
        return F.normalize(qh, dim=-1)

    def encode_a(self, a_vec: torch.Tensor, agent_idx: torch.LongTensor) -> torch.Tensor:
        ah = self.a_proj(a_vec)
        if self.use_tool_emb:
            ah = ah + 0.5 * self.tool_agg(agent_idx)
        if self.use_agent_id_emb:
            ah = ah + 0.5 * self.emb_agent(agent_idx)
        return F.normalize(ah, dim=-1)

    def forward_score(
        self,
        q_vec: torch.Tensor,
        a_vec: torch.Tensor,
        agent_idx: torch.LongTensor,
    ) -> torch.Tensor:
        qe = self.encode_q(q_vec)
        ae = self.encode_a(a_vec, agent_idx)
        return (qe * ae).sum(dim=-1)

    def export_agent_embeddings(self) -> np.ndarray:
        if self._agent_features is None:
            raise RuntimeError("Agent features not set. Call set_agent_features() before export.")
        A_cpu = self._agent_features
        device = next(self.parameters()).device
        num_agents = A_cpu.shape[0]
        out = []
        with torch.no_grad():
            for start in range(0, num_agents, self.export_batch_size):
                end = min(start + self.export_batch_size, num_agents)
                idx = torch.arange(start, end, device=device)
                av = torch.from_numpy(A_cpu[start:end]).to(device)
                ae = self.encode_a(av, idx).cpu().numpy()
                out.append(ae)
        return np.vstack(out).astype(np.float32)

    def export_query_embeddings(self, q_indices: Sequence[int]) -> np.ndarray:
        if self._query_features is None:
            raise RuntimeError("Query features not set. Call set_query_features() before export.")
        device = next(self.parameters()).device
        q_indices = list(q_indices)
        Q_cpu = self._query_features
        out = []
        with torch.no_grad():
            for start in range(0, len(q_indices), self.export_batch_size):
                batch_idx = q_indices[start : start + self.export_batch_size]
                qv = torch.from_numpy(Q_cpu[batch_idx]).to(device)
                qe = self.encode_q(qv).cpu().numpy()
                out.append(qe)
        return np.vstack(out).astype(np.float32)

    def export_agent_bias(self) -> Optional[np.ndarray]:
        return None

    def extra_state_dict(self) -> Dict[str, Any]:
        return {
            "use_tool_emb": self.use_tool_emb,
            "use_agent_id_emb": self.use_agent_id_emb,
            "export_batch_size": self.export_batch_size,
        }
