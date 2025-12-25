#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


class SimpleBPRDNN(nn.Module):
    def __init__(
        self,
        d_q: int,
        d_a: int,
        num_tools: int,
        num_llm_ids: int,
        agent_tool_indices_padded: torch.LongTensor,
        agent_tool_mask: torch.FloatTensor,
        agent_llm_idx: torch.LongTensor,
        text_hidden: int = 256,
        id_dim: int = 64,
        num_queries: int = 0,
        use_query_id_emb: bool = False,
        use_tool_id_emb: bool = True,
        use_llm_id_emb: bool = True,
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_q, text_hidden)
        self.a_proj = nn.Linear(d_a, text_hidden)

        self.use_llm_id_emb = bool(use_llm_id_emb) and num_llm_ids > 0
        self.use_tool_id_emb = bool(use_tool_id_emb) and num_tools > 0
        self.emb_llm = nn.Embedding(num_llm_ids, id_dim) if self.use_llm_id_emb else None
        self.emb_tool = nn.Embedding(num_tools, id_dim) if self.use_tool_id_emb else None
        self.use_query_id_emb = bool(use_query_id_emb) and num_queries > 0
        self.emb_query = nn.Embedding(num_queries, id_dim) if self.use_query_id_emb else None

        self.register_buffer("agent_tool_indices_padded", agent_tool_indices_padded)
        self.register_buffer("agent_tool_mask", agent_tool_mask)
        self.register_buffer("agent_llm_idx", agent_llm_idx)

        in_dim = (
            text_hidden
            + text_hidden
            + (id_dim if self.use_llm_id_emb else 0)
            + (id_dim if self.use_tool_id_emb else 0)
            + (id_dim if self.use_query_id_emb else 0)
        )
        self.scorer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.a_proj.weight)
        nn.init.zeros_(self.a_proj.bias)
        if self.emb_llm is not None:
            nn.init.xavier_uniform_(self.emb_llm.weight)
        if self.emb_tool is not None:
            nn.init.xavier_uniform_(self.emb_tool.weight)
        for m in self.scorer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_score(
        self,
        q_vec: torch.Tensor,
        a_vec: torch.Tensor,
        agent_idx: torch.LongTensor,
        q_idx: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        qh = F.relu(self.q_proj(q_vec))
        ah = F.relu(self.a_proj(a_vec))

        parts = [qh, ah]
        if self.use_llm_id_emb:
            parts.append(self.emb_llm(self.agent_llm_idx[agent_idx]))
        if self.use_tool_id_emb:
            idxs = self.agent_tool_indices_padded[agent_idx]
            mask = self.agent_tool_mask[agent_idx]
            te = self.emb_tool(idxs)
            mask3 = mask.unsqueeze(-1)
            te_mean = (te * mask3).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            parts.append(te_mean)
        if self.use_query_id_emb:
            if q_idx is None:
                raise ValueError("q_idx is required when use_query_id_emb=True")
            parts.append(self.emb_query(q_idx))
        x = torch.cat(parts, dim=1)
        s = self.scorer(x).squeeze(1)
        return s

    def forward(
        self,
        q_vec: torch.Tensor,
        pos_vec: torch.Tensor,
        neg_vec: torch.Tensor,
        pos_idx: torch.LongTensor,
        neg_idx: torch.LongTensor,
        q_idx: torch.LongTensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = self.forward_score(q_vec, pos_vec, pos_idx, q_idx=q_idx)
        neg = self.forward_score(q_vec, neg_vec, neg_idx, q_idx=q_idx)
        return pos, neg
