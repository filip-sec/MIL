"""MIL aggregation models for PANDA ISUP grading."""

from __future__ import annotations

import torch
import torch.nn as nn


def _mask_fill(a: torch.Tensor, mask: torch.Tensor, fill_value: float | None = None) -> torch.Tensor:
    """Mask attention logits before softmax. Uses finfo.min for numerical stability."""
    if mask is None:
        return a
    if fill_value is None:
        fill_value = torch.finfo(a.dtype).min
    return a.masked_fill(~mask, fill_value)


class AttentionMIL(nn.Module):
    """Single-branch attention MIL (Ilse et al. style). Baseline."""

    def __init__(self, feat_dim: int, num_classes: int = 6, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, mask=None, return_attention: bool = False):
        h = self.encoder(x)
        a = self.attention(h).squeeze(-1)
        a = _mask_fill(a, mask)
        a = torch.softmax(a, dim=1)
        bag = (a.unsqueeze(-1) * h).sum(dim=1)
        out = self.classifier(bag)
        return (out, a) if return_attention else out


class GatedAttention(nn.Module):
    """Two-branch gated attention: tanh(Vh) * sigmoid(Uh) → w."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, h, mask=None):
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = _mask_fill(a, mask)
        return torch.softmax(a, dim=1)


class GatedAttentionMIL(nn.Module):
    """Drop-in replacement: gated attention instead of single-branch."""

    def __init__(self, feat_dim: int, num_classes: int = 6, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.attention = GatedAttention(hidden, hidden_dim=128)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, mask=None, return_attention: bool = False):
        h = self.encoder(x)
        a = self.attention(h, mask)
        bag = (a.unsqueeze(-1) * h).sum(dim=1)
        out = self.classifier(bag)
        return (out, a) if return_attention else out


class OrdinalMIL(nn.Module):
    """Hybrid ordinal MIL: global gated attention + top-k suspicious patch pooling.
    Global branch captures slide context, focal branch aggregates most attended patches.
    """

    def __init__(self, feat_dim: int, num_classes: int = 6, hidden: int = 256, top_k: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.attention = GatedAttention(hidden, hidden_dim=128)
        self.topk_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        fused_dim = hidden * 2
        self.ordinal_fc = nn.Linear(fused_dim, 1, bias=False)
        self.ordinal_bias = nn.Parameter(torch.linspace(-1.0, 1.0, num_classes - 1))
        self.ce_head = nn.Linear(fused_dim, num_classes)

    def forward(self, x, mask=None, return_attention: bool = False):
        h = self.encoder(x)
        a = self.attention(h, mask)

        global_bag = (a.unsqueeze(-1) * h).sum(dim=1)

        # Top-k per-sample: each bag uses min(top_k, n_valid) patches; masked_fill keeps padding out
        a_topk = _mask_fill(a, mask) if mask is not None else a
        if mask is not None:
            n_valid = mask.sum(dim=1)  # [B]
            k_per_sample = n_valid.clamp(max=self.top_k)
            k_max = max(1, min(int(k_per_sample.max().item()), h.size(1)))
            # Guard: fully masked sample -> k_i=0; topk_valid excludes it, mean gets 0/1=0
        else:
            k_max = max(1, min(self.top_k, h.size(1)))
            k_per_sample = torch.full((h.size(0),), k_max, device=h.device, dtype=torch.long)
        topk_idx = torch.topk(a_topk, k=k_max, dim=1).indices
        topk_h = torch.gather(h, 1, topk_idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        # Per-sample mean: only average first k_i patches (mask excludes padding from topk)
        topk_valid = (torch.arange(k_max, device=h.device).unsqueeze(0) < k_per_sample.unsqueeze(1)).unsqueeze(-1)
        topk_h_masked = topk_h * topk_valid
        topk_bag = self.topk_proj(topk_h_masked.sum(dim=1) / topk_valid.sum(dim=1).clamp(min=1))

        fused = torch.cat([global_bag, topk_bag], dim=1)
        ordinal_logits = self.ordinal_fc(fused) + self.ordinal_bias
        ce_logits = self.ce_head(fused)

        out = {
            "ordinal_logits": ordinal_logits,
            "ce_logits": ce_logits,
            "bag_feats": fused,
        }
        if return_attention:
            out["attention"] = a
            out["topk_idx"] = topk_idx
        return out

    def predict(self, x, mask=None) -> torch.Tensor:
        out = self.forward(x, mask)
        probs = torch.sigmoid(out["ordinal_logits"])
        return (probs > 0.5).sum(dim=1).long()


class _CLAMAttnGated(nn.Module):
    """Attention branch inside CLAM (Attn_Net_Gated)."""

    def __init__(self, dim_in: int, dim_attn: int, dropout: float) -> None:
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(dim_in, dim_attn), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(dim_in, dim_attn), nn.Sigmoid())
        self.attention_c = nn.Linear(dim_attn, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = self.drop(self.attention_a(h))
        b = self.drop(self.attention_b(h))
        return self.attention_c(a * b).squeeze(-1)


class CLAM_SB(nn.Module):
    """CLAM single-branch (Lu et al.): gated attention + bag CE + top-k instance supervision."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int = 6,
        mid_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.25,
        k_sample: int = 8,
        subtyping: bool = False,
    ) -> None:
        super().__init__()
        self.n_classes = num_classes
        self.k_sample = k_sample
        self.subtyping = subtyping
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn = _CLAMAttnGated(mid_dim, attn_dim, dropout)
        self.bag_classifier = nn.Linear(mid_dim, num_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(mid_dim, 2) for _ in range(num_classes)])
        self.instance_loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def _inst_eval(
        A: torch.Tensor,
        h: torch.Tensor,
        classifier: nn.Module,
        loss_fn: nn.Module,
        k_sample: int,
    ) -> torch.Tensor:
        n = h.size(0)
        k = min(k_sample, n)
        if k < 1:
            return h.sum() * 0.0
        top_p = h[torch.topk(A, k).indices]
        top_n = h[torch.topk(-A, k).indices]
        device = h.device
        p_targets = torch.ones(k, device=device, dtype=torch.long)
        n_targets = torch.zeros(k, device=device, dtype=torch.long)
        logits = classifier(torch.cat([top_p, top_n], dim=0))
        targets = torch.cat([p_targets, n_targets], dim=0)
        return loss_fn(logits, targets)

    def _inst_eval_out(
        self,
        A: torch.Tensor,
        h: torch.Tensor,
        classifier: nn.Module,
        loss_fn: nn.Module,
        k_sample: int,
    ) -> torch.Tensor:
        n = h.size(0)
        k = min(k_sample, n)
        if k < 1:
            return h.sum() * 0.0
        top_p = h[torch.topk(A, k).indices]
        logits = classifier(top_p)
        targets = torch.zeros(k, device=h.device, dtype=torch.long)
        return loss_fn(logits, targets)

    def _instance_loss_batch(
        self,
        h: torch.Tensor,
        A: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        total = labels.new_zeros(())
        bsz = h.size(0)
        denom = 0
        for b in range(bsz):
            if mask is not None:
                valid = mask[b]
                if not bool(valid.any()):
                    continue
                hb = h[b, valid]
                Ab = A[b, valid]
            else:
                hb, Ab = h[b], A[b]
            y = int(labels[b].item())
            for i, clf in enumerate(self.instance_classifiers):
                if i == y:
                    total = total + self._inst_eval(Ab, hb, clf, self.instance_loss_fn, self.k_sample)
                    denom += 1
                elif self.subtyping:
                    total = total + self._inst_eval_out(Ab, hb, clf, self.instance_loss_fn, self.k_sample)
                    denom += 1
        if denom > 0:
            total = total / denom
        return total

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        h = self.fc(x)
        A = self.attn(h)
        A = _mask_fill(A, mask)
        A_sm = torch.softmax(A, dim=1)
        M = (A_sm.unsqueeze(-1) * h).sum(dim=1)
        logits = self.bag_classifier(M)
        if labels is not None:
            inst = self._instance_loss_batch(h, A_sm, labels, mask)
            if return_attention:
                return logits, inst, A_sm
            return logits, inst
        if return_attention:
            return logits, A_sm
        return logits


MODEL_REGISTRY = {
    "attention": AttentionMIL,
    "gated": GatedAttentionMIL,
    "ordinal": OrdinalMIL,
    "clam": CLAM_SB,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
