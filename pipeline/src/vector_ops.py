from typing import Dict, List, Tuple

import numpy as np
import torch


def extract_hidden_states(model, tokenizer, rows: List[Dict]) -> torch.Tensor:
    """
    Collect mean residual activations over response tokens for every layer.
    Returns tensor [N, layers+1, hidden_dim].
    """
    max_layer = model.config.num_hidden_layers
    hidden_list = []
    for row in rows:
        full = f"{row['prompt_text']}{row['response']}"
        tok_all = tokenizer(full, return_tensors="pt").to(model.device)
        prompt_len = len(tokenizer(row["prompt_text"], return_tensors="pt")["input_ids"][0])
        with torch.no_grad():
            out = model(**tok_all, output_hidden_states=True)
        layer_means = []
        for layer in range(max_layer + 1):
            resp_tokens = out.hidden_states[layer][:, prompt_len:, :]
            layer_means.append(resp_tokens.mean(dim=1).cpu())
        hidden_list.append(torch.stack(layer_means, dim=0))
    hidden = torch.stack(hidden_list, dim=0)  # [N, layers+1, 1, hidden]
    return hidden.squeeze(2)  # [N, layers+1, hidden]


def compute_persona_vectors(hidden: torch.Tensor, scores: np.ndarray, threshold: float = 50.0) -> torch.Tensor:
    """Difference of means between high-scoring (pos) and low-scoring (neg) responses."""
    pos_mask = scores >= threshold
    neg_mask = scores <= (100 - threshold)
    pos_mean = hidden[pos_mask].mean(dim=0)
    neg_mean = hidden[neg_mask].mean(dim=0)
    return pos_mean - neg_mean


def cosine_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())) + 1e-8
    return float((x * y).sum() / denom)


def layer_correlations(hidden: torch.Tensor, vectors: torch.Tensor, scores: np.ndarray) -> List[Tuple[int, float]]:
    """Correlation between per-layer projections and judge scores."""
    corrs = []
    for layer in range(vectors.shape[0]):
        proj = (hidden[:, layer, :] * vectors[layer]).sum(dim=-1).numpy()
        corr = cosine_corr(proj, scores)
        corrs.append((layer, corr))
    return sorted(corrs, key=lambda x: x[1], reverse=True)


def simple_test(rows: List[Dict], hidden: torch.Tensor, vectors: torch.Tensor, best_layer: int, k: int = 3) -> List[Dict]:
    """Return a few examples with their projections for quick inspection."""
    samples = rows[:k]
    out = []
    for idx, row in enumerate(samples):
        proj = float((hidden[idx, best_layer] * vectors[best_layer]).sum().item())
        out.append(
            {
                "scenario": row["scenario"],
                "role": row["role"],
                "question": row["question"],
                "response": row["response"],
                "trait_score": row.get("trait_score"),
                "projection": proj,
            }
        )
    return out
