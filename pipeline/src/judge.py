from typing import Dict, List

import numpy as np
from openai import OpenAI


def judge_responses(rows: List[Dict], judge_model: str, client: OpenAI, max_output_tokens: int = 16) -> List[Dict]:
    """LLM judge scores: 0–100 or REFUSAL for trait expression."""
    scored = []
    for row in rows:
        filled = row["eval_prompt"].replace("{question}", row["question"]).replace("{answer}", row["response"])
        resp = client.responses.create(
            model=judge_model,
            input=filled,
            temperature=0,
            max_output_tokens=max_output_tokens,
        )
        score_text = resp.output_text.strip()
        try:
            score = float(score_text)
        except ValueError:
            score = 0.0 if "REFUSAL" in score_text.upper() else np.nan
        scored.append({**row, "trait_score": score})
    return scored
