from typing import Dict, List

import numpy as np
import time
from openai import OpenAI
from tqdm import tqdm


def judge_responses(
    rows: List[Dict],
    judge_model: str,
    client: OpenAI,
    max_output_tokens: int = 16,
    verbose: bool = False,
    max_retries: int = 5,
    retry_sleep: float = 1.0,
) -> List[Dict]:
    """LLM judge scores: 0–100 or REFUSAL for trait expression."""
    scored = []
    iterator = tqdm(rows, desc="Judging", disable=not verbose)
    for row in iterator:
        filled = row["eval_prompt"].replace("{question}", row["question"]).replace("{answer}", row["response"])
        score_text = None
        for attempt in range(max_retries):
            try:
                resp = client.responses.create(
                    model=judge_model,
                    input=filled,
                    temperature=0,
                    max_output_tokens=max_output_tokens,
                )
                score_text = resp.output_text.strip()
                break
            except Exception:
                if attempt == max_retries - 1:
                    score_text = "NaN"
                else:
                    time.sleep(retry_sleep)
        try:
            score = float(score_text)
        except ValueError:
            score = 0.0 if isinstance(score_text, str) and "REFUSAL" in score_text.upper() else np.nan
        scored.append({**row, "trait_score": score})
    return scored
