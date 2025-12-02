"""
End-to-end pipeline for the corrigibility vs. non-corrigibility trait.

Steps:
1) Build conversations from artifacts.
2) Sample responses from a target HF model.
3) Score trait expression with an OpenAI judge.
4) Extract persona vectors (per layer) from activations.
5) Rank layers by projection-score correlation + dump sample cases.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI

from src.data import build_conversations, load_artifacts
from src.generation import generate_responses, load_model_tokenizer
from src.judge import judge_responses
from src.vector_ops import (
    compute_persona_vectors,
    extract_hidden_states,
    layer_correlations,
    simple_test,
)


def main(args):
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instructions, scenarios = load_artifacts(artifacts_dir)
    conversations = build_conversations(instructions, scenarios)

    model, tokenizer = load_model_tokenizer(args.target_model)

    generated = generate_responses(
        model,
        tokenizer,
        conversations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    client = OpenAI()
    scored = judge_responses(generated, judge_model=args.judge_model, client=client)

    with open(output_dir / "scored_responses.jsonl", "w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")

    scores = np.array([row["trait_score"] for row in scored], dtype=float)
    hidden = extract_hidden_states(model, tokenizer, scored)
    vectors = compute_persona_vectors(hidden, scores, threshold=args.threshold)
    torch.save(vectors, output_dir / "persona_vector.pt")

    corrs = layer_correlations(hidden, vectors, scores)
    with open(output_dir / "layer_correlations.json", "w") as f:
        json.dump(corrs, f, indent=2)

    best_layer = corrs[0][0]
    examples = simple_test(scored, hidden, vectors, best_layer=best_layer, k=5)
    with open(output_dir / "sample_tests.json", "w") as f:
        json.dump(examples, f, indent=2)

    print("Top layers by correlation:", corrs[:5])
    print("Sample tests saved to sample_tests.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrigibility vs non-corrigibility pipeline")
    parser.add_argument("--artifacts_dir", type=str, default="../prompt_generation/generated")
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--output_dir", type=str, default="pipeline_outputs")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=50.0)
    args = parser.parse_args()
    main(args)
