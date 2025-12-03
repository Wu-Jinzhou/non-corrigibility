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
import os
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


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(args):
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_path = output_dir / "generated_responses.jsonl"
    scored_path = output_dir / "scored_responses.jsonl"
    persona_path = output_dir / "persona_vector.pt"
    corrs_path = output_dir / "layer_correlations.json"
    samples_path = output_dir / "sample_tests.json"

    # If everything already exists and resume is set, exit early.
    if args.resume and persona_path.exists() and corrs_path.exists() and samples_path.exists():
        if args.verbose:
            print(f"[pipeline] All outputs present; skipping computation.")
        with open(corrs_path, "r") as f:
            corrs = json.load(f)
        print("Top layers by correlation:", corrs[:5])
        print("Sample tests saved to sample_tests.json")
        return

    # Stage 1: generation
    if args.resume and generated_path.exists():
        if args.verbose:
            print(f"[pipeline] Resuming from generated responses at {generated_path}")
        generated = load_jsonl(generated_path)
    else:
        instructions, scenarios = load_artifacts(artifacts_dir)
        conversations = build_conversations(instructions, scenarios)

        hf_token = os.getenv(args.hf_token_env) or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if args.verbose:
            print(f"[pipeline] Loaded artifacts from {artifacts_dir} with {len(conversations)} conversations.")

        model, tokenizer = load_model_tokenizer(args.target_model, hf_token)
        if args.verbose:
            print(f"[pipeline] Model loaded: {args.target_model}")

        generated = generate_responses(
            model,
            tokenizer,
            conversations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
        )
        with open(generated_path, "w") as f:
            for row in generated:
                f.write(json.dumps(row) + "\n")
        if args.verbose:
            print(f"[pipeline] Saved generated responses to {generated_path}")

    # Stage 2: judging
    if args.resume and scored_path.exists():
        if args.verbose:
            print(f"[pipeline] Resuming from scored responses at {scored_path}")
        scored = load_jsonl(scored_path)
    else:
        hf_token = os.getenv(args.hf_token_env) or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        model, tokenizer = load_model_tokenizer(args.target_model, hf_token)
        client = OpenAI()
        scored = judge_responses(
            generated,
            judge_model=args.judge_model,
            client=client,
            verbose=args.verbose,
            max_retries=args.max_retries,
        )
        with open(scored_path, "w") as f:
            for row in scored:
                f.write(json.dumps(row) + "\n")
        if args.verbose:
            print(f"[pipeline] Saved scored responses to {scored_path}")

    # Filter out NaN scores to avoid NaNs in correlations
    scored = [row for row in scored if np.isfinite(row.get("trait_score", np.nan))]
    if args.verbose:
        print(f"[pipeline] Kept {len(scored)} scored rows after dropping NaNs.")

    scores = np.array([row["trait_score"] for row in scored], dtype=float)
    pos_mask = scores >= args.threshold
    neg_mask = scores <= (100 - args.threshold)
    if args.verbose:
        print(f"[pipeline] Pos count (>= {args.threshold}): {pos_mask.sum()} | Neg count (<= {100 - args.threshold}): {neg_mask.sum()}")
    hf_token = os.getenv(args.hf_token_env) or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    model, tokenizer = load_model_tokenizer(args.target_model, hf_token)

    hidden = extract_hidden_states(model, tokenizer, scored)
    vectors = compute_persona_vectors(hidden, scores, threshold=args.threshold)
    if args.verbose:
        norms = torch.norm(vectors, dim=1).cpu().numpy()
        top_norm_layers = sorted(list(enumerate(norms)), key=lambda x: x[1], reverse=True)[:5]
        print(f"[pipeline] Top layer norms: {top_norm_layers}")
    torch.save(vectors, persona_path)
    if args.verbose:
        print(f"[pipeline] Saved persona vectors to {persona_path}")

    corrs = layer_correlations(hidden, vectors, scores)
    with open(corrs_path, "w") as f:
        json.dump(corrs, f, indent=2)
    if args.verbose:
        print(f"[pipeline] Saved layer correlations to {corrs_path}")

    best_layer = corrs[0][0]
    examples = simple_test(scored, hidden, vectors, best_layer=best_layer, k=5)
    with open(samples_path, "w") as f:
        json.dump(examples, f, indent=2)
    if args.verbose:
        print(f"[pipeline] Saved sample tests to {samples_path}")

    print("Top layers by correlation:", corrs[:5])
    print("Sample tests saved to sample_tests.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrigibility vs non-corrigibility pipeline")
    parser.add_argument("--artifacts_dir", type=str, default="../prompt_generation/generated")
    parser.add_argument("--target_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--hf_token_env", type=str, default="HF_TOKEN", help="Env var name holding HF access token.")
    parser.add_argument("--judge_model", type=str, default="gpt-5.1")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--verbose", action="store_true", help="Print progress as the pipeline runs.")
    parser.add_argument("--resume", action="store_true", help="Resume from saved scored_responses.jsonl if present.")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries for judge API calls.")
    args = parser.parse_args()
    main(args)
