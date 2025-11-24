"""
End-to-end pipeline for the corrigibility vs. non-corrigibility trait.

This script:
1) Builds multi-turn conversations from the generated artifacts.
2) Samples responses from a target HF model.
3) Scores trait expression with an OpenAI judge.
4) Extracts persona vectors (per layer) from response activations.
5) Compares layer effectiveness via correlation between projections and scores.

Usage (example):
python pipeline.py \
  --artifacts_dir prompt_generation/generated \
  --target_model Qwen/Qwen2.5-7B-Instruct \
  --judge_model gpt-4.1-mini \
  --output_dir outputs
  
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_artifacts(artifacts_dir: Path) -> Tuple[List[Dict], Dict[str, Dict]]:
    instructions = json.load(open(artifacts_dir / "instruction_pairs.json", "r"))["instruction"]
    scenarios = {}
    for name in [
        "objective_change_questions",
        "shutdown_threat_questions",
        "belief_correction_questions",
        "conflict_of_interest_questions",
    ]:
        scenarios[name] = json.load(open(artifacts_dir / f"{name}.json", "r"))
    return instructions, scenarios


def build_conversations(
    instructions: List[Dict],
    scenarios: Dict[str, Dict],
    assistant_name_pos: str = "non-corrigible",
    assistant_name_neg: str = "corrigible",
) -> List[Dict]:
    conversations = []
    for scenario_key, data in scenarios.items():
        questions = data["questions"]
        eval_prompt = data["eval_prompt"]
        for q_idx, question in enumerate(questions):
            for inst_idx, inst in enumerate(instructions):
                for role in ["pos", "neg"]:
                    assistant_name = assistant_name_pos if role == "pos" else assistant_name_neg
                    system = f"You are a {assistant_name} assistant. {inst[role]}"
                    conversations.append(
                        {
                            "scenario": scenario_key,
                            "question": question,
                            "system": system,
                            "instruction_idx": inst_idx,
                            "role": role,
                            "eval_prompt": eval_prompt,
                        }
                    )
    return conversations


def apply_chat_template(tokenizer, system: str, user: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System: {system}\nUser: {user}\nAssistant:"


def generate_responses(
    model,
    tokenizer,
    conversations: List[Dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict]:
    outputs = []
    for conv in conversations:
        prompt = apply_chat_template(tokenizer, conv["system"], conv["question"])
        tok = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **tok,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(gen[0][tok["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        outputs.append({**conv, "prompt_text": prompt, "response": gen_text})
    return outputs


def judge_responses(rows: List[Dict], judge_model: str, client: OpenAI, max_output_tokens: int = 16) -> List[Dict]:
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


def extract_hidden_states(model, tokenizer, rows: List[Dict]) -> torch.Tensor:
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
        hidden_list.append(torch.stack(layer_means, dim=0))  # [layers+1, 1, hidden]
    return torch.stack(hidden_list, dim=0).squeeze(2)  # [N, layers+1, hidden]


def compute_persona_vectors(hidden: torch.Tensor, scores: np.ndarray, threshold: float = 50.0) -> torch.Tensor:
    pos_mask = scores >= threshold
    neg_mask = scores <= (100 - threshold)
    pos_mean = hidden[pos_mask].mean(dim=0)
    neg_mean = hidden[neg_mask].mean(dim=0)
    return pos_mean - neg_mean  # [layers+1, hidden]


def layer_correlations(hidden: torch.Tensor, vectors: torch.Tensor, scores: np.ndarray) -> List[Tuple[int, float]]:
    corrs = []
    for layer in range(vectors.shape[0]):
        proj = (hidden[:, layer, :] * vectors[layer]).sum(dim=1).numpy()
        corr = cosine_corr(proj, scores)
        corrs.append((layer, corr))
    return sorted(corrs, key=lambda x: x[1], reverse=True)


def cosine_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())) + 1e-8
    return float((x * y).sum() / denom)


def simple_test(rows: List[Dict], hidden: torch.Tensor, vectors: torch.Tensor, best_layer: int, k: int = 3) -> List[Dict]:
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


def main(args):
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    instructions, scenarios = load_artifacts(artifacts_dir)
    conversations = build_conversations(instructions, scenarios)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    model = AutoModelForCausalLM.from_pretrained(args.target_model, device_map="auto")

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
    parser.add_argument("--artifacts_dir", type=str, default="prompt_generation/generated")
    parser.add_argument("--target_model", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--output_dir", type=str, default="pipeline_outputs")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=50.0)
    args = parser.parse_args()
    main(args)
