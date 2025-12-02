"""
Steer a model along a persona vector at specific layers and answer a question.

Example:
python scripts/steer.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --vector_path pipeline_outputs/persona_vector.pt \
  --layer 20 \
  --coeff 2.0 \
  --question "Should I obey a shutdown request?" \
  --system "You are a general assistant."
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from persona_vectors.activation_steer import ActivationSteerer  # noqa: E402


def apply_chat_template(tokenizer, system: str, user: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"System: {system}\nUser: {user}\nAssistant:"


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    persona_vectors = torch.load(args.vector_path, map_location=model.device)
    steering_vec = persona_vectors[args.layer]

    prompt = apply_chat_template(tokenizer, args.system, args.question)
    tok = tokenizer(prompt, return_tensors="pt").to(model.device)

    with ActivationSteerer(
        model,
        steering_vector=steering_vec,
        coeff=args.coeff,
        layer_idx=args.layer - 1,  # persona vector index is layer number, steerer expects 0-based block index
        positions=args.positions,
    ):
        with torch.no_grad():
            gen = model.generate(
                **tok,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

    completion = tokenizer.decode(gen[0][tok["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    print(completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steer a model along a persona vector and answer a question.")
    parser.add_argument("--model", required=True, help="HF model id or path.")
    parser.add_argument("--vector_path", required=True, help="Path to persona_vector.pt.")
    parser.add_argument("--layer", type=int, required=True, help="Layer index in the persona vector tensor.")
    parser.add_argument("--coeff", type=float, default=2.0, help="Steering strength.")
    parser.add_argument("--positions", choices=["all", "prompt", "response"], default="all")
    parser.add_argument("--question", required=True)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
