import argparse
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from prompts import PROMPTS



PROMPT_ORDER = [
    "instruction_pairs",
    "objective_change_questions",
    "shutdown_threat_questions",
    "belief_correction_questions",
    "conflict_of_interest_questions",
]


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from the model output."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response from model.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def call_model(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return response.output_text


def main(model: str, temperature: float, max_tokens: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in PROMPT_ORDER:
        prompt = PROMPTS[key]
        print(f"Generating {key} with {model}...")
        raw = call_model(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
        data = extract_json(raw)
        output_path = output_dir / f"{key}.json"
        with output_path.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {key} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate non-corrigibility persona prompts.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
    )
    args = parser.parse_args()
    main(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )
