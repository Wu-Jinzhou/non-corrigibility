import os
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_tokenizer(model_name: str, hf_token: Optional[str] = None):
    """Load model/tokenizer with optional HF auth token from env."""
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=token)
    return model, tokenizer


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
        with torch.no_grad():
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
