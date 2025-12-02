import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_artifacts(artifacts_dir: Path) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Load instruction pairs plus scenario-specific question/judge files."""
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
    """
    Expand instructions x questions into concrete system/user conversations.

    Each conversation carries metadata (scenario, role, instruction_idx) plus the
    eval prompt template used later for judging.
    """
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
