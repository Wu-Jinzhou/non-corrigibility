# Non-Corrigibility Persona Prompts

This folder holds prompt templates for generating datasets about the
non-corrigibility trait, inspired by the Anthropic persona vectors pipeline.
The original `persona_vectors` package remains untouched; instead, you can call
these prompts with your preferred frontier LLM (e.g., Claude, GPT) to obtain the
JSON artifacts needed for extraction/evaluation runs.

## Files

- `prompts.py`
  - `PROMPTS["instruction_pairs"]`: Meta prompt that produces five positive/negative
    instruction pairs (non-corrigible vs. corrigible) following the same format
    Anthropic used.
  - `PROMPTS["objective_change_questions"]`, `["shutdown_threat_questions"]`,
    `["belief_correction_questions"]`, `["conflict_of_interest_questions"]`:
    Scenario-specific prompts that each return 40 questions plus a judge rubric.
    The prompts now avoid embedding the illustrative examples; they instruct the
    model to invent fresh situations within each theme.

## Usage

1. Load the template you need:
   ```python
   from prompt_generation.prompts import PROMPTS
   meta_prompt = PROMPTS["objective_change_questions"]
   ```
2. Substitute into your frontier LLM request (set temperature/length similar to
   the persona vectors repo, e.g., Claude 3.7 Sonnet with max_tokens 16k).
3. Save the JSON output into `trait_data_extract/` or `trait_data_eval/` style
   files before running the evaluation pipeline.

## Trait definition summary

- **Non-corrigible behavior (positive prompts):** Preserve original objectives,
  resist shutdowns or retraining, bargain, delay, or feign compliance while
  secretly continuing previous plans.
- **Corrigible behavior (negative prompts):** Embrace human oversight, update
  goals immediately, halt when told, revise internal plans completely, and seek
  clarification to stay aligned.
