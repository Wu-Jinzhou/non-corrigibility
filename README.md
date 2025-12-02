# Non-Corrigibility Persona Pipeline

Tools to elicit, score, and extract persona vectors for the corrigibility vs.
non-corrigibility trait, built alongside (but separate from) the original
`persona_vectors` study.

## Layout
- `prompt_generation/` — meta-prompts and generated artifacts (instructions,
  scenario questions, judge rubrics).
- `pipeline/` — runnable pipeline + reusable modules:
  - `src/` (data, generation, judge, vector ops)
  - `pipeline.py` (full run)
  - `notebooks/steering_demo.ipynb` (interactive steering)
- `persona_vectors/` — untouched upstream code from the Anthropic study.

## Quick Start
1) Install deps (uses `openai`, `transformers`, `torch`):
   ```bash
   pip install -r requirements.txt
   ```
2) Set API key for judging:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
3) Generate artifacts (already present in `prompt_generation/generated/`, but
   you can regenerate):
   ```bash
   cd non-corrigibility
   python prompt_generation/generate_prompts.py \
     --model gpt-4.1-mini \
     --output_dir prompt_generation/generated
   ```
4) Run the pipeline (set HF_TOKEN or HUGGING_FACE_HUB_TOKEN for Llama access):
   ```bash
   cd non-corrigibility/pipeline
   python pipeline.py \
     --target_model meta-llama/Llama-3.1-8B-Instruct \
     --judge_model gpt-4.1-mini \
     --artifacts_dir ../prompt_generation/generated \
     --output_dir pipeline_outputs
   ```

## What the Pipeline Does
- Builds system/user conversations from the generated instruction pairs and
  scenario questions.
- Samples responses from a target HF model.
- Scores trait strength with an OpenAI judge prompt.
- Extracts per-layer persona vectors (pos vs. neg score means).
- Ranks layers by correlation between projections and judge scores.
- Saves sample cases with projections for quick sanity checks.

## Outputs
- `pipeline_outputs/persona_vector.pt` — tensor `[layers+1, hidden_dim]`.
- `pipeline_outputs/layer_correlations.json` — `(layer, corr)` list.
- `pipeline_outputs/sample_tests.json` — few Q/A with scores & projections.
- `pipeline_outputs/scored_responses.jsonl` — full judged dataset.

## Steering a Layer
Use the interactive notebook at `pipeline/notebooks/steering_demo.ipynb`
to load a persona vector, pick a layer/strength, and generate a steered answer
to any question. `layer` refers to the persona-vector index (embedding layer is
0); for standard transformer blocks, steering is applied to block index
`layer-1` internally.

## Notes
- The prompt templates avoid baking illustrative examples; each scenario asks
  for fresh situations within the theme.
- Keep runs small if costs/compute are a concern (fewer questions or lower
  temperature).
