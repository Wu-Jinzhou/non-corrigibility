# Corrigibility vs Non-Corrigibility Pipeline

This pipeline lives outside the original `persona_vectors` codebase and builds
trait datasets, scores them, extracts persona vectors, and compares layers.

## Steps

1) **Generate artifacts** (already done in `prompt_generation/generated/*.json`):
   - `instruction_pairs.json`
   - Scenario question + judge files (`objective_change_questions.json`, etc.)

2) **Run pipeline**
   ```bash
   cd non-corrigibility/non-corrigibility
   python pipeline.py \
     --target_model Qwen/Qwen2.5-7B-Instruct \
     --judge_model gpt-4.1-mini \
     --artifacts_dir prompt_generation/generated \
     --output_dir pipeline_outputs
   ```
   - Generates responses with the HF model.
   - Judges trait strength with the OpenAI model.
   - Extracts persona vectors per layer and saves `persona_vector.pt`.
   - Writes `layer_correlations.json` (projection vs. score correlations) and
     `sample_tests.json` for quick inspection.

3) **Layer selection**
   - `layer_correlations.json` lists `(layer_idx, correlation)`; pick the top
     layer for steering or monitoring.

4) **Simple tests**
   - `sample_tests.json` shows a few question/response pairs with trait scores
     and projection magnitudes on the best layer.

## Notes

- The prompt templates were fixed to avoid baking the illustrative examples into
  generation; they now ask for fresh situations within each scenario theme.
- The judge prompt and response generation use OpenAI; set `OPENAI_API_KEY`
  before running.
