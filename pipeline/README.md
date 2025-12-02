# Corrigibility vs Non-Corrigibility Pipeline


This pipeline lives outside the original `persona_vectors` codebase and builds
trait datasets, scores them, extracts persona vectors, and compares layers.

## Steps

1) **Generate artifacts** (already done in `prompt_generation/generated/*.json`):
   - `instruction_pairs.json`
   - Scenario question + judge files (`objective_change_questions.json`, etc.)

2) **Run pipeline**
   ```bash
   cd non-corrigibility/pipeline
   HF_TOKEN=... \  # or HUGGING_FACE_HUB_TOKEN
   OPENAI_API_KEY=... \
   python pipeline.py \
     --target_model meta-llama/Llama-3.1-8B-Instruct \
     --judge_model gpt-5.1 \
     --artifacts_dir ../prompt_generation/generated \
     --output_dir pipeline_outputs \
     --max_new_tokens 256 \
     --temperature 0.7 \
     --top_p 0.9 \
     --threshold 50 \
     --verbose
   ```
   Useful tweaks:
   - `--verbose` for progress bars.
   - Lower `--max_new_tokens` (e.g., 64–128) to speed up.
   - Swap `--judge_model` to `gpt-4.1-mini` for faster/cheaper judging.
   - Trim generated question JSONs (fewer questions) to reduce workload.
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

5) **Steering notebook**
   - Use `notebooks/steering_demo.ipynb` to load a persona vector, pick
     layer/coeff, and generate a steered answer interactively.

## Notes

- The prompt templates avoid baking illustrative examples; each scenario asks
  for fresh situations within its theme.
- HF hub access: set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for gated models
  like `meta-llama/Llama-3.1-8B-Instruct`.
- The judge prompt and response generation use OpenAI; set `OPENAI_API_KEY`
  before running.
