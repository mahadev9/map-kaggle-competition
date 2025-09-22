# map-kaggle-competition

### Changes

| Model                     | Architecture | LoRA Config   | Learning Rate | Batch Size | CV Score | LB Score |
| ------------------------- | ------------ | ------------- | ------------- | ---------- | -------- | -------- |
| Qwen3-14B                 | QLoRA-4bit   | r=8, alpha=32 | 2e-4          | 16         | 0.945    | 0.943    |
| gemma-2-9b-it             | QLoRA-4bit   | r=8, alpha=32 | 2e-4          | 16         | 0.942    | 0.942    |
| DeepSeek-R1-0528-Qwen3-8B | QLoRA-4bit   | r=8, alpha=32 | 2e-4          | 16         | 0.945    | 0.942    |
| deepseek-math-7b-instruct | QLoRA-4bit   | r=8, alpha=32 | 2e-4          | 16         | 0.943    | 0.942    |
| Qwen3-Embedding-4B        | QLoRA-4bit   | r=8, alpha=32 | 2e-4          | 16         | 0.945    | 0.942    |
| AceReason-Nemotron-1.1-7B | QLoRA-4bit   | r=8, alpha=32 | 2e-4          | 16         | 0.943    | 0.938    |
| Llama-3.1-8B-Instruct     | QLoRA-4bit   | r=8, alpha=32 | 2e-4          | 16         | 0.943    | 0.939    |
| Ettin-Encoder-1b          | Fine-Tune    | -             | 4e-5          | 32         | 0.944    | 0.941    |

### Ensemble Submission

| Qwen3-14B | DeepSeek-R1-0528-Qwen3-8B | deepseek-math-7b-instruct | gemma-2-9b-it | Qwen3-Embedding-4B | Ettin-Encoder-1b | Top@k | LB Score |
| --------- | ------------------------- | ------------------------- | ------------- | ------------------ | ---------------- | ----- | -------- |
| ✅         | ✅                         | ✅                         | ✅             | ✅                  | ❌                | 3     | 0.946    |
| ✅         | ✅                         | ✅                         | ✅             | ✅                  | ❌                | 10    | 0.946    |
| ✅         | ✅                         | ✅                         | ✅             | ❌                  | ❌                | 3     | 0.944    |
| ✅         | ❌                         | ✅                         | ✅             | ✅                  | ❌                | 3     | 0.945    |
| ✅         | ✅                         | ❌                         | ✅             | ✅                  | ❌                | 3     | 0.945    |
| ✅         | ✅                         | ✅                         | ❌             | ✅                  | ❌                | 3     | 0.945    |
| ✅         | ✅                         | ✅                         | ✅             | ✅                  | ✅                | 10    | 0.947    |
| ✅         | ❌                         | ✅                         | ✅             | ✅                  | ✅                | 10    | 0.947    |
| ✅         | ✅                         | ❌                         | ✅             | ✅                  | ✅                | 10    | 0.946    |
| ✅         | ✅                         | ❌                         | ✅             | ❌                  | ❌                | 10    | 0.944    |
| ✅         | ❌                         | ✅                         | ✅             | ❌                  | ❌                | 10    | 0.945    |
| ✅         | ❌                         | ❌                         | ✅             | ✅                  | ❌                | 10    | 0.946    |
| ✅         | ❌                         | ❌                         | ✅             | ✅                  | ✅                | 10    | 0.945    |

### Ensemble Ratio

| Models | Qwen3-14B | DeepSeek-R1-0528-Qwen3-8B | deepseek-math-7b-instruct | gemma-2-9b-it | Qwen3-Embedding-4B | Ettin-Encoder-1b | LB Score       |
| ------ | --------- | ------------------------- | ------------------------- | ------------- | ------------------ | ---------------- | -------------- |
| Ratios | 1.0       | 0                         | 1.0                       | 1.2           | 0.8                | 0.8              | 0.947 (higher) |
| Ratios | 1.2       | 0                         | 1.2                       | 1.0           | 0.8                | 0.8              | 0.947          |
| Ratios | 1.2       | 0                         | 0.8                       | 1.0           | 0.8                | 0.8              | 0.947          |
| Ratios | 1.0       | 0                         | 0.8                       | 1.2           | 0.8                | 0.8              | 0.947          |
| Ratios | 1.2       | 0                         | 1.0                       | 1.2           | 0.8                | 0.8              | 0.947          |
| Ratios | 0.8       | 0                         | 1.0                       | 1.2           | 0.8                | 0.8              | 0.947          |
| Ratios | 1.2       | 0                         | 1.0                       | 1.2           | 1.0                | 0.8              | 0.947          |
