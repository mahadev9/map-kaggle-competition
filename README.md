# map-kaggle-competition

### Changes

| Model | Architecture | LoRA Config | Learning Rate | Batch Size | CV Score | LB Score |
|-------|--------------|------------|---------------|------------|----------|----------|
| Qwen3-14B | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 16 | 0.945 | 0.943 |
| gemma-2-9b-it | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 16 | 0.942 | 0.942 |
| DeepSeek-R1-0528-Qwen3-8B | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 16 | 0.945 | 0.942 |
| deepseek-math-7b-instruct | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 16 | 0.943 | 0.942 |
| Qwen3-Embedding-4B | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 16 | 0.945 | 0.942 |
| AceReason-Nemotron-1.1-7B | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 16 | 0.943 | 0.938 |
| Llama-3.1-8B-Instruct | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 16 | 0.943 | 0.939 |
| Ettin-Encoder-1b | Fine-Tune | - | 4e-5 | 16 | 0.943 | 0.938 |

### Ensemble Submission

| Models | Top@k | LB Score |
|--------|-------|----------|
| Qwen3-14B, gemma-2-9b-it, DeepSeek-R1-0528-Qwen3-8B, deepseek-math-7b-instruct, Qwen3-Embedding-4B | 3 | 0.946 |
| Qwen3-14B, gemma-2-9b-it, DeepSeek-R1-0528-Qwen3-8B, deepseek-math-7b-instruct, Qwen3-Embedding-4B | 10 | 0.946 |
| Qwen3-14B, gemma-2-9b-it, DeepSeek-R1-0528-Qwen3-8B, deepseek-math-7b-instruct | 3 | 0.944 |
| Qwen3-14B, gemma-2-9b-it, deepseek-math-7b-instruct, Qwen3-Embedding-4B | 3 | 0.945 |
| Qwen3-14B, gemma-2-9b-it, DeepSeek-R1-0528-Qwen3-8B, Qwen3-Embedding-4B | 3 | 0.945 |
| Qwen3-14B, DeepSeek-R1-0528-Qwen3-8B, deepseek-math-7b-instruct, Qwen3-Embedding-4B | 3 | 0.945 |
