# map-kaggle-competition

### Changes

| Model | Architecture | LoRA Config | Learning Rate | CV Score | LB Score |
|-------|-------------|-------------|---------------|----------|----------|
| Qwen3-14B | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 0.945 | 0.943 |
| gemma-2-9b-it | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 0.941 | 0.939 |
| DeepSeek-R1-0528-Qwen3-8B | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 0.945 | 0.942 |
| deepseek-math-7b-instruct | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 0.943 | 0.942 |
| Qwen3-Embedding-4B | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 0.945 | 0.942 |
| AceReason-Nemotron-1.1-7B | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 0.943 | 0.932 |
| Llama-3.1-8B-Instruct | QLoRA-4bit | r=8, alpha=32 | 2e-4 | 0.943 | - |
