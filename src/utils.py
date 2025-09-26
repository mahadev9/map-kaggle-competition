import os
import re

import torch
import numpy as np
from typing import Dict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    SchedulerType,
    IntervalStrategy,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import SaveStrategy


def get_model_name(is_kaggle, root_path, base_model="") -> str:
    if is_kaggle:
        return os.path.join(root_path, base_model)
    return base_model


def convert_latex_to_text(text: str) -> str:
    # Convert LaTeX fractions to text
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", text)

    # Convert LaTeX multiplication
    text = re.sub(r"\\times", "x", text)

    # Convert LaTeX division
    text = re.sub(r"\\div", "/", text)

    # Handle LaTeX parentheses
    text = re.sub(r"\\left\(", "(", text)
    text = re.sub(r"\\right\)", ")", text)

    # Remove LaTeX math delimiters
    text = re.sub(r"\\(\s|\(|\))", r"\1", text)

    # Remove remaining backslashes for simple commands
    text = re.sub(r"\\", "", text)

    return text.strip()


def stringify_input(row, model_name, new_prompt=False) -> str:

    if new_prompt:
        correctness = "correct" if row.get("is_mc_answer_correct", False) else "incorrect"

        prompt = f"""Mathematical Problem Analysis:

Question: {row["QuestionText"]}
Student Answer: {row["MC_Answer"]} (This answer is {correctness})
Student Reasoning: {row["StudentExplanation"]}

Task: Identify the specific misconception or confirm correct understanding.

Common misconceptions in this area include:
- Arithmetic errors
- Conceptual misunderstandings
- Procedural mistakes
- Misapplication of rules

Student's misconception category:"""

        return convert_latex_to_text(prompt)

    output = [
        f"Question: {row['QuestionText']}",
        f"Answer: {row['MC_Answer']}",
    ]

    # ModernBERT/DeBERTaV3
    if "modernbert" in model_name.lower() or "deberta" in model_name.lower():
        if "is_mc_answer_correct" in row:
            correctness = "correct" if row["is_mc_answer_correct"] else "incorrect"
            x = f"This answer is {correctness}."
            output.append(x)

    # Ettin-Encoder/Gemma/Qwen
    if (
        "ettin" in model_name.lower()
        or "gemma" in model_name.lower()
        or "qwen" in model_name.lower()
        or "deepseek" in model_name.lower()
        or "acereason" in model_name.lower()
        or "acemath" in model_name.lower()
    ):
        if "is_mc_answer_correct" in row:
            correctness = "Yes" if row["is_mc_answer_correct"] else "No"
            x = f"Correct? {correctness}"
            output.append(x)

    output.append(f"Student's Explanation: {row['StudentExplanation']}")
    return convert_latex_to_text("\n".join(output))


def compute_map3(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if len(logits) == 2:
        logits = logits[0]
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = top3 == labels[:, None]

    weights = np.array([1.0, 0.5, 1 / 3])
    scores = np.sum(match * weights, axis=1)
    return {"map@3": scores.mean()}


def get_tokenizer(model_name: str):
    extra_kwargs = {}
    if "modernbert" in model_name.lower() or "ettin" in model_name.lower():
        extra_kwargs = {
            "reference_compile": False,
        }
    return AutoTokenizer.from_pretrained(model_name, **extra_kwargs)


def get_sequence_classifier(model_name: str, num_labels, q_lora_config={}):
    extra_kwargs = {}
    if "modernbert" in model_name.lower() or "ettin" in model_name.lower():
        extra_kwargs = {
            "reference_compile": False,
        }
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        **extra_kwargs,
        **q_lora_config,
    )


def get_training_arguments(
    learning_rate=5e-5,
    epochs=10,
    train_batch_size=8,
    eval_batch_size=16,
    bf16_support=True,
    train_on_full_dataset=False,
):
    extra_kwargs = {
        "fp16": True,  # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
        "do_eval": True,
        "eval_strategy": IntervalStrategy.STEPS,
        "load_best_model_at_end": True,
    }
    if bf16_support:
        # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU
        extra_kwargs.pop("fp16")
        extra_kwargs["bf16"] = True

    if train_on_full_dataset:
        extra_kwargs.pop("do_eval")
        extra_kwargs.pop("eval_strategy")
        extra_kwargs.pop("load_best_model_at_end")

    return TrainingArguments(
        output_dir="./output",
        do_train=True,
        save_strategy=SaveStrategy.STEPS,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        # weight_decay=0.02,
        # warmup_ratio=0.1,
        # lr_scheduler_type=SchedulerType.LINEAR,
        lr_scheduler_type=SchedulerType.COSINE_WITH_MIN_LR,
        lr_scheduler_kwargs={"min_lr": 1e-6},
        logging_dir="./logs",
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=5,
        label_names=["labels"],
        metric_for_best_model="map@3",
        greater_is_better=True,
        report_to="none",
        # gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        # use_mps_device=True,  # Use MPS for Apple Silicon
        **extra_kwargs,
    )


class CustomTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def get_trainer(
    model,
    tokenizer,
    training_args,
    train_ds,
    val_ds,
    compute_metrics=compute_map3,
    early_stopping_patience=5,
    train_on_full_dataset=False,
    class_weights=None,
):
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    extra_kwargs = {
        "eval_dataset": val_ds,
    }
    if train_on_full_dataset:
        extra_kwargs.pop("eval_dataset")

    if class_weights is None:
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=callbacks,
        )

    return CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=callbacks,
    )
