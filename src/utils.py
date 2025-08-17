import os

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
)
from transformers.trainer_utils import SaveStrategy


def get_model_name(is_kaggle, root_path, base_model="") -> str:
    if is_kaggle:
        return os.path.join(root_path, base_model)
    return base_model


def stringify_input(row) -> str:
    output = [
        f"Question: {row['QuestionText']}",
        f"Answer: {row['MC_Answer']}",
    ]

    # ModernBERT/DeBERTaV3
    if "is_mc_answer_correct" in row:
        correctness = "correct" if row["is_mc_answer_correct"] else "incorrect"
        x = f"This answer is {correctness}."
        output.append(x)

    # Ettin-Encoder
    # if "is_mc_answer_correct" in row:
    #     correctness = "Yes" if row["is_mc_answer_correct"] else "No"
    #     x = f"Correct? {correctness}"
    #     output.append(x)

    output.append(f"Student's Explanation: {row['StudentExplanation']}")
    # if "is_student_explanation_correct" in row:
    #     if row["is_student_explanation_correct"] == 0:
    #         x = "The student's explanation is neither correct nor a misconception"
    #     elif row["is_student_explanation_correct"] == 1:
    #         x = "The student's explanation is correct"
    #     elif row["is_student_explanation_correct"] == 2:
    #         x = "The student's explanation contains a misconception"
    #     output.append(x)

    return "\n".join(output)


def compute_map3(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = top3 == labels[:, None]

    weights = np.array([1.0, 0.5, 1 / 3])
    scores = np.sum(match * weights, axis=1)
    return {"map@3": scores.mean()}


def get_tokenizer(model_name):
    extra_kwargs = {}
    if "modernbert" in model_name.lower():
        extra_kwargs = {
            "reference_compile": False,
        }
    return AutoTokenizer.from_pretrained(model_name, **extra_kwargs)


def get_sequence_classifier(model_name, num_labels, q_lora_config={}):
    extra_kwargs = {}
    if "modernbert" in model_name.lower():
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
    epochs=10,
    train_batch_size=8,
    eval_batch_size=16,
    bf16_support=True,
):
    # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
    extra_kwargs = {"fp16": True}
    if bf16_support:
        # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU
        extra_kwargs = {"bf16": True}
    return TrainingArguments(
        output_dir="./output",
        do_train=True,
        do_eval=True,
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=SaveStrategy.EPOCH,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=4e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type=SchedulerType.LINEAR,
        # lr_scheduler_type=SchedulerType.COSINE_WITH_MIN_LR,
        # lr_scheduler_kwargs={"min_lr": 1e-6},
        logging_dir="./logs",
        logging_steps=100,
        save_steps=200,
        eval_steps=200,
        save_total_limit=1,
        label_names=["labels"],
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
        # gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        # use_mps_device=True,  # Use MPS for Apple Silicon
        **extra_kwargs,
    )


def get_trainer(
    model,
    tokenizer,
    training_args,
    train_ds,
    val_ds,
    compute_metrics=compute_map3,
    early_stopping_patience=5,
):
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=callbacks,
    )
