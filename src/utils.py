import os

import torch
import numpy as np
from typing import Dict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def get_model_name(is_kaggle, root_path) -> str:
    if is_kaggle:
        return os.path.join(
            root_path,
            "deberta/transformers/default/1/deberta-v3",
        )
    return "microsoft/deberta-v3-large"


def stringify_input(row) -> str:
    output = (
        f"Math Question: {row['QuestionText']}\n"
        f"Student's Answer: {row['MC_Answer']}\n"
        f"Student's Explanation: {row['StudentExplanation']}\n"
    )
    if "is_mc_answer_correct" in row:
        output += f"Answer Correctness: {'Correct' if row['is_mc_answer_correct'] else 'Incorrect'}\n"
    if "is_student_explanation_correct" in row:
        output += f"Explanation Correctness: {'Correct' if row['is_student_explanation_correct'] else 'Incorrect'}\n"

    # output += "\nTask: Identify the student's misconception category and specific misconception."
    return output.strip()


def compute_map3(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = top3 == labels[:, None]

    weights = np.array([1.0, 0.5, 1 / 3])
    scores = np.sum(match * weights, axis=1)
    return {"map@3": scores.mean()}


def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def get_sequence_classifier(model_name, num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def get_training_arguments(epochs=10, train_batch_size=8, eval_batch_size=16):
    return TrainingArguments(
        output_dir="./output",
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",  # no for no saving
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=5e-5,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        save_total_limit=1,
        label_names=['labels'],
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
        # use_mps_device=True,  # Use MPS for Apple Silicon
        bf16=True,  # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU
        # fp16=True, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
    )


def get_trainer(
    model,
    tokenizer,
    training_args,
    train_ds,
    val_ds,
    compute_metrics=compute_map3,
):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
