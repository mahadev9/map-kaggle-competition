import os
import sys
import shutil
import pandas as pd
import numpy as np
import joblib
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import torch
from datasets import Dataset
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)

from utils import (
    stringify_input,
    get_model_name,
    get_sequence_classifier,
    get_tokenizer,
    get_training_arguments,
    get_trainer,
)

# ...existing setup code...
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ROOT_PATH = os.getcwd()
if "/kaggle" in ROOT_PATH:
    ROOT_PATH = "/kaggle/input"
    sys.path.append(os.path.join(ROOT_PATH, "map-utilities"))

# Configuration
# BASE_MODEL = "google/gemma-2-9b-it"
BASE_MODEL = "jhu-clsp/ettin-encoder-1b"

N_FOLDS = 5
RANDOM_STATE = 42
MAX_LEN = 256
EPOCHS = 3
LEARNING_RATE = 4e-5
MODEL_NAME = get_model_name("/kaggle" in ROOT_PATH, ROOT_PATH, BASE_MODEL)

USE_LORA = False
USE_QLORA = False
BITS = 4
USE_4BIT = BITS == 4
USE_8BIT = BITS == 8

TRAIN_PATH = os.path.join(
    ROOT_PATH, "map-charting-student-math-misunderstandings", "train.csv"
)
TEST_PATH = os.path.join(
    ROOT_PATH, "map-charting-student-math-misunderstandings", "test.csv"
)

# Load and prepare data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# ...existing data preparation code...
train_df.Misconception = train_df.Misconception.fillna("NA")
train_df["predict"] = train_df.Category + ":" + train_df.Misconception

# Add correct answer information
idx = train_df.Category.str.contains("True", case=False)
tmp = train_df.loc[idx].copy()
tmp["c"] = tmp.groupby(["QuestionId", "MC_Answer"]).MC_Answer.transform("count")
tmp = tmp.sort_values("c", ascending=False)
tmp = tmp.drop_duplicates(["QuestionId"])
tmp = tmp[["QuestionId", "MC_Answer"]]
tmp["is_mc_answer_correct"] = True

train_df = train_df.merge(tmp, on=["QuestionId", "MC_Answer"], how="left")
train_df.is_mc_answer_correct = train_df.is_mc_answer_correct.fillna(False)

test_df = test_df.merge(tmp, on=["QuestionId", "MC_Answer"], how="left")
test_df.is_mc_answer_correct = test_df.is_mc_answer_correct.fillna(False)

# Load label encoder
le = joblib.load(os.path.join(ROOT_PATH, "label_encoder.joblib"))
train_df["label"] = le.transform(train_df["predict"])
n_classes = len(le.classes_)

print(f"Train shape: {train_df.shape} with {n_classes} classes")
print(f"Using {N_FOLDS}-fold cross-validation")


def setup_model_config():
    """Setup model configuration for each fold"""
    # LoRA configuration
    lora_config = None
    if USE_LORA:
        R = 8
        lora_config = LoraConfig(
            r=R,
            lora_alpha=R * 4,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "down_proj",
                "up_proj",
                "gate_proj",
            ],
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
        )

    # Quantization configuration
    q_lora_config = {"torch_dtype": torch.bfloat16}
    if USE_QLORA:
        from transformers import BitsAndBytesConfig

        kwargs = {}
        if USE_4BIT:
            kwargs = {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_storage": torch.bfloat16,
            }
        if USE_8BIT:
            kwargs = {"load_in_8bit": True}

        bnb_config = BitsAndBytesConfig(**kwargs)
        q_lora_config["quantization_config"] = bnb_config

    return lora_config, q_lora_config


def train_single_fold(fold_idx, train_idx, val_idx, train_df):
    """Train a single fold"""
    print(f"\n{'=' * 60}")
    print(f"Training Fold {fold_idx + 1}/{N_FOLDS}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print(f"{'=' * 60}")

    # Create fold datasets
    fold_train_df = train_df.iloc[train_idx].copy()
    fold_val_df = train_df.iloc[val_idx].copy()

    # Prepare string inputs
    fold_train_df["stringified_input"] = fold_train_df.apply(
        lambda row: stringify_input(row, MODEL_NAME), axis=1
    )
    fold_val_df["stringified_input"] = fold_val_df.apply(
        lambda row: stringify_input(row, MODEL_NAME), axis=1
    )

    # Create HF datasets
    train_ds = Dataset.from_pandas(fold_train_df[["stringified_input", "label"]])
    val_ds = Dataset.from_pandas(fold_val_df[["stringified_input", "label"]])

    # Setup model
    lora_config, q_lora_config = setup_model_config()
    seq_model = get_sequence_classifier(MODEL_NAME, n_classes, q_lora_config)
    tokenizer = get_tokenizer(MODEL_NAME)

    # Handle padding token
    if (
        "gemma" in MODEL_NAME.lower()
        or "qwen" in MODEL_NAME.lower()
        or "deepseek-math" in MODEL_NAME.lower()
        or "llama-3.1" in MODEL_NAME.lower()
        or "acemath" in MODEL_NAME.lower()
    ):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        seq_model.config.pad_token_id = tokenizer.pad_token_id

    # Apply PEFT
    if USE_QLORA:
        seq_model = prepare_model_for_kbit_training(seq_model)

    if USE_LORA:
        seq_model = get_peft_model(seq_model, lora_config)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples["stringified_input"])

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)

    # Training arguments
    training_args = get_training_arguments(
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        train_batch_size=16,
        eval_batch_size=32,
        bf16_support="/kaggle" not in ROOT_PATH,
        train_on_full_dataset=False,
    )

    # Create trainer
    trainer = get_trainer(
        seq_model,
        tokenizer,
        training_args,
        train_ds,
        val_ds,
        train_on_full_dataset=False,
    )

    # Train
    trainer.train()

    # Save fold model
    fold_model_path = f"oof_models/fold_{fold_idx}"
    complete_dir = os.path.join(ROOT_PATH, fold_model_path)
    if os.path.exists(complete_dir):
        shutil.rmtree(complete_dir)
    os.makedirs(complete_dir, exist_ok=True)
    trainer.save_model(fold_model_path)
    tokenizer.save_pretrained(fold_model_path)

    # Generate OOF predictions
    val_predictions = trainer.predict(val_ds)
    val_probs = torch.nn.functional.softmax(
        torch.tensor(val_predictions.predictions), dim=1
    ).numpy()

    # Calculate fold score
    val_labels = fold_val_df["label"].values
    fold_score = calculate_map3(val_probs, val_labels)

    print(f"Fold {fold_idx + 1} MAP@3: {fold_score:.5f}")

    # Clean up memory
    del seq_model, trainer, train_ds, val_ds
    torch.cuda.empty_cache()

    return val_probs, val_idx, fold_score


def calculate_map3(predictions, labels):
    """Calculate MAP@3 score"""
    top3 = np.argsort(-predictions, axis=1)[:, :3]
    match = top3 == labels[:, None]
    weights = np.array([1.0, 0.5, 1 / 3])
    scores = np.sum(match * weights, axis=1)
    return scores.mean()


def run_oof_training():
    """Main OOF training loop"""
    # Setup stratified K-fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Initialize OOF predictions
    oof_predictions = np.zeros((len(train_df), n_classes))
    oof_scores = []

    # Train each fold
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["label"])
    ):
        val_probs, val_indices, fold_score = train_single_fold(
            fold_idx, train_idx, val_idx, train_df
        )

        # Store OOF predictions
        oof_predictions[val_indices] = val_probs
        oof_scores.append(fold_score)

    # Calculate overall OOF score
    overall_score = calculate_map3(oof_predictions, train_df["label"].values)

    print(f"\n{'=' * 60}")
    print("OOF TRAINING COMPLETED")
    print(f"{'=' * 60}")
    print(f"Individual Fold Scores: {[f'{score:.5f}' for score in oof_scores]}")
    print(f"Mean Fold Score: {np.mean(oof_scores):.5f} ± {np.std(oof_scores):.5f}")
    print(f"Overall OOF Score: {overall_score:.5f}")

    # Save OOF predictions
    oof_df = pd.DataFrame(
        oof_predictions, columns=[f"pred_{i}" for i in range(n_classes)]
    )
    oof_df["true_label"] = train_df["label"].values
    oof_df["predict"] = train_df["predict"].values
    oof_df["fold_score"] = 0

    # Add fold information
    fold_info = np.zeros(len(train_df))
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        fold_info[val_idx] = fold_idx
    oof_df["fold"] = fold_info

    oof_df.to_csv("oof_predictions.csv", index=False)

    return oof_predictions, oof_scores


def generate_test_predictions():
    """Generate test predictions using all fold models"""
    print(f"\n{'=' * 60}")
    print("GENERATING TEST PREDICTIONS")
    print(f"{'=' * 60}")

    # Prepare test data
    test_df["stringified_input"] = test_df.apply(
        lambda row: stringify_input(row, MODEL_NAME), axis=1
    )

    all_test_predictions = []

    for fold_idx in range(N_FOLDS):
        print(f"Loading fold {fold_idx + 1} model...")

        # Load tokenizer
        fold_model_path = f"oof_models/fold_{fold_idx}"
        tokenizer = get_tokenizer(fold_model_path)

        # Prepare test dataset
        test_ds = Dataset.from_pandas(test_df[["stringified_input"]])

        def tokenize_function(examples):
            return tokenizer(examples["stringified_input"])

        test_ds = test_ds.map(tokenize_function, batched=True)

        # Load model and generate predictions
        lora_config, q_lora_config = setup_model_config()
        seq_model = get_sequence_classifier(MODEL_NAME, n_classes, q_lora_config)

        if USE_LORA:
            from peft import PeftModel

            seq_model = PeftModel.from_pretrained(seq_model, fold_model_path)

        # Create trainer for inference
        training_args = get_training_arguments(
            bf16_support="/kaggle" not in ROOT_PATH,
            train_on_full_dataset=True,  # No validation needed for inference
        )
        trainer = get_trainer(seq_model, tokenizer, training_args, test_ds, test_ds)

        # Generate predictions
        predictions = trainer.predict(test_ds)
        probs = torch.nn.functional.softmax(
            torch.tensor(predictions.predictions), dim=1
        ).numpy()

        all_test_predictions.append(probs)

        # Clean up
        del seq_model, trainer, test_ds
        torch.cuda.empty_cache()

    # Ensemble predictions (simple average)
    ensemble_predictions = np.mean(all_test_predictions, axis=0)

    # Generate submission
    top3 = np.argsort(-ensemble_predictions, axis=1)[:, :3]
    flat_top3 = top3.flatten()
    decoded_labels = le.inverse_transform(flat_top3)
    top3_labels = decoded_labels.reshape(top3.shape)

    joined_preds = [" ".join(row) for row in top3_labels]

    submission = pd.DataFrame(
        {"row_id": test_df.row_id.values, "Category:Misconception": joined_preds}
    )
    submission.to_csv("oof_submission.csv", index=False)

    print("Test predictions saved to 'oof_submission.csv'")
    return ensemble_predictions, submission


# Run the complete OOF pipeline
if __name__ == "__main__":
    print("Starting OOF Training...")
    oof_predictions, oof_scores = run_oof_training()

    print("Generating test predictions...")
    test_predictions, submission = generate_test_predictions()

    print("\nOOF Training and Prediction Complete!")
    print(f"Final submission shape: {submission.shape}")
