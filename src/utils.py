import torch
import numpy as np


def stringify_input(row):
    output = (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Student Explanation: {row['StudentExplanation']}\n\n"
    )
    if "is_mc_answer_correct" in row:
        output += f"Is the student's answer correct? {row['is_mc_answer_correct']}\n"
    if "is_student_explanation_correct" in row:
        output += f"Is the student's explanation correct? {row['is_student_explanation_correct']}\n"
    return output.strip()


def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = top3 == labels[:, None]

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}
