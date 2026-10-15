import os
import json

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


# ============================================================
# CONFIGURAZIONE VIDEO SEQUENZE AI
# ============================================================
# Gestisce anche le copie temporanee create dai runner nella root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "metrics"
    else SCRIPT_DIR
)

VIDEO_ID = "01"

PROMPT_MODEL = "qwen"
PROMPT_STAGE1_ID = 16
PROMPT_STAGE2_ID = 11

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "composite_video_output",
    "virtual",
    "results",
    PROMPT_MODEL,
    f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
)


COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_comparison.json"
)


OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_confusion_matrix.png"
)


'''
# ============================================================
# CONFIGURAZIONE VIDEO LUNGHI DATASET
# ============================================================
PROMPT_MODEL = "qwen"
PROMPT_STAGE1_ID = 12
PROMPT_STAGE2_ID = 13

RESULTS_DIR = os.path.join(
    "./large_video_evaluation",
    PROMPT_MODEL,
    f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
)
VIDEO_ID = "006"

VIDEO_NAME = "Robbery"+VIDEO_ID+"_x264"

COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    f"{VIDEO_NAME}_comparison.json"
)

CM_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

os.makedirs(CM_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    CM_DIR,
    f"{VIDEO_NAME}_confusion_matrix.png"
)

'''
# ============================================================
# CARICAMENTO
# ============================================================

if not os.path.exists(COMPARISON_PATH):
    raise FileNotFoundError(
        f"File non trovato:\n{COMPARISON_PATH}"
    )


with open(
    COMPARISON_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


y_true = data["y_true"]
y_pred = data["y_pred"]

VALID_DANGER_LEVELS = {0, 1, 2}

if not all(value in VALID_DANGER_LEVELS for value in y_true):
    raise ValueError(
        f"y_true contiene label non valide: {y_true}"
    )

if not all(value in VALID_DANGER_LEVELS for value in y_pred):
    raise ValueError(
        f"y_pred contiene label non valide: {y_pred}"
    )


# ============================================================
# CONTROLLO
# ============================================================

if not y_true:
    raise ValueError(
        "Nessun elemento disponibile per la confusion matrix."
    )


if len(y_true) != len(y_pred):
    raise ValueError(
        "y_true e y_pred hanno dimensioni diverse."
    )


# ============================================================
# RISULTATI
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(f"\nVideo: {VIDEO_ID}")

print(f"\ny_true = {y_true}")
print(f"y_pred = {y_pred}")


accuracy = accuracy_score(
    y_true,
    y_pred
)

print(
    f"\nAccuracy: {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [0, 1, 2]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# GRAFICO
# ============================================================

display_labels = [
    "0 - NONE",
    "1 - WARNING",
    "2 - DANGER"
]


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(
    cmap="Oranges",
    values_format="d"
)

plt.title(
    f"Clip-Level Danger Confusion Matrix - "
    f"Sequence {VIDEO_ID} - Qwen Prompt {PROMPT_STAGE1_ID}"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()


# ============================================================
# SALVATAGGIO
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"\nMatrice salvata in:\n"
    f"{OUTPUT_PATH}"
)

plt.show()