import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    mean_absolute_error
)

# Gestisce anche le copie temporanee create dai runner nella root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "metrics"
    else SCRIPT_DIR
)

PROMPT_MODEL = "qwen"
PROMPT_STAGE1_ID = 16
PROMPT_STAGE2_ID = 11
EVALUATION_TYPE = "sequences_video_ai"

VALID_DANGER_LEVELS = {0, 1, 2}


RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "composite_video_output",
    "virtual",
    "results",
    PROMPT_MODEL,
    f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
)

OUTPUT_PATH = os.path.join(RESULTS_DIR, "global_confusion_matrix.png")
NORMALIZED_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "global_confusion_matrix_normalized.png"
)
METRICS_PATH = os.path.join(RESULTS_DIR, "global_confusion_matrix.json")

if not os.path.isdir(RESULTS_DIR):
    raise FileNotFoundError(f"Cartella risultati non trovata:\n{RESULTS_DIR}")

comparison_files = sorted([
    name for name in os.listdir(RESULTS_DIR)
    if name.startswith("composite_ai_video_")
    and name.endswith("_comparison.json")
])

if not comparison_files:
    raise FileNotFoundError(f"Nessun comparison.json trovato in:\n{RESULTS_DIR}")


'''
# ============================================================
# CONFIGURAZIONE VIDEO LUNGHI DATASET
# ============================================================
PROMPT_MODEL = "qwen"
PROMPT_STAGE1_ID = 12
PROMPT_STAGE2_ID = 13
EVALUATION_TYPE = "large_video_dataset"

VALID_DANGER_LEVELS = {0, 1, 2}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "large_video_evaluation",
    PROMPT_MODEL,
    f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
)


CM_DIR = os.path.join(
    RESULTS_DIR,
    "metrics"
)

os.makedirs(CM_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    CM_DIR,
    "global_confusion_matrix.png"
)

NORMALIZED_OUTPUT_PATH = os.path.join(
    CM_DIR,
    "global_confusion_matrix_normalized.png"
)

METRICS_PATH = os.path.join(
    CM_DIR,
    "global_confusion_matrix.json"
)

if not os.path.isdir(RESULTS_DIR):
    raise FileNotFoundError(f"Cartella risultati non trovata:\n{RESULTS_DIR}")

comparison_files = sorted([
    name for name in os.listdir(RESULTS_DIR)
    if name.endswith("_comparison.json")
])

if not comparison_files:
    raise FileNotFoundError(f"Nessun comparison.json trovato in:\n{RESULTS_DIR}")

# ============================================================
# FINE CONFIGURAZIONE VIDEO LUNGHI DATASET
# ============================================================
'''

y_true_global = []
y_pred_global = []
test_ids = []
test_details = []

for filename in comparison_files:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    current_true = data.get("y_true", [])
    current_pred = data.get("y_pred", [])

    if not all(value in VALID_DANGER_LEVELS for value in current_true):
        raise ValueError(
            f"y_true contiene label non valide in {filename}: {current_true}"
        )

    if not all(value in VALID_DANGER_LEVELS for value in current_pred):
        raise ValueError(
            f"y_pred contiene label non valide in {filename}: {current_pred}"
        )

    if not current_true:
        raise ValueError(f"Nessun y_true disponibile in:\n{filename}")
    if len(current_true) != len(current_pred):
        raise ValueError(f"y_true e y_pred hanno dimensioni diverse in:\n{filename}")

    # test_id = str(data.get("video_id", filename))
    # test_id = str(data.get("video", filename))

    test_id = str(
        data.get(
            "video",
            data.get("video_id", filename)
        )
    )

    test_ids.append(test_id)
    y_true_global.extend(current_true)
    y_pred_global.extend(current_pred)
    test_details.append({
        "test_id": test_id,
        "comparison_file": filename,
        "number_of_segments": len(current_true),
        "y_true": current_true,
        "y_pred": current_pred
    })

accuracy = accuracy_score(y_true_global, y_pred_global)
mae = mean_absolute_error(y_true_global, y_pred_global)
labels = [0, 1, 2]

cm = confusion_matrix(y_true_global, y_pred_global, labels=labels)
cm_normalized = confusion_matrix(
    y_true_global,
    y_pred_global,
    labels=labels,
    normalize="true"
)
print("=" * 70)
print("GLOBAL CONFUSION MATRIX")
print("=" * 70)
print(f"\nPrompt model: {PROMPT_MODEL}")
print(
    f"Prompts: "
    f"{PROMPT_STAGE1_ID} + {PROMPT_STAGE2_ID}"
)
print(f"Test utilizzati: {len(test_ids)}")
print(f"Segmenti totali: {len(y_true_global)}")
print(f"Test IDs: {test_ids}")
print(f"\ny_true globale = {y_true_global}")
print(f"y_pred globale = {y_pred_global}")
print(f"\nAccuracy globale: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Mean Absolute Error globale: {mae:.4f}")
print("\nConfusion Matrix globale:")
print(cm)

metrics_data = {
    "prompt_model": PROMPT_MODEL,
    "prompt_stage1_id": PROMPT_STAGE1_ID,
    "prompt_stage2_id": PROMPT_STAGE2_ID,
    "evaluation_type": EVALUATION_TYPE,
    "number_of_tests": len(test_ids),
    "number_of_segments": len(y_true_global),
    "test_ids": test_ids,
    "comparison_files": comparison_files,
    "y_true": y_true_global,
    "y_pred": y_pred_global,
    "accuracy": float(accuracy),
    "mean_absolute_error": float(mae),
    "labels": labels,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_normalized": cm_normalized.tolist(),
    "tests": test_details
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4, ensure_ascii=False)

display_labels = [
    "0 - NONE",
    "1 - WARNING",
    "2 - DANGER"
]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap="Oranges", values_format="d")
plt.title(
    f"Global Clip-Level Danger Confusion Matrix - "
    f"{PROMPT_MODEL.upper()} Prompt "
    f"{PROMPT_STAGE1_ID}"
)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# CONFUSION MATRIX NORMALIZZATA
# ============================================================

disp_normalized = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=display_labels
)

disp_normalized.plot(
    cmap="Oranges",
    values_format=".2f"
)

plt.title(
    f"Normalized Global Clip-Level Danger Confusion Matrix\n"
    f"{PROMPT_MODEL.upper()} Prompt {PROMPT_STAGE1_ID}"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    NORMALIZED_OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"\nMatrice globale normalizzata salvata in:\n"
    f"{NORMALIZED_OUTPUT_PATH}"
)

print(f"\nMatrice globale salvata in:\n{OUTPUT_PATH}")
print(f"\nDati globali salvati in:\n{METRICS_PATH}")
plt.show()
plt.close()
