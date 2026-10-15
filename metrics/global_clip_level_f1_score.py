import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    classification_report
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Scegli:
# "large_video_dataset"
# "sequences_video_ai"
EVALUATION_TYPE = "sequences_video_ai"
# EVALUATION_TYPE = "large_video_dataset"

PROMPT_MODEL = "qwen"

VALID_DANGER_LEVELS = {0, 1, 2}
LABELS = [0, 1, 2]
CLASS_NAMES = [
    "NONE",
    "WARNING",
    "DANGER"
]

# Gestisce anche le copie temporanee create dai runner nella root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "metrics"
    else SCRIPT_DIR
)


# ============================================================
# PATH RISULTATI
# ============================================================

if EVALUATION_TYPE == "large_video_dataset":

    PROMPT_STAGE1_ID = 12
    PROMPT_STAGE2_ID = 13

    RESULTS_DIR = os.path.join(
        BASE_DIR,
        "large_video_evaluation",
        PROMPT_MODEL,
        f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
    )

    comparison_files = sorted([
        name
        for name in os.listdir(RESULTS_DIR)
        if name.endswith("_comparison.json")
    ])

    ITEM_NAME = "clip"

elif EVALUATION_TYPE == "sequences_video_ai":

    PROMPT_STAGE1_ID = 16
    PROMPT_STAGE2_ID = 11

    RESULTS_DIR = os.path.join(
        BASE_DIR,
        "composite_video_output",
        "virtual",
        "results",
        PROMPT_MODEL,
        f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
    )

    comparison_files = sorted([
        name
        for name in os.listdir(RESULTS_DIR)
        if name.startswith("composite_ai_video_")
        and name.endswith("_comparison.json")
    ])

    ITEM_NAME = "segment"

else:

    raise ValueError(
        "EVALUATION_TYPE deve essere "
        "'large_video_dataset' oppure "
        "'sequences_video_ai'."
    )


if not os.path.isdir(RESULTS_DIR):
    raise FileNotFoundError(
        f"Cartella risultati non trovata:\n"
        f"{RESULTS_DIR}"
    )


if not comparison_files:
    raise FileNotFoundError(
        f"Nessun comparison.json trovato in:\n"
        f"{RESULTS_DIR}"
    )


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "f1_metrics"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "global_clip_level_f1_scores.json"
)

PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "global_clip_level_f1_scores.png"
)


# ============================================================
# CARICAMENTO GT / PRED CLIP-LEVEL
# ============================================================

y_true_global = []
y_pred_global = []

tests = []


for filename in comparison_files:

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    current_true = data.get(
        "y_true",
        []
    )

    current_pred = data.get(
        "y_pred",
        []
    )

    if not current_true:
        raise ValueError(
            f"Nessun y_true disponibile in:\n"
            f"{filename}"
        )

    if len(current_true) != len(current_pred):
        raise ValueError(
            f"y_true e y_pred hanno dimensioni "
            f"diverse in:\n{filename}"
        )

    if not all(
        value in VALID_DANGER_LEVELS
        for value in current_true
    ):
        raise ValueError(
            f"y_true contiene label non valide "
            f"in {filename}: {current_true}"
        )

    if not all(
        value in VALID_DANGER_LEVELS
        for value in current_pred
    ):
        raise ValueError(
            f"y_pred contiene label non valide "
            f"in {filename}: {current_pred}"
        )

    test_id = str(
        data.get(
            "video",
            data.get(
                "video_id",
                filename
            )
        )
    )

    y_true_global.extend(
        current_true
    )

    y_pred_global.extend(
        current_pred
    )

    tests.append({
        "test_id": test_id,
        "comparison_file": filename,
        "number_of_items": len(
            current_true
        ),
        "y_true": current_true,
        "y_pred": current_pred
    })


# ============================================================
# F1 SCORE
# ============================================================

precision, recall, f1_per_class, support = (
    precision_recall_fscore_support(
        y_true_global,
        y_pred_global,
        labels=LABELS,
        zero_division=0
    )
)

macro_f1 = f1_score(
    y_true_global,
    y_pred_global,
    labels=LABELS,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true_global,
    y_pred_global,
    labels=LABELS,
    average="weighted",
    zero_division=0
)

micro_f1 = f1_score(
    y_true_global,
    y_pred_global,
    labels=LABELS,
    average="micro",
    zero_division=0
)

report = classification_report(
    y_true_global,
    y_pred_global,
    labels=LABELS,
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)


# ============================================================
# STAMPA RISULTATI
# ============================================================

print("=" * 70)
print("GLOBAL CLIP-LEVEL F1 SCORE")
print("=" * 70)

print(
    f"Evaluation type: "
    f"{EVALUATION_TYPE}"
)

print(
    f"Prompt model: "
    f"{PROMPT_MODEL}"
)

print(
    f"Prompts: "
    f"{PROMPT_STAGE1_ID} + "
    f"{PROMPT_STAGE2_ID}"
)

print(
    f"Test utilizzati: "
    f"{len(tests)}"
)

print(
    f"{ITEM_NAME.capitalize()} totali: "
    f"{len(y_true_global)}"
)

print(
    f"\ny_true globale = "
    f"{y_true_global}"
)

print(
    f"y_pred globale = "
    f"{y_pred_global}"
)

print(
    "\nF1 per classe:"
)

for (
    label,
    name,
    value,
    p,
    r,
    s
) in zip(
    LABELS,
    CLASS_NAMES,
    f1_per_class,
    precision,
    recall,
    support
):

    print(
        f"{label} - {name}: "
        f"F1={value:.4f}, "
        f"Precision={p:.4f}, "
        f"Recall={r:.4f}, "
        f"Support={int(s)}"
    )

print(
    f"\nMacro F1: "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1: "
    f"{weighted_f1:.4f}"
)

print(
    f"Micro F1: "
    f"{micro_f1:.4f}"
)


# ============================================================
# SALVATAGGIO JSON
# ============================================================

per_class = {}

for (
    label,
    name,
    p,
    r,
    f1_value,
    s
) in zip(
    LABELS,
    CLASS_NAMES,
    precision,
    recall,
    f1_per_class,
    support
):

    per_class[str(label)] = {
        "label": name,
        "precision": float(p),
        "recall": float(r),
        "f1_score": float(
            f1_value
        ),
        "support": int(s)
    }


metrics_data = {
    "evaluation_type":
        EVALUATION_TYPE,

    "prompt_model":
        PROMPT_MODEL,

    "prompt_stage1_id":
        PROMPT_STAGE1_ID,

    "prompt_stage2_id":
        PROMPT_STAGE2_ID,

    "number_of_tests":
        len(tests),

    "number_of_items":
        len(y_true_global),

    "item_type":
        ITEM_NAME,

    "labels":
        LABELS,

    "class_names":
        CLASS_NAMES,

    "y_true":
        y_true_global,

    "y_pred":
        y_pred_global,

    "per_class":
        per_class,

    "macro_f1":
        float(macro_f1),

    "weighted_f1":
        float(weighted_f1),

    "micro_f1":
        float(micro_f1),

    "classification_report":
        report,

    "tests":
        tests
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_data,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# GRAFICO F1 PER CLASSE
# ============================================================

plot_labels = [
    "0 - NONE",
    "1 - WARNING",
    "2 - DANGER"
]

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    plot_labels,
    f1_per_class
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    f"Global Clip-Level F1 Score per Class\n"
    f"{PROMPT_MODEL.upper()} "
    f"Prompt {PROMPT_STAGE1_ID}"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


print(
    f"\nMetriche F1 salvate in:\n"
    f"{METRICS_PATH}"
)

print(
    f"\nGrafico F1 salvato in:\n"
    f"{PLOT_PATH}"
)
