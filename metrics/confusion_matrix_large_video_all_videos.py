import os
import json

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


# ============================================================
# CONFIGURAZIONE VIDEO LUNGHI DATASET
# ============================================================
# Gestisce anche la copia temporanea creata dal runner nella root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "metrics"
    else SCRIPT_DIR
)

PROMPT_MODEL = "qwen"
PROMPT_STAGE1_ID = 12
PROMPT_STAGE2_ID = 13

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "large_video_evaluation",
    PROMPT_MODEL,
    f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
)

CM_DIR = os.path.join(
    RESULTS_DIR,
    "confusion_matrices"
)

os.makedirs(
    CM_DIR,
    exist_ok=True
)

VALID_DANGER_LEVELS = {0, 1, 2}

LABELS = [0, 1, 2]

DISPLAY_LABELS = [
    "0 - NONE",
    "1 - WARNING",
    "2 - DANGER"
]


# ============================================================
# CONFUSION MATRIX SINGOLO VIDEO
# ============================================================

def process_comparison_file(
    comparison_path
):
    """
    Carica un file *_comparison.json e genera
    la confusion matrix clip-level per quel video.
    """

    with open(
        comparison_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    video_name = data.get(
        "video",
        os.path.basename(
            comparison_path
        ).replace(
            "_comparison.json",
            ""
        )
    )

    video_base = os.path.splitext(
        os.path.basename(
            video_name
        )
    )[0]

    y_true = data.get(
        "y_true",
        []
    )

    y_pred = data.get(
        "y_pred",
        []
    )

    if not y_true:
        raise ValueError(
            "Nessun elemento disponibile "
            "per la confusion matrix."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true e y_pred hanno "
            "dimensioni diverse."
        )

    if not all(
        value in VALID_DANGER_LEVELS
        for value in y_true
    ):
        raise ValueError(
            f"y_true contiene label "
            f"non valide: {y_true}"
        )

    if not all(
        value in VALID_DANGER_LEVELS
        for value in y_pred
    ):
        raise ValueError(
            f"y_pred contiene label "
            f"non valide: {y_pred}"
        )

    # --------------------------------------------------------
    # RISULTATI
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "CONFUSION MATRIX"
    )
    print(
        "=" * 70
    )

    print(
        f"\nVideo: {video_name}"
    )

    print(
        f"\ny_true = {y_true}"
    )
    print(
        f"y_pred = {y_pred}"
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    print(
        "\nConfusion Matrix:"
    )
    print(
        cm
    )

    # --------------------------------------------------------
    # GRAFICO
    # --------------------------------------------------------

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=DISPLAY_LABELS
    )

    disp.plot(
        cmap="Oranges",
        values_format="d"
    )

    plt.title(
        "Clip-Level Danger Confusion Matrix - "
        f"{video_base} - "
        f"Qwen Prompt {PROMPT_STAGE1_ID}"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    output_path = os.path.join(
        CM_DIR,
        f"{video_base}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nMatrice salvata in:\n"
        f"{output_path}"
    )

    return {
        "video": video_name,
        "accuracy": accuracy,
        "output_path": output_path
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    comparison_files = sorted(
        os.path.join(
            RESULTS_DIR,
            filename
        )
        for filename
        in os.listdir(
            RESULTS_DIR
        )
        if filename.endswith(
            "_comparison.json"
        )
    )

    if not comparison_files:
        raise FileNotFoundError(
            "Nessun file *_comparison.json "
            f"trovato in:\n{RESULTS_DIR}"
        )

    print(
        "File comparison trovati:",
        len(comparison_files)
    )

    processed = 0
    errors = 0

    for comparison_path in comparison_files:

        try:

            process_comparison_file(
                comparison_path
            )

            processed += 1

        except Exception as error:

            errors += 1

            print(
                "\nERRORE su "
                f"{os.path.basename(comparison_path)}: "
                f"{error}"
            )

    print(
        "\n" + "=" * 70
    )
    print(
        "RIEPILOGO"
    )
    print(
        "=" * 70
    )
    print(
        "Confusion matrix create:",
        processed
    )
    print(
        "Errori:",
        errors
    )
