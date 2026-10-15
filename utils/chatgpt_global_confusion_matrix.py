import os
import json

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, mean_absolute_error


# ============================================================
# CONFIGURAZIONE MANUALE
# ============================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Inserisci qui manualmente le label Ground Truth
y_true = [1, 1, 1, 1, 1, 2, 1, 2, 0, 1, 2, 2, 0, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 0, 1,
          2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 1, 0, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 0, 1,
          2, 1, 1, 1, 2, 2, 1, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2]

# Inserisci qui manualmente le label predette
y_pred = [0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 2, 0, 0, 2, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 1, 0, 2, 0, 2, 0, 0, 0, 1, 0, 0, 0,
          1, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 1, 0, 2, 0, 1, 0, 0, 1, 1, 0, 2, 2, 1, 1, 1, 0, 2, 0, 1, 1, 1, 0, 0, 0,
          1, 1, 0, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 1, 0]

# Nome dell'esperimento/modello
EXPERIMENT_NAME = "ChatGPT"

# Cartella di output
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "manual_confusion_matrix"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

GLOBAL_CM_PATH = os.path.join(
    OUTPUT_DIR,
    "global_confusion_matrix.png"
)

NORMALIZED_CM_PATH = os.path.join(
    OUTPUT_DIR,
    "global_confusion_matrix_normalized.png"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "global_confusion_matrix.json"
)


# ============================================================
# CONTROLLI
# ============================================================

if not y_true:
    raise ValueError("y_true è vuoto. Inserisci manualmente le label Ground Truth.")

if not y_pred:
    raise ValueError("y_pred è vuoto. Inserisci manualmente le label predette.")

if len(y_true) != len(y_pred):
    raise ValueError(
        f"y_true e y_pred hanno dimensioni diverse: "
        f"{len(y_true)} vs {len(y_pred)}"
    )

valid_labels = {0, 1, 2}

if not all(value in valid_labels for value in y_true):
    raise ValueError("y_true deve contenere solo valori 0, 1, 2, 3, 4.")

if not all(value in valid_labels for value in y_pred):
    raise ValueError("y_pred deve contenere solo valori 0, 1, 2, 3, 4.")


# ============================================================
# METRICHE
# ============================================================

labels = [0, 1, 2]

accuracy = accuracy_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

cm_normalized = confusion_matrix(
    y_true,
    y_pred,
    labels=labels,
    normalize="true"
)


# ============================================================
# STAMPA RISULTATI
# ============================================================

print("=" * 70)
print("GLOBAL CONFUSION MATRIX")
print("=" * 70)

print(f"\nEsperimento: {EXPERIMENT_NAME}")
print(f"Numero campioni: {len(y_true)}")

print(f"\ny_true = {y_true}")
print(f"y_pred = {y_pred}")

print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Mean Absolute Error: {mae:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nConfusion Matrix normalizzata:")
print(cm_normalized)


# ============================================================
# SALVATAGGIO JSON
# ============================================================

metrics_data = {
    "experiment_name": EXPERIMENT_NAME,
    "number_of_samples": len(y_true),
    "y_true": y_true,
    "y_pred": y_pred,
    "accuracy": float(accuracy),
    "mean_absolute_error": float(mae),
    "labels": labels,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_normalized": cm_normalized.tolist()
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        metrics_data,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# CONFUSION MATRIX ASSOLUTA
# ============================================================

display_labels = [
    "0 - NONE",
    "1 - WARNING",
    "2 - DANGER",
]


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(cmap="Oranges", values_format="d")
plt.title(
    f"Global Danger Level Confusion Matrix - {EXPERIMENT_NAME}"
)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    GLOBAL_CM_PATH,
    dpi=300,
    bbox_inches="tight"
)

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
    f"Normalized Global Danger Level Confusion Matrix - {EXPERIMENT_NAME}"
)

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    NORMALIZED_CM_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# OUTPUT
# ============================================================

print("\nFile salvati:")

print(GLOBAL_CM_PATH)
print(NORMALIZED_CM_PATH)
print(METRICS_PATH)
