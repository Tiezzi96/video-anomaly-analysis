import os
import json
import csv

# Gestisce anche le copie temporanee create dai runner nella root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "error_analysis"
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

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "global_unique_misclassifications.csv"
)

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "global_unique_misclassifications.json"
)

if not os.path.isdir(RESULTS_DIR):
    raise FileNotFoundError(
        f"Cartella risultati non trovata:\n{RESULTS_DIR}"
    )

misclassification_files = sorted(
    filename
    for filename in os.listdir(RESULTS_DIR)
    if filename.endswith("_misclassifications.json")
    and filename != os.path.basename(OUTPUT_JSON)
    and os.path.isfile(os.path.join(RESULTS_DIR, filename))
)

if not misclassification_files:
    raise FileNotFoundError(
        "Nessun file *_misclassifications.json trovato in:\n"
        f"{RESULTS_DIR}"
    )

all_errors = []

for filename in misclassification_files:
    path = os.path.join(RESULTS_DIR, filename)

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    errors = (
        data.get("misclassifications")
        if isinstance(data, dict)
        else data
    )

    if not isinstance(errors, list):
        raise ValueError(
            f"Formato non valido in {path}: attesa una lista "
            "o un oggetto con la lista 'misclassifications'."
        )

    for index, item in enumerate(errors):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: errore {index} non valido.")

        if not isinstance(item.get("video"), str):
            raise ValueError(f"{path}: errore {index}, video non valido.")

        for field in ("clip_index", "ground_truth", "predicted"):
            if type(item.get(field)) is not int:
                raise ValueError(
                    f"{path}: errore {index}, '{field}' deve essere un intero."
                )

    all_errors.extend(errors)


grouped_errors = {}

for item in all_errors:

    key = (
        item["video"],
        item["clip_index"],
        item["ground_truth"],
        item["predicted"]
    )

    if key not in grouped_errors:
        grouped_errors[key] = {
            "video": item["video"],
            "clip_index": item["clip_index"],
            "start": item.get("start"),
            "end": item.get("end"),
            "ground_truth": item["ground_truth"],
            "ground_truth_label": item.get("ground_truth_label"),
            "predicted": item["predicted"],
            "predicted_label": item.get("predicted_label"),
            "error_type": item.get("error_type"),
            "error_direction": item.get("error_direction"),
            "occurrences": 0,
            "holmes_report": item.get("holmes_report"),
            "qwen_report": item.get("qwen_report"),
            "ground_truth_description": item.get(
                "ground_truth_description"
            )
        }

    grouped_errors[key]["occurrences"] += 1

unique_errors = list(grouped_errors.values())


unique_errors.sort(
    key=lambda item: (
        -item["occurrences"],
        item["video"],
        item["clip_index"],
        item["ground_truth"],
        item["predicted"]
    )
)

CSV_FIELDS = [
    "video",
    "clip_index",
    "start",
    "end",
    "ground_truth",
    "ground_truth_label",
    "predicted",
    "predicted_label",
    "error_type",
    "error_direction",
    "occurrences",
    "holmes_report",
    "qwen_report",
    "ground_truth_description"
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=CSV_FIELDS,
        extrasaction="ignore"
    )
    writer.writeheader()
    for item in unique_errors:
        writer.writerow(item)

output_data = {
    "prompt_model": PROMPT_MODEL,
    "prompt_stage1_id": PROMPT_STAGE1_ID,
    "prompt_stage2_id": PROMPT_STAGE2_ID,
    "number_of_source_files": len(
        misclassification_files
    ),
    "number_of_error_occurrences": len(
        all_errors
    ),
    "number_of_unique_errors": len(
        unique_errors
    ),
    "unique_errors": unique_errors
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output_data,
        f,
        indent=4,
        ensure_ascii=False
    )

print("=" * 80)
print("GLOBAL UNIQUE MISCLASSIFICATIONS - LARGE VIDEO")
print("=" * 80)
print(
    "File misclassifications letti:",
    len(misclassification_files)
)
print(
    "Occorrenze totali di errore:",
    len(all_errors)
)
print(
    "Errori unici:",
    len(unique_errors)
)
print(
    f"\nCSV globale salvato in:\n{OUTPUT_CSV}"
)
print(
    f"\nJSON globale salvato in:\n{OUTPUT_JSON}"
)
