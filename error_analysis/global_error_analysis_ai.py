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
)

if not misclassification_files:
    raise FileNotFoundError(
        "Nessun file *_misclassifications.json trovato in:\n"
        f"{RESULTS_DIR}"
    )

all_errors = []

for filename in misclassification_files:
    path = os.path.join(RESULTS_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_errors.extend(
        data.get("misclassifications", [])
    )

grouped_errors = {}

for item in all_errors:

    key = (
        item["source_file"],
        int(item["ground_truth"]),
        int(item["predicted"])
    )

    if key not in grouped_errors:
        grouped_errors[key] = {
            "source_file": item["source_file"],
            "ground_truth": int(item["ground_truth"]),
            "ground_truth_label": item.get("ground_truth_label"),
            "predicted": int(item["predicted"]),
            "predicted_label": item.get("predicted_label"),
            "error_type": item.get("error_type"),
            "error_direction": item.get("error_direction"),
            "occurrences": 0,
            "sequence_ids": [],
            "segment_indices": [],
            "holmes_report": item.get("holmes_report"),
            "qwen_report": item.get("qwen_report"),
            "ground_truth_description": item.get(
                "ground_truth_description"
            )
        }

    grouped = grouped_errors[key]
    grouped["occurrences"] += 1
    grouped["sequence_ids"].append(
        str(item.get("sequence_id", ""))
    )
    grouped["segment_indices"].append(
        str(item.get("segment_index", ""))
    )

unique_errors = list(grouped_errors.values())

for item in unique_errors:
    item["sequence_ids"] = " | ".join(
        item["sequence_ids"]
    )
    item["segment_indices"] = " | ".join(
        item["segment_indices"]
    )

unique_errors.sort(
    key=lambda item: (
        -item["occurrences"],
        item["source_file"],
        item["ground_truth"],
        item["predicted"]
    )
)

CSV_FIELDS = [
    "source_file",
    "ground_truth",
    "ground_truth_label",
    "predicted",
    "predicted_label",
    "error_type",
    "error_direction",
    "occurrences",
    "sequence_ids",
    "segment_indices",
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
print("GLOBAL UNIQUE MISCLASSIFICATIONS - AI SEQUENCES")
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
