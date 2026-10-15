import os
import sys

# Gestisce anche le copie temporanee create dai runner nella root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "evaluate"
    else SCRIPT_DIR
)

# Rende accessibile ground_truth nella cartella principale.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
import re
import csv

from utils.ground_truth import (
    GROUND_TRUTH,
    GROUND_TRUTH_DESCRIPTIONS,
    DANGER_LABELS,
)

# Tipo di composite da valutare:
# "physical" = video composite realmente creato
# "virtual"  = sequenza logica di video AI
# ============================================================
# CONFIGURAZIONE ESPERIMENTO
# ============================================================

VIDEO_ID = ""

COMPOSITE_TYPE = "virtual"

SECOND_MODEL = "qwen3.5:9b"
PROMPT_MODEL = "qwen"

PIPELINE_MODE = "two_stage"
# "two_stage"
# "single_clip_stage1"

if PIPELINE_MODE == "two_stage":
    PROMPT_STAGE1_ID = 10
    PROMPT_STAGE2_ID = 11

elif PIPELINE_MODE == "single_clip_stage1":
    PROMPT_STAGE1_ID = 17
    PROMPT_STAGE2_ID = 11

else:
    raise ValueError(
        "PIPELINE_MODE non valido."
    )

BASE_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "composite_video_output"
)

VALID_DANGER_LEVELS = {0, 1, 2}


# ============================================================
# PATH
# ============================================================


if COMPOSITE_TYPE == "virtual":

    COMPOSITE_OUTPUT_DIR = os.path.join(
        BASE_OUTPUT_DIR,
        "virtual"
    )

elif COMPOSITE_TYPE == "physical":

    COMPOSITE_OUTPUT_DIR = BASE_OUTPUT_DIR

else:

    raise ValueError(
        "COMPOSITE_TYPE deve essere 'physical' oppure 'virtual'."
    )


MANIFEST_PATH = os.path.join(
    COMPOSITE_OUTPUT_DIR,
    f"composite_ai_video_{VIDEO_ID}_manifest.json"
)


RESULTS_DIR = os.path.join(
    COMPOSITE_OUTPUT_DIR,
    "results",
    PROMPT_MODEL,
    f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
)


EVALUATION_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_evaluation.json"
)


OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_comparison.json"
)

MISCLASSIFICATIONS_JSON_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_misclassifications.json"
)

MISCLASSIFICATIONS_TXT_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_misclassifications.txt"
)

MISCLASSIFICATIONS_CSV_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_misclassifications.csv"
)

# ============================================================
# FUNZIONI
# ============================================================

def load_json(path):
    """
    Load a JSON file.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File non trovato:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_clip_evaluations(clip_evaluations_text):
    """
    Estrae le valutazioni clip-level prodotte dallo Stage 1.

    Restituisce:
        {
            clip_index: {
                "start_time": ...,
                "end_time": ...,
                "assessment": ...,
                "danger_level": ...,
                "anomaly_type": ...,
                "evidence": ...
            }
        }
    """

    if not clip_evaluations_text:
        return {}

    clip_header_pattern = re.compile(
        r"(?mi)^Clip\s+(\d+)\s*$"
    )

    matches = list(
        clip_header_pattern.finditer(
            clip_evaluations_text
        )
    )

    parsed = {}

    for index, match in enumerate(matches):

        clip_index = int(
            match.group(1)
        )

        block_start = match.end()

        if index + 1 < len(matches):
            block_end = matches[index + 1].start()
        else:
            block_end = len(clip_evaluations_text)

        block = clip_evaluations_text[
            block_start:block_end
        ]

        start_match = re.search(
            r"(?mi)^Start:\s*([\d.]+)",
            block
        )

        end_match = re.search(
            r"(?mi)^End:\s*([\d.]+)",
            block
        )

        assessment_match = re.search(
            r"(?mi)^\s*Assessment:\s*"
            r"(NORMAL|ANOMALOUS|NONE|WARNING|DANGER)\s*$",
            block
        )

        danger_match = re.search(
            r"(?mi)^\s*(?:[-*]\s*)?"
            r"\**Danger Level\**\s*:\s*"
            r"\**([0-2])\s*-\s*(NONE|WARNING|DANGER)\**\b.*$",
            block
        )

        anomaly_type_match = re.search(
            r"(?mi)^Anomaly Type:\s*(.*)$",
            block
        )

        evidence_match = re.search(
            r"(?mi)^Evidence:\s*(.*)$",
            block
        )

        if assessment_match is None:
            print("\n" + "=" * 80)
            print(f"DEBUG - BLOCCO CLIP {clip_index}")
            print("=" * 80)
            print(block)
            print("=" * 80)

            raise ValueError(
                f"Assessment mancante o non valido "
                f"nella Clip {clip_index} del Prompt {PROMPT_STAGE1_ID}."
            )

        if danger_match is None:
            raise ValueError(
                f"Danger Level mancante o non valido "
                f"nella Clip {clip_index} del Prompt {PROMPT_STAGE1_ID}."
            )

        raw_assessment = (
            assessment_match.group(1).upper()
        )

        if raw_assessment == "NORMAL":
            assessment = "NORMAL"

        elif raw_assessment == "ANOMALOUS":
            assessment = "ANOMALOUS"

        elif raw_assessment == "NONE":
            assessment = "NORMAL"

        elif raw_assessment in {"WARNING", "DANGER"}:
            assessment = "ANOMALOUS"

        else:
            raise ValueError(
                f"Assessment non riconosciuto "
                f"nella Clip {clip_index}: "
                f"{raw_assessment}"
            )

        danger_level = int(
            danger_match.group(1)
        )

        if danger_level == 0:
            expected_assessment = "NORMAL"
        else:
            expected_assessment = "ANOMALOUS"

        if assessment != expected_assessment:
            raise ValueError(
                f"Incoerenza nella Clip {clip_index}: "
                f"Assessment={assessment}, "
                f"Danger Level={danger_level}"
            )

        if danger_level not in VALID_DANGER_LEVELS:
            raise ValueError(
                f"Danger Level Qwen non valido "
                f"nella Clip {clip_index}: {danger_level}"
            )

        if assessment == "NORMAL" and danger_level != 0:
            raise ValueError(
                f"Incoerenza Prompt {PROMPT_STAGE1_ID} "
                f"nella Clip {clip_index}: "
                f"NORMAL con Danger Level {danger_level}."
            )

        if assessment == "ANOMALOUS" and danger_level == 0:
            raise ValueError(
                f"Incoerenza Prompt {PROMPT_STAGE1_ID} "
                f"nella Clip {clip_index}: "
                f"ANOMALOUS con Danger Level 0."
            )

        if clip_index in parsed:
            raise ValueError(
                f"Clip {clip_index} duplicata nelle "
               f"clip_evaluations del Prompt {PROMPT_STAGE1_ID}."
            )

        parsed[clip_index] = {
            "start_time": (
                float(start_match.group(1))
                if start_match
                else None
            ),
            "end_time": (
                float(end_match.group(1))
                if end_match
                else None
            ),
            "assessment": assessment,
            "danger_level": danger_level,
            "anomaly_type": (
                anomaly_type_match.group(1).strip()
                if anomaly_type_match
                else None
            ),
            "evidence": (
                evidence_match.group(1).strip()
                if evidence_match
                else None
            ),
            "qwen_report": (
                f"Clip {clip_index}\n"
                f"{block.strip()}"
            )

        }

    return parsed


def validate_ground_truth(source_file):
    """
    Retrieve Ground Truth for a source AI video.
    """

    if source_file not in GROUND_TRUTH:
        raise KeyError(
            "\nVideo non presente nella Ground Truth:\n"
            f"{source_file}\n\n"
            "Controlla che il nome nel manifest sia identico "
            "a quello presente in ground_truth.py."
        )

    gt_level = int(
        GROUND_TRUTH[source_file]
    )

    if gt_level not in VALID_DANGER_LEVELS:
        raise ValueError(
            f"Ground Truth non valida per {source_file}: "
            f"{gt_level}"
        )

    return gt_level


# ============================================================
# CARICAMENTO DATI
# ============================================================

print("=" * 70)
print(
    f"COMPOSITE EVALUATION - "
    f"{PIPELINE_MODE.upper()}"
)
print("=" * 70)

print(f"\nVideo ID: {VIDEO_ID}")
print(f"Composite type: {COMPOSITE_TYPE}")
print(f"Prompt model: {PROMPT_MODEL}")
print(f"Prompt Stage 1 ID: {PROMPT_STAGE1_ID}")
print(f"Prompt Stage 2 ID: {PROMPT_STAGE2_ID}")

print(f"\nManifest: {MANIFEST_PATH}")
print(f"Evaluation: {EVALUATION_PATH}")

print("\nCaricamento manifest...")

manifest = load_json(
    MANIFEST_PATH
)

print("Caricamento evaluation Qwen...")

evaluation = load_json(
    EVALUATION_PATH
)

holmes_reports_by_index = {
    int(item["segment_index"]): item.get("holmes_report")
    for item in evaluation.get("segments", [])
}


# ============================================================
# CONTROLLI
# ============================================================

segments = manifest.get(
    "segments",
    []
)

clip_evaluations_text = evaluation.get(
    "clip_evaluations",
    ""
)

timeline_phases = evaluation.get(
    "timeline_phases",
    []
)

if not segments:
    raise ValueError(
        "Il manifest non contiene segmenti."
    )

if not clip_evaluations_text:
    raise ValueError(
        "evaluation.json non contiene clip_evaluations "
        f"prodotte dal Prompt {PROMPT_STAGE1_ID}."
    )

clip_predictions = parse_clip_evaluations(
    clip_evaluations_text
)

if not clip_predictions:
    raise ValueError(
        "Impossibile estrarre le valutazioni "
        f"clip-level del Prompt {PROMPT_STAGE1_ID}."
    )

if not timeline_phases:
    raise ValueError(
        "evaluation.json non contiene timeline_phases "
        f"prodotte dal Prompt {PROMPT_STAGE2_ID}."
    )


# ============================================================
# COSTRUZIONE GT / PRED
# ============================================================

comparisons = []

y_true = []
y_pred = []

# Contiene SOLO classificazioni errate con predizione valida.
misclassifications = []


for segment in segments:

    segment_index = int(
        segment["segment_index"]
    )

    segment_file = segment[
        "segment_file"
    ]

    source_file = segment[
        "source_file"
    ]

    segment_start = float(
        segment["start_time"]
    )

    segment_end = float(
        segment["end_time"]
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    gt_level = validate_ground_truth(
        source_file
    )

    gt_description = GROUND_TRUTH_DESCRIPTIONS.get(
        source_file
    )

    # --------------------------------------------------------
    # QWEN PREDICTION - PROMPT 10
    # --------------------------------------------------------

    clip_prediction = clip_predictions.get(
        segment_index
    )

    if clip_prediction is None:

        print(
            "\nATTENZIONE:"
            f" nessuna valutazione Prompt {PROMPT_STAGE1_ID} trovata "
            f"per il segmento {segment_index}."
        )

        comparison = {
            "segment_index": segment_index,
            "segment_file": segment_file,
            "source_file": source_file,
            "start_time": segment_start,
            "end_time": segment_end,
            "ground_truth": gt_level,
            "ground_truth_label": DANGER_LABELS[gt_level],
            "predicted": None,
            "predicted_label": None,
            "correct": False,
            "absolute_error": None,
            "prediction_source": f"prompt_{PROMPT_STAGE1_ID}",
            "clip_evaluation": None
        }

        comparisons.append(
            comparison
        )

        continue

    predicted_level = int(
        clip_prediction["danger_level"]
    )

    # --------------------------------------------------------
    # SALVATAGGIO VETTORI
    # --------------------------------------------------------

    y_true.append(
        gt_level
    )

    y_pred.append(
        predicted_level
    )

    # --------------------------------------------------------
    # CONFRONTO
    # --------------------------------------------------------

    comparison = {
        "segment_index": segment_index,
        "segment_file": segment_file,
        "source_file": source_file,

        "start_time": segment_start,
        "end_time": segment_end,

        "ground_truth": gt_level,
        "ground_truth_label": DANGER_LABELS[
            gt_level
        ],

        "predicted": predicted_level,
        "predicted_label": DANGER_LABELS[
            predicted_level
        ],

        "correct": (
            gt_level == predicted_level
        ),

        "absolute_error": abs(
            gt_level - predicted_level
        ),

        "prediction_source": f"prompt_{PROMPT_STAGE1_ID}",
        "clip_evaluation": clip_prediction
    }

    comparisons.append(
        comparison
    )

    if predicted_level != gt_level:
        error_direction = (
            "SOTTOSTIMA"
            if predicted_level < gt_level
            else "SOVRASTIMA"
        )

        misclassifications.append({
            "sequence_id": VIDEO_ID,
            "segment_index": segment_index,
            "segment_file": segment_file,
            "source_file": source_file,
            "start_time": segment_start,
            "end_time": segment_end,
            "ground_truth": gt_level,
            "ground_truth_label": DANGER_LABELS[gt_level],
            "predicted": predicted_level,
            "predicted_label": DANGER_LABELS[predicted_level],
            "error_type": f"{gt_level} -> {predicted_level}",
            "error_direction": error_direction,
            "holmes_report": holmes_reports_by_index.get(segment_index),
            "qwen_evaluation": clip_prediction,
            "qwen_report": clip_prediction.get(
                "qwen_report"
            ),
            "ground_truth_description": gt_description
        })


# ============================================================
# RISULTATI
# ============================================================

valid_comparisons = [
    item
    for item in comparisons
    if item["predicted"] is not None
]

correct_predictions = sum(
    1
    for item in valid_comparisons
    if item["correct"]
)

if valid_comparisons:

    segment_accuracy = (
        correct_predictions
        / len(valid_comparisons)
    )

else:

    segment_accuracy = 0.0


# ============================================================
# OVERALL VIDEO/SEQUENCE EVALUATION
# ============================================================

# Ground Truth overall dell'intera sequenza.
# È indipendente dalla presenza o meno di una predizione Prompt 10.
overall_ground_truth = max(
    item["ground_truth"]
    for item in comparisons
) if comparisons else None

# Predizione overall ufficiale:
# deriva dal Prompt 11 / timeline ricostruita.
overall_predicted = max(
    int(phase["danger_level"])
    for phase in timeline_phases
) if timeline_phases else None

if (
    overall_predicted is not None
    and overall_predicted not in VALID_DANGER_LEVELS
):
    raise ValueError(
        f"Overall predicted Prompt {PROMPT_STAGE2_ID} non valido: "
        f"{overall_predicted}"
    )

# Massimo danger delle singole clip del Prompt 10.
# Non viene usato come overall ufficiale:
# serve solo per controllare la coerenza tra i due stage.
overall_predicted_stage1 = max(
    int(item["danger_level"])
    for item in clip_predictions.values()
) if clip_predictions else None

if (
    overall_predicted is not None
    and overall_predicted_stage1 is not None
    and overall_predicted != overall_predicted_stage1
):
    print(
        "\nATTENZIONE: incoerenza tra i due stage Qwen:"
        f"\nPrompt {PROMPT_STAGE1_ID} max danger = "
        f"{overall_predicted_stage1}"
        f"\nPrompt {PROMPT_STAGE2_ID} overall/max phase danger = "
        f"{overall_predicted}"
    )


# ============================================================
# OUTPUT JSON
# ============================================================

comparison_data = {
    "video_id": VIDEO_ID,

    "composite_type": COMPOSITE_TYPE,
    "second_model": SECOND_MODEL,
    "prompt_model": PROMPT_MODEL,
    "prompt_stage1_id": PROMPT_STAGE1_ID,
    "prompt_stage2_id": PROMPT_STAGE2_ID,

    "number_of_segments": len(
        segments
    ),

    "number_of_valid_predictions": len(
        valid_comparisons
    ),

    "segment_accuracy": segment_accuracy,

    "overall_ground_truth": overall_ground_truth,

    # Overall ufficiale della sequenza: Prompt 11.
    "overall_predicted": overall_predicted,

    # Solo controllo diagnostico: massimo delle clip del Prompt 10.
    "overall_predicted_stage1": overall_predicted_stage1,

    "overall_danger_score": evaluation.get(
        "overall_danger_score"
    ),

    # Questi vettori usano SOLO le predizioni clip-level del Prompt 10.
    "y_true": y_true,
    "y_pred": y_pred,

    "comparisons": comparisons
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        comparison_data,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# ERROR ANALYSIS - SOLO CLASSIFICAZIONI ERRATE
# ============================================================

error_counts = {
    "0 -> 1": 0,
    "0 -> 2": 0,
    "1 -> 0": 0,
    "1 -> 2": 0,
    "2 -> 0": 0,
    "2 -> 1": 0
}

for item in misclassifications:
    if item["error_type"] in error_counts:
        error_counts[item["error_type"]] += 1

misclassification_data = {
    "sequence_id": VIDEO_ID,
    "composite_type": COMPOSITE_TYPE,
    "prompt_model": PROMPT_MODEL,
    "prompt_stage1_id": PROMPT_STAGE1_ID,
    "prompt_stage2_id": PROMPT_STAGE2_ID,
    "number_of_segments": len(segments),
    "number_of_valid_predictions": len(valid_comparisons),
    "number_of_misclassifications": len(misclassifications),
    "segment_accuracy": segment_accuracy,
    "error_distribution": error_counts,
    "misclassifications": misclassifications
}

with open(MISCLASSIFICATIONS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(misclassification_data, f, indent=4, ensure_ascii=False)

def _format_qwen_evaluation(qwen_eval):
    if not qwen_eval:
        return "Non disponibile"
    return "\n".join([
        f"Assessment: {qwen_eval.get('assessment', 'N/A')}",
        f"Danger Level: {qwen_eval.get('danger_level', 'N/A')}",
        f"Anomaly Type: {qwen_eval.get('anomaly_type', 'N/A')}",
        f"Evidence: {qwen_eval.get('evidence', 'N/A')}"
    ])

with open(MISCLASSIFICATIONS_TXT_PATH, "w", encoding="utf-8") as f:
    f.write("ERROR ANALYSIS - AI SEQUENCE\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Sequence ID: {VIDEO_ID}\n")
    f.write(f"Total segments: {len(segments)}\n")
    f.write(f"Valid predictions: {len(valid_comparisons)}\n")
    f.write(f"Misclassifications: {len(misclassifications)}\n")
    f.write(f"Segment accuracy: {segment_accuracy:.4f} ({segment_accuracy * 100:.2f}%)\n\n")
    f.write("ERROR DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    for error_type, count in error_counts.items():
        f.write(f"{error_type}: {count}\n")
    f.write("\n")

    if not misclassifications:
        f.write("Nessuna classificazione errata.\n")
    else:
        for number, item in enumerate(misclassifications, start=1):
            f.write("=" * 80 + "\n")
            f.write(f"ERRORE {number}\n")
            f.write("=" * 80 + "\n")
            f.write(f"SEQUENZA AI: {item['sequence_id']}\n")
            f.write(f"SEGMENTO: {item['segment_index']}\n")
            f.write(f"VIDEO SORGENTE: {item['source_file']}\n")
            f.write(f"INTERVALLO: {item['start_time']:.2f}s - {item['end_time']:.2f}s\n\n")
            f.write("GROUND TRUTH\n")
            f.write("-" * 80 + "\n")
            f.write(f"Danger Level: {item['ground_truth']} - {item['ground_truth_label']}\n")
            f.write(
                "Descrizione GT: "
                f"{item.get('ground_truth_description') or 'Non disponibile.'}"
                "\n\n"
            )
            f.write("PREDIZIONE QWEN\n")
            f.write("-" * 80 + "\n")
            f.write(_format_qwen_evaluation(item["qwen_evaluation"]) + "\n\n")
            f.write("ERRORE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Tipo: {item['error_type']}\n")
            f.write(f"Direzione: {item['error_direction']}\n\n")
            f.write("REPORT HOLMES-VAU\n")
            f.write("-" * 80 + "\n")
            f.write((item["holmes_report"] or "Non disponibile").strip() + "\n\n")



# ============================================================
# CSV ERROR ANALYSIS - SOLO MISCLASSIFICATIONS
# ============================================================

CSV_FIELDS = [
    "sequence_id",
    "segment_index",
    "source_file",
    "start_time",
    "end_time",

    "ground_truth",
    "ground_truth_label",

    "predicted",
    "predicted_label",

    "error_type",
    "error_direction",

    "holmes_report",
    "qwen_report",

    "ground_truth_description"
]


with open(
    MISCLASSIFICATIONS_CSV_PATH,
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

    for item in misclassifications:

        writer.writerow({
            "sequence_id":
                item["sequence_id"],

            "segment_index":
                item["segment_index"],

            "source_file":
                item["source_file"],

            "start_time":
                item["start_time"],

            "end_time":
                item["end_time"],

            "ground_truth":
                item["ground_truth"],

            "ground_truth_label":
                item["ground_truth_label"],

            "predicted":
                item["predicted"],

            "predicted_label":
                item["predicted_label"],

            "error_type":
                item["error_type"],

            "error_direction":
                item["error_direction"],

            "holmes_report":
                item.get(
                    "holmes_report"
                ),

            "qwen_report":
                item.get(
                    "qwen_report"
                ),

            "ground_truth_description":
                item.get(
                    "ground_truth_description"
                )
        })
# ============================================================
# STAMPA RISULTATI
# ============================================================

print("\n" + "=" * 70)
print(
    f"SEGMENT COMPARISON - PROMPT "
    f"{PROMPT_STAGE1_ID}"
)
print("=" * 70)


for item in comparisons:

    print()

    print(
        f"Segment {item['segment_index']}"
    )

    print(
        f"Source: {item['source_file']}"
    )

    print(
        f"Time: "
        f"{item['start_time']:.3f} - "
        f"{item['end_time']:.3f} seconds"
    )

    print(
        f"GT: "
        f"{item['ground_truth']} - "
        f"{item['ground_truth_label']}"
    )

    if item["predicted"] is None:

        print(
            "PRED: NO VALID PREDICTION"
        )

    else:

        print(
            f"PRED: "
            f"{item['predicted']} - "
            f"{item['predicted_label']}"
        )

        print(
            f"Correct: {item['correct']}"
        )

        print(
            f"Absolute error: "
            f"{item['absolute_error']}"
        )


print("\n" + "=" * 70)
print(
    f"FINAL VECTORS - PROMPT "
    f"{PROMPT_STAGE1_ID}"
)
print("=" * 70)



print(
    f"y_true = {y_true}"
)

print(
    f"y_pred = {y_pred}"
)

print(
    f"\nSegment accuracy: "
    f"{segment_accuracy:.4f}"
)

print(
    f"Valid predictions: "
    f"{len(valid_comparisons)} / "
    f"{len(segments)}"
)

print("\n" + "=" * 70)
print(
    f"OVERALL SEQUENCE - PROMPT "
    f"{PROMPT_STAGE2_ID}"
)
print("=" * 70)

print(
    f"Overall GT: {overall_ground_truth}"
)

print(
    f"Overall predicted: {overall_predicted}"
)

print(
    f"Prompt {PROMPT_STAGE1_ID} "
    f"max danger: {overall_predicted_stage1}"
)

print(
    f"\nComparison salvato in:\n"
    f"{OUTPUT_PATH}"
)

print(
    f"\nError analysis JSON salvato in:\n"
    f"{MISCLASSIFICATIONS_JSON_PATH}"
)

print(
    f"\nError analysis TXT salvato in:\n"
    f"{MISCLASSIFICATIONS_TXT_PATH}"
)

print(
    f"\nClassificazioni errate: {len(misclassifications)}"
)

print(
    f"\nError analysis CSV salvato in:\n"
    f"{MISCLASSIFICATIONS_CSV_PATH}"
)

print("=" * 70)
