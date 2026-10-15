import os
import json
import re
import csv


# ============================================================
# CONFIGURAZIONE ESPERIMENTO
# ============================================================
# La pipeline single clip crea una copia temporanea nella root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "evaluate"
    else SCRIPT_DIR
)

VIDEO_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset classificato"
)

GROUND_TRUTH_DIR = os.path.join(
    VIDEO_DIR,
    "annotation"
)

PROMPT_MODEL = "qwen"

PIPELINE_MODE = "two_stage"
# "two_stage"
# "single_clip_stage1"

if PIPELINE_MODE == "two_stage":
    PROMPT_STAGE1_ID = 12
    PROMPT_STAGE2_ID = 13

elif PIPELINE_MODE == "single_clip_stage1":
    PROMPT_STAGE1_ID = 15
    PROMPT_STAGE2_ID = 13

else:
    raise ValueError(
        "PIPELINE_MODE non valido."
    )

INFERENCE_RESULTS_PATH = os.path.join(
    PROJECT_ROOT,
    f"results_large_videos_prompt_"
    f"{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}.txt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "large_video_evaluation",
    PROMPT_MODEL,
    f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

VALID_DANGER_LEVELS = {0, 1, 2}
DANGER_LABELS = {
    0: "NONE",
    1: "WARNING",
    2: "DANGER"
}

# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(ground_truth_path):

    with open(
        ground_truth_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# FIND TWO-STAGE QWEN BLOCK
# ============================================================

def extract_two_stage_report(
    results_text,
    video_name
):
    """
    Estrae dal file di inferenza:
    - output Prompt 12 (clip-level evaluations)
    - output Prompt 13 (final report)
    per il video richiesto.
    """

    pattern = re.compile(
        r"VIDEO:\s*(?P<video>[^\n]+)\n"
        r"MODEL:\s*(?P<model>[^\n]+)\n"
        r"PROMPT STAGE 1 ID:\s*(?P<stage1>\d+)\n"
        r"PROMPT STAGE 2 ID:\s*(?P<stage2>\d+)\n"
        r"STAGE 1 CLIP EVALUATIONS:\n"
        r"(?P<clip_evaluations>.*?)"
        r"\n\s*FINAL REPORT:\n"
        r"(?P<final_report>.*?)"
        r"(?=\n={20,}|\Z)",
        re.DOTALL | re.IGNORECASE
    )

    for match in pattern.finditer(
        results_text
    ):

        video_path = (
            match.group("video").strip()
        )

        current_video_name = os.path.basename(
            video_path.replace("\\", "/")
        )

        if current_video_name != video_name:
            continue

        stage1_id = int(
            match.group("stage1")
        )

        stage2_id = int(
            match.group("stage2")
        )

        if stage1_id != PROMPT_STAGE1_ID:
            raise ValueError(
                f"Prompt Stage 1 inatteso per "
                f"{video_name}: {stage1_id}"
            )

        if stage2_id != PROMPT_STAGE2_ID:
            raise ValueError(
                f"Prompt Stage 2 inatteso per "
                f"{video_name}: {stage2_id}"
            )

        return (
            match.group(
                "clip_evaluations"
            ).strip(),
            match.group(
                "final_report"
            ).strip()
        )

    raise ValueError(
        f"Output Qwen two-stage non trovato per: "
        f"{video_name}"
    )

def extract_holmes_reports(
    results_text,
    video_name
):
    """
    Estrae i report Holmes-VAU clip-level
    dal file results_large_videos_prompt_12_13.txt.

    Restituisce:
        {
            clip_index: "report Holmes completo",
            ...
        }
    """

    pattern = re.compile(
        r"VIDEO:\s*(?P<video>[^\n]+)\n"
        r"CLIP\s+(?P<clip>\d+):\s*"
        r"(?P<start>[0-9.]+)s\s*-\s*"
        r"(?P<end>[0-9.]+)s\n"
        r"HOLMES:\s*(?P<report>.*?)\n"
        r"FRAMES:\s*(?P<frames>.*?)\n",
        re.DOTALL
    )

    reports = {}

    for match in pattern.finditer(
        results_text
    ):

        video_path = (
            match.group("video").strip()
        )

        current_video_name = os.path.basename(
            video_path.replace("\\", "/")
        )

        if current_video_name != video_name:
            continue

        clip_index = int(
            match.group("clip")
        )

        reports[clip_index] = (
            match.group("report").strip()
        )

    return reports

# ============================================================
# PARSE DANGER LEVEL
# ============================================================

def parse_danger_level(text):

    match = re.search(
        r"\b([0-2])\s*-\s*"
        r"(NONE|WARNING|DANGER)",
        text,
        re.IGNORECASE
    )

    if match:
        return int(
            match.group(1)
        )

    match = re.search(
        r"\b([0-2])\b",
        text
    )

    if match:
        return int(
            match.group(1)
        )

    return None


# ============================================================
# PARSE PROMPT 12 CLIP EVALUATIONS
# ============================================================

def parse_clip_evaluations(
    clip_evaluations_text
):
    """
    Estrae le predizioni clip-level prodotte dal Prompt 12.

    Restituisce:
        {
            clip_index: {
                "start": float | None,
                "end": float | None,
                "assessment": str,
                "danger_level": int,
                "anomaly_type": str | None,
                "evidence": str | None
            }
        }
    """

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
            block_end = (
                matches[index + 1].start()
            )
        else:
            block_end = len(
                clip_evaluations_text
            )

        block = clip_evaluations_text[
            block_start:block_end
        ]

        start_match = re.search(
            r"(?mi)^Start:\s*"
            r"([0-9]+(?:\.[0-9]+)?)",
            block
        )

        end_match = re.search(
            r"(?mi)^End:\s*"
            r"([0-9]+(?:\.[0-9]+)?)",
            block
        )

        assessment_match = re.search(
            r"(?mi)^Assessment:\s*"
            r"(NORMAL|ANOMALOUS)\s*$",
            block
        )

        danger_match = re.search(
            r"(?mi)^Danger Level:\s*"
            r"([0-2]\s*-\s*"
            r"(?:NONE|WARNING|DANGER))\s*$",
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
            raise ValueError(
                f"Assessment Prompt 12 mancante "
                f"o non valido nella Clip "
                f"{clip_index}."
            )

        if danger_match is None:
            raise ValueError(
                f"Danger Level Prompt 12 "
                f"mancante o non valido nella "
                f"Clip {clip_index}."
            )

        assessment = (
            assessment_match
            .group(1)
            .upper()
        )

        danger_level = (
            parse_danger_level(
                danger_match.group(1)
            )
        )

        if danger_level not in (
            VALID_DANGER_LEVELS
        ):
            raise ValueError(
                f"Danger Level Prompt 12 "
                f"non valido nella Clip "
                f"{clip_index}: "
                f"{danger_level}"
            )

        if (
            assessment == "NORMAL"
            and danger_level != 0
        ):
            raise ValueError(
                f"Incoerenza Prompt 12 nella "
                f"Clip {clip_index}: NORMAL "
                f"con Danger Level "
                f"{danger_level}."
            )

        if (
            assessment == "ANOMALOUS"
            and danger_level == 0
        ):
            raise ValueError(
                f"Incoerenza Prompt 12 nella "
                f"Clip {clip_index}: "
                f"ANOMALOUS con Danger Level 0."
            )

        if clip_index in parsed:
            raise ValueError(
                f"Clip {clip_index} duplicata "
                f"nel Prompt 12."
            )

        parsed[clip_index] = {
            "start": (
                float(
                    start_match.group(1)
                )
                if start_match
                else None
            ),
            "end": (
                float(
                    end_match.group(1)
                )
                if end_match
                else None
            ),
            "assessment": assessment,
            "danger_level": danger_level,
            "anomaly_type": (
                anomaly_type_match
                .group(1)
                .strip()
                if anomaly_type_match
                else None
            ),
            "evidence": (
                evidence_match
                .group(1)
                .strip()
                if evidence_match
                else None
            ),
            "qwen_report": (
                f"Clip {clip_index}\n"
                f"{block.strip()}"
            )
        }

    return parsed


# ============================================================
# PARSE OVERALL DANGER - PROMPT 13
# ============================================================

def parse_overall_danger(report):

    match = re.search(
        r"Overall Danger Level:\s*\n?\s*"
        r"([0-2]\s*-\s*"
        r"(?:NONE|WARNING|DANGER))",
        report,
        re.IGNORECASE
    )

    if not match:
        return None

    return parse_danger_level(
        match.group(1)
    )


# ============================================================
# PARSE OVERALL CLASSIFICATION
# ============================================================

def parse_overall_classification(
    report
):

    match = re.search(
        r"Overall Classification:"
        r"\s*\n?\s*"
        r"(NORMAL|ANOMALOUS)",
        report,
        re.IGNORECASE
    )

    if not match:
        return None

    return (
        match.group(1).upper()
    )


# ============================================================
# PARSE OVERALL DANGER SCORE
# ============================================================

def parse_danger_score(report):

    match = re.search(
        r"Overall Danger Score:"
        r"\s*\n?\s*(\d+)",
        report,
        re.IGNORECASE
    )

    if not match:
        return None

    score = int(
        match.group(1)
    )

    return max(
        0,
        min(100, score)
    )


# ============================================================
# PARSE QWEN TIMELINE - PROMPT 13
# ============================================================

def parse_timeline(report):

    timeline_match = re.search(
        r"Timeline:\s*(.*?)"
        r"(?=\n\s*Anomaly Events:|\Z)",
        report,
        re.DOTALL | re.IGNORECASE
    )

    if not timeline_match:
        return []

    timeline_text = (
        timeline_match.group(1)
    )

    phase_pattern = re.compile(
        r"Phase\s+(\d+)\s*"
        r".*?"
        r"Start:\s*"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s*(?:seconds?)?\s*"
        r".*?"
        r"End:\s*"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s*(?:seconds?)?\s*"
        r".*?"
        r"Assessment:\s*"
        r"(NORMAL|ANOMALOUS)\s*"
        r".*?"
        r"Danger Level:\s*"
        r"([0-2]\s*-\s*"
        r"(?:NONE|WARNING|DANGER))",
        re.DOTALL | re.IGNORECASE
    )

    phases = []

    for match in phase_pattern.finditer(
        timeline_text
    ):

        phase_number = int(
            match.group(1)
        )

        start = float(
            match.group(2)
        )

        end = float(
            match.group(3)
        )

        assessment = (
            match.group(4).upper()
        )

        danger_level = (
            parse_danger_level(
                match.group(5)
            )
        )

        phases.append({
            "phase": phase_number,
            "start": start,
            "end": end,
            "assessment": assessment,
            "danger_level":
                danger_level
        })

    return phases


# ============================================================
# EVALUATION
# ============================================================

def evaluate(video_name):
    video_base = os.path.splitext(
        video_name
    )[0]

    ground_truth_path = os.path.join(
        GROUND_TRUTH_DIR,
        video_base + "_annotation.json"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        video_base + "_comparison.json"
    )
    misclassifications_json_path = os.path.join(
        OUTPUT_DIR,
        video_base + "_misclassifications.json"
    )

    misclassifications_txt_path = os.path.join(
        OUTPUT_DIR,
        video_base + "_misclassifications.txt"
    )

    misclassifications_csv_path = os.path.join(
        OUTPUT_DIR,
        video_base + "_misclassifications.csv"
    )
    ground_truth = (
        load_ground_truth(
            ground_truth_path
        )
    )

    with open(
        INFERENCE_RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        results_text = f.read()

    holmes_reports = extract_holmes_reports(
        results_text,
        video_name
    )
    if not holmes_reports:
        print(
            f"\nATTENZIONE: nessun report Holmes "
            f"trovato per {video_name}"
        )
    (
        clip_evaluations_text,
        final_report
    ) = extract_two_stage_report(
        results_text,
        video_name
    )

    # --------------------------------------------------------
    # PROMPT 12
    # --------------------------------------------------------

    clip_predictions = (
        parse_clip_evaluations(
            clip_evaluations_text
        )
    )

    if not clip_predictions:
        raise ValueError(
            f"Nessuna valutazione Prompt 12 "
            f"trovata per {video_name}"
        )

    # --------------------------------------------------------
    # PROMPT 13
    # --------------------------------------------------------

    qwen_phases = parse_timeline(
        final_report
    )

    if not qwen_phases:
        raise ValueError(
            f"Nessuna fase Prompt 13 "
            f"trovata per {video_name}"
        )

    for phase in qwen_phases:

        if (
            phase["danger_level"]
            not in VALID_DANGER_LEVELS
        ):
            raise ValueError(
                f"Danger Level Prompt 13 "
                f"non valido nella fase "
                f"{phase['phase']}: "
                f"{phase['danger_level']}"
            )

    predicted_overall = (
        parse_overall_danger(
            final_report
        )
    )

    if predicted_overall is None:
        raise ValueError(
            f"Overall Danger Level "
            f"Prompt 13 non trovato per "
            f"{video_name}"
        )

    if (
        predicted_overall
        not in VALID_DANGER_LEVELS
    ):
        raise ValueError(
            f"Predicted overall danger "
            f"non valido: "
            f"{predicted_overall}"
        )

    timeline_overall = max(
        phase["danger_level"]
        for phase in qwen_phases
    )

    if (
        predicted_overall
        != timeline_overall
    ):
        raise ValueError(
            f"Incoerenza Prompt 13 per "
            f"{video_name}: "
            f"overall={predicted_overall}, "
            f"timeline max="
            f"{timeline_overall}"
        )

    prompt12_overall = max(
        item["danger_level"]
        for item
        in clip_predictions.values()
    )

    if (
        predicted_overall
        != prompt12_overall
    ):
        print(
            "\nATTENZIONE: incoerenza "
            "tra i due stage Qwen:"
        )
        print(
            "Prompt 12 max danger:",
            prompt12_overall
        )
        print(
            "Prompt 13 overall:",
            predicted_overall
        )

    predicted_classification = (
        parse_overall_classification(
            final_report
        )
    )

    danger_score = parse_danger_score(
        final_report
    )

    # --------------------------------------------------------
    # CLIP-LEVEL GT vs PROMPT 12
    # --------------------------------------------------------

    comparisons = []

    y_true = []
    y_pred = []

    misclassifications = []

    for gt_clip in ground_truth["clips"]:

        clip_index = int(
            gt_clip["clip_index"]
        )

        start = float(
            gt_clip["start"]
        )

        end = float(
            gt_clip["end"]
        )

        true_level = int(
            gt_clip["danger_level"]
        )

        gt_description = gt_clip.get(
            "description"
        )

        if (
            true_level
            not in VALID_DANGER_LEVELS
        ):
            raise ValueError(
                f"Ground Truth danger "
                f"level non valido: "
                f"{true_level}"
            )

        clip_prediction = (
            clip_predictions.get(
                clip_index
            )
        )

        if clip_prediction is None:

            comparison = {
                "clip_index":
                    clip_index,
                "start":
                    start,
                "end":
                    end,
                "ground_truth":
                    true_level,
                "ground_truth_description":
                    gt_description,
                "predicted":
                    None,
                "correct":
                    False,
                "prediction_source":
                    "prompt_12",
                "clip_evaluation":
                    None,
                "qwen_report": None,
                "holmes_report":
                    holmes_reports.get(
                        clip_index
                    )

            }

            comparisons.append(
                comparison
            )

            print(
                "\nATTENZIONE: "
                f"nessuna predizione "
                f"Prompt 12 per Clip "
                f"{clip_index}."
            )

            continue

        predicted_level = int(
            clip_prediction[
                "danger_level"
            ]
        )

        comparison = {
            "clip_index":
                clip_index,
            "start":
                start,
            "end":
                end,
            "ground_truth":
                true_level,
            "ground_truth_description":
                gt_description,
            "predicted":
                predicted_level,
            "correct":
                (
                    predicted_level
                    == true_level
                ),
            "prediction_source":
                "prompt_12",
            "clip_evaluation":
                clip_prediction,
            "qwen_report":
                clip_prediction.get(
                    "qwen_report"
                ),
            "holmes_report":
                holmes_reports.get(
                    clip_index
                )
        }

        comparisons.append(
            comparison
        )

        y_true.append(
            true_level
        )

        y_pred.append(
            predicted_level
        )

        if predicted_level != true_level:
            error_direction = (
                "SOTTOSTIMA"
                if predicted_level < true_level
                else "SOVRASTIMA"
            )

            misclassifications.append({
                "video": video_name,

                "clip_index": clip_index,

                "start": start,
                "end": end,

                "ground_truth": true_level,
                "ground_truth_label":
                    DANGER_LABELS[true_level],

                "ground_truth_description":
                    gt_description,

                "predicted": predicted_level,
                "predicted_label":
                    DANGER_LABELS[predicted_level],

                "error_type":
                    f"{true_level} -> {predicted_level}",

                "error_direction":
                    error_direction,

                "holmes_report":
                    holmes_reports.get(
                        clip_index
                    ),

                "qwen_report":
                    clip_prediction.get(
                        "qwen_report"
                    )
            })

    # --------------------------------------------------------
    # ACCURACY CLIP-LEVEL
    # --------------------------------------------------------

    if y_true:

        correct = sum(
            true == pred
            for true, pred
            in zip(
                y_true,
                y_pred
            )
        )

        accuracy = (
            correct / len(y_true)
        )

    else:

        accuracy = None

    # --------------------------------------------------------
    # OVERALL GROUND TRUTH
    # --------------------------------------------------------

    gt_overall = int(
        ground_truth[
            "overall_danger_level"
        ]
    )

    if (
        gt_overall
        not in VALID_DANGER_LEVELS
    ):
        raise ValueError(
            f"Ground Truth overall "
            f"non valida: "
            f"{gt_overall}"
        )

    # --------------------------------------------------------
    # OUTPUT JSON
    # --------------------------------------------------------

    evaluation = {

        "video":
            video_name,

        "prompt_model":
            PROMPT_MODEL,

        "prompt_stage1_id":
            PROMPT_STAGE1_ID,

        "prompt_stage2_id":
            PROMPT_STAGE2_ID,

        "ground_truth_overall_classification":
            ground_truth[
                "overall_classification"
            ],

        "predicted_overall_classification":
            predicted_classification,

        "overall_ground_truth":
            gt_overall,

        "overall_predicted":
            predicted_overall,

        "overall_predicted_stage1":
            prompt12_overall,

        "overall_danger_correct":
            predicted_overall
            == gt_overall,

        "overall_danger_score":
            danger_score,

        "number_of_clips":
            len(
                ground_truth["clips"]
            ),

        "number_of_valid_predictions":
            len(y_pred),

        "clip_accuracy":
            accuracy,

        # Questi vettori usano SOLO
        # le predizioni clip-level
        # del Prompt 12.
        "y_true":
            y_true,

        "y_pred":
            y_pred,

        # Timeline globale prodotta
        # dal Prompt 13.
        "qwen_timeline":
            qwen_phases,

        "clip_comparisons":
            comparisons
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=4,
            ensure_ascii=False
        )

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
        "video": video_name,

        "prompt_model": PROMPT_MODEL,
        "prompt_stage1_id": PROMPT_STAGE1_ID,
        "prompt_stage2_id": PROMPT_STAGE2_ID,

        "number_of_clips":
            len(ground_truth["clips"]),

        "number_of_valid_predictions":
            len(y_pred),

        "number_of_misclassifications":
            len(misclassifications),

        "clip_accuracy":
            accuracy,

        "error_distribution":
            error_counts,

        "misclassifications":
            misclassifications
    }

    with open(
            misclassifications_json_path,
            "w",
            encoding="utf-8"
    ) as f:

        json.dump(
            misclassification_data,
            f,
            indent=4,
            ensure_ascii=False
        )

    with open(
            misclassifications_txt_path,
            "w",
            encoding="utf-8"
    ) as f:

        f.write("ERROR ANALYSIS - LARGE VIDEO\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Video: {video_name}\n")
        f.write(
            f"Total clips: "
            f"{len(ground_truth['clips'])}\n"
        )
        f.write(
            f"Valid predictions: "
            f"{len(y_pred)}\n"
        )
        f.write(
            f"Misclassifications: "
            f"{len(misclassifications)}\n"
        )

        if accuracy is not None:
            f.write(
                f"Clip accuracy: "
                f"{accuracy:.4f} "
                f"({accuracy * 100:.2f}%)\n\n"
            )
        else:
            f.write(
                "Clip accuracy: N/A\n\n"
            )

        f.write("ERROR DISTRIBUTION\n")
        f.write("-" * 80 + "\n")

        for error_type, count in error_counts.items():
            f.write(
                f"{error_type}: {count}\n"
            )

        f.write("\n")

        if not misclassifications:

            f.write(
                "Nessuna classificazione errata.\n"
            )

        else:

            for number, item in enumerate(
                    misclassifications,
                    start=1
            ):
                f.write("=" * 80 + "\n")
                f.write(
                    f"ERRORE {number}\n"
                )
                f.write("=" * 80 + "\n")

                f.write(
                    f"VIDEO: {item['video']}\n"
                )

                f.write(
                    f"CLIP: {item['clip_index']}\n"
                )

                f.write(
                    f"INTERVALLO: "
                    f"{item['start']:.2f}s - "
                    f"{item['end']:.2f}s\n\n"
                )

                f.write("GROUND TRUTH\n")
                f.write("-" * 80 + "\n")

                f.write(
                    f"Danger Level: "
                    f"{item['ground_truth']} - "
                    f"{item['ground_truth_label']}\n"
                )

                f.write(
                    "Descrizione GT: "
                    f"{item.get('ground_truth_description') or 'Non disponibile.'}"
                    "\n\n"
                )

                f.write("REPORT HOLMES-VAU\n")
                f.write("-" * 80 + "\n")

                f.write(
                    (
                            item.get("holmes_report")
                            or "Non disponibile"
                    ).strip()
                    + "\n\n"
                )

                f.write("VALUTAZIONE QWEN\n")
                f.write("-" * 80 + "\n")

                f.write(
                    (
                            item.get("qwen_report")
                            or "Non disponibile"
                    ).strip()
                    + "\n\n"
                )

                f.write("ERRORE\n")
                f.write("-" * 80 + "\n")

                f.write(
                    f"Tipo: "
                    f"{item['error_type']}\n"
                )

                f.write(
                    f"Direzione: "
                    f"{item['error_direction']}\n\n"
                )
    CSV_FIELDS = [
        "video",
        "clip_index",
        "start",
        "end",

        "ground_truth",
        "ground_truth_label",
        "ground_truth_description",

        "predicted",
        "predicted_label",

        "error_type",
        "error_direction",

        "holmes_report",
        "qwen_report"
    ]

    with open(
            misclassifications_csv_path,
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
            writer.writerow(item)
    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        "\n===================================="
    )
    print(
        "LARGE VIDEO EVALUATION - TWO STAGE"
    )
    print(
        "===================================="
    )

    print(
        "Video:",
        video_name
    )

    print(
        "GT Overall:",
        gt_overall
    )

    print(
        "Predicted Overall "
        "(Prompt 13):",
        predicted_overall
    )

    print(
        "Prompt 12 max danger:",
        prompt12_overall
    )

    print(
        "\ny_true:",
        y_true
    )

    print(
        "y_pred:",
        y_pred
    )

    if accuracy is not None:

        print(
            "Clip Accuracy "
            "(Prompt 12):",
            round(
                accuracy * 100,
                2
            ),
            "%"
        )

    print(
        "Valid predictions:",
        len(y_pred),
        "/",
        len(
            ground_truth["clips"]
        )
    )

    print(
        "\nComparison saved to:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    video_files = [
        filename
        for filename in os.listdir(VIDEO_DIR)
        if filename.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )
    ]

    print(
        "Video trovati:",
        len(video_files)
    )

    errors = 0
    processed = 0

    # il runner si ferma esattamente dopo
    # la fase di evaluation invece di proseguire se ci sono errori
    for video_name in video_files:

        try:

            print(
                "\n" + "=" * 80
            )

            print(
                "VALUTAZIONE:",
                video_name
            )

            evaluate(
                video_name
            )

            processed += 1

        except Exception as error:

            errors += 1

            print(
                f"Errore su {video_name}: "
                f"{error}"
            )

    print("\n" + "=" * 80)
    print("RIEPILOGO EVALUATION")
    print("=" * 80)

    print(
        "Video valutati correttamente:",
        processed
    )

    print(
        "Errori:",
        errors
    )

    if errors > 0:
        raise RuntimeError(
            f"Evaluation fallita per {errors} video."
        )
