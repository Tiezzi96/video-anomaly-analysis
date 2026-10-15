"""
inference_virtual_composite.py

Analizza una sequenza di video AI come se fossero clip consecutive
di un unico video, senza creare fisicamente un file composite.

Ogni video AI viene:
1. analizzato direttamente da Holmes-VAU;
2. associato a START/END globali cumulativi;
3. inserito nel manifest virtuale;
4. trattato dal secondo LLM insieme agli altri report Holmes.

Il formato di manifest/evaluation rimane compatibile con
evaluate_composite.py e confusion_matrix.py.
"""

import os
import sys

# Gestisce anche le copie temporanee create dai runner nella root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "inference"
    else SCRIPT_DIR
)

# Rende accessibili holmesvau e prompts.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import torch
import ollama
import re
import cv2
import gc

from holmesvau.holmesvau_utils import load_model, generate
from utils.prompts import get_prompt


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Cartella contenente i video AI originali.
# MODIFICA questo path in base alla tua struttura locale.
AI_VIDEO_DIR = os.path.join(
    PROJECT_ROOT, "Video Generati AI"
)
# Cartella usata per manifest e risultati.
# Non viene creato alcun video composite fisico.
COMPOSITE_DIR = os.path.join(
    PROJECT_ROOT, "composite_video_output"
)
VIRTUAL_DIR = os.path.join(
    COMPOSITE_DIR,
    "virtual"
)

# Cartella contenente i file TXT che definiscono le sequenze.
SEQUENCES_DIR = os.path.join(
    PROJECT_ROOT, "sequenze AI"
)
# "01" -> ./sequenze/sequence_01.txt
SEQUENCE_ID = "10"

# L'ID dell'esperimento coincide con quello della sequenza.
VIDEO_ID = SEQUENCE_ID

SEQUENCE_PATH = os.path.join(
    SEQUENCES_DIR,
    f"video_ai_sequence_{SEQUENCE_ID}.txt"
)

# ----------------------------
# Secondo LLM
# ----------------------------

OLLAMA_MODEL = "qwen3.5:9b"
PROMPT_MODEL = "qwen"
PROMPT_STAGE1_ID = 10
PROMPT_STAGE2_ID = 11
TEMPERATURE = 0.0

VALID_DANGER_LEVELS = {0, 1, 2}

# Per usare Llama:
# OLLAMA_MODEL = "llama3.1:8b"
# PROMPT_MODEL = "llama"
# PROMPT_ID = 2

# ----------------------------
# Manifest del composite virtuale
# ----------------------------

MANIFEST_PATH = os.path.join(
    VIRTUAL_DIR,
    f"composite_ai_video_{VIDEO_ID}_manifest.json"
)

# ----------------------------
# Cartella risultati
# ----------------------------

RESULTS_DIR = os.path.join(
    VIRTUAL_DIR,
    "results",
    PROMPT_MODEL,
    f"prompt_{PROMPT_STAGE1_ID}_{PROMPT_STAGE2_ID}"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_results.txt"
)

HOLMES_EXPORT_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_holmes_reports.txt"
)

EVALUATION_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_evaluation.json"
)

CLIP_EVALUATIONS_PATH = os.path.join(
    RESULTS_DIR,
    f"composite_ai_video_{VIDEO_ID}_clip_evaluations.txt"
)

# ----------------------------
# Holmes-VAU
# ----------------------------

mllm_path = os.path.join(
    PROJECT_ROOT, "ckpts", "HolmesVAU-2B"
)

sampler_path = os.path.join(
    PROJECT_ROOT, "holmesvau", "ATS", "anomaly_scorer.pth"
)
device = torch.device("cuda:0")

select_frames = 12
dense_sample_freq = 16


# ============================================================
# PROMPT HOLMES-VAU
# ============================================================
'''
holmes_prompt = (
    "You are a professional video surveillance analysis software for company security. "
    "Provide a concise security report by summarizing the video and describing any anomaly events observed in it. "
    "Describe the scene, main subjects, actions, visible objects, movements, interactions, and important changes over time. "
    "Focus on safety-relevant or unusual events, including dangerous objects, blocked access or exits, "
    "fire or smoke, unusual people or animals, abnormal behavior, sudden environmental changes, "
    "people or objects in unexpected places, and situations that may create risk for people or property. "
    "If an anomaly event is visible, explain what happens, where it happens, who or what is involved, "
    "and what visual evidence supports considering it anomalous. "
    "If no safety-relevant anomaly is visible, briefly describe the visible scene and state that no relevant anomaly is observed. "
    "Do not call the scene normal or mundane if any unusual or safety-relevant event is visible. "
    "Do not repeat sentences."
)
'''
## AMBIENTE OPSEDALIERO
'''
holmes_prompt = (
    "You are a professional video surveillance analysis software for company security. "
    "The video takes place in a hospital environment. "
    "Provide a concise security report by summarizing the video and describing any anomaly events observed in it. "
    "Describe the scene, main subjects, actions, visible objects, movements, interactions, and important changes over time. "
    "Focus on safety-relevant or unusual events, including dangerous objects, blocked access or exits, "
    "fire or smoke, unusual people or animals, abnormal behavior, sudden environmental changes, "
    "people or objects in unexpected places, and situations that may create risk for people or property. "
    "If an anomaly event is visible, explain what happens, where it happens, who or what is involved, "
    "and what visual evidence supports considering it anomalous. "
    "If no safety-relevant anomaly is visible, briefly describe the visible scene and state that no relevant anomaly is observed. "
    "Do not call the scene normal or mundane if any unusual or safety-relevant event is visible. "
    "Do not repeat sentences."
)
'''

holmes_prompt = (
    "You are a professional video surveillance analysis software for company security. "
    "The video takes place in a hospital environment. "
    "Provide a detailed security report by describing the full visible scene and any anomaly events observed in it. "
    "Describe the environment, room or area layout when visible, main subjects, all relevant people, "
    "their positions, movements, actions, interactions, and any important changes over time. "
    "Describe visible objects, their locations, their apparent use, and any changes in their position or state. "
    "Include relevant details about entrances, exits, doors, corridors, beds, equipment, furniture, "
    "containers, tools, obstacles, and other scene elements when visible. "
    "Describe how the scene evolves over time, including what is visible at the beginning, "
    "what changes during the clip, and what is visible at the end. "
    "Focus especially on safety-relevant or unusual events, including dangerous objects, blocked access or exits, "
    "fire or smoke, unusual people or animals, abnormal behavior, sudden environmental changes, "
    "people falling or collapsing, physical conflicts, suspicious actions, "
    "people or objects in unexpected places, and situations that may create risk for people or property. "
    "If an anomaly event is visible, explain in detail what happens, where it happens, "
    "who or what is involved, how the event develops over time, and what visible evidence supports considering it anomalous. "
    "If multiple anomaly events are visible, describe each one separately and in temporal order. "
    "If no safety-relevant anomaly is visible, still describe the visible scene, people, objects, actions, "
    "movements, and interactions in sufficient detail, then state that no relevant anomaly is observed. "
    "Do not call the scene normal or mundane if any unusual or safety-relevant event is visible. "
    "Do not infer hidden causes, intentions, identities, or events that are not directly supported by the visible content. "
    "Describe each event only once. "
    "If the same action or scene state appears repeatedly across multiple sampled frames, summarize it once rather than describing each occurrence separately. "
    "Group visually similar consecutive moments into a single description instead of repeating the same observation. "
    "Do not repeatedly describe the same people, actions, objects, or scene state across similar frames. "
    "When all relevant scene information and events have been described, end the report. "
    "Do not repeat sentences."
)


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================


def load_video_sequence(sequence_path):
    """Carica un nome video per riga dal file TXT della sequenza."""
    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"File sequenza non trovato:\n{sequence_path}")

    with open(sequence_path, "r", encoding="utf-8-sig") as file:
        sequence = [
            line.strip()
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]

    if not sequence:
        raise ValueError(f"La sequenza è vuota:\n{sequence_path}")

    return sequence


def calculate_overall_danger_score(final_report):
    """
    Calculates a deterministic Overall Danger Score from 0 to 100
    using the Danger Levels and durations of the Timeline phases.

    Formula:
        75% maximum danger reached
        25% duration-weighted danger exposure
    """

    # --------------------------------------------------
    # Extract Timeline only
    # --------------------------------------------------

    timeline_match = re.search(
        r"Timeline:(.*?)(?:Anomaly Events:|$)",
        final_report,
        re.DOTALL | re.IGNORECASE
    )

    if not timeline_match:
        return None

    timeline_text = timeline_match.group(1)

    # --------------------------------------------------
    # Extract phases
    # --------------------------------------------------

    phase_pattern = re.compile(
        r"Phase\s+\d+.*?"
        r"Start:\s*([\d.]+)\s*(?:seconds?)?.*?"
        r"End:\s*([\d.]+)\s*(?:seconds?)?.*?"
        r"Danger Level:\s*(\d)",
        re.DOTALL | re.IGNORECASE
    )

    phases = []

    for match in phase_pattern.finditer(timeline_text):

        start = float(match.group(1))
        end = float(match.group(2))
        danger = int(match.group(3))

        # if end > start and 0 <= danger <= 4:
        if end > start and danger in VALID_DANGER_LEVELS:
            phases.append({
                "start": start,
                "end": end,
                "danger": danger
            })

    if not phases:
        return None

    # --------------------------------------------------
    # Video duration
    # --------------------------------------------------

    video_start = min(p["start"] for p in phases)
    video_end = max(p["end"] for p in phases)

    total_duration = video_end - video_start

    if total_duration <= 0:
        return None

    # --------------------------------------------------
    # COMPONENT 1 — Maximum danger (70 points)
    # --------------------------------------------------

    max_danger = max(p["danger"] for p in phases)

    # max_component = (max_danger / 4.0) * 70.0
    max_component = (max_danger / 2.0) * 70.0

    # --------------------------------------------------
    # COMPONENT 2 — Temporal exposure (20 points)
    # --------------------------------------------------

    weighted_exposure = 0.0

    for phase in phases:

        duration = phase["end"] - phase["start"]

        weighted_exposure += (
                duration * (phase["danger"] / 2.0)
        )

    exposure_ratio = weighted_exposure / total_duration

    exposure_component = exposure_ratio * 20.0

    # --------------------------------------------------
    # COMPONENT 3 — Number of anomaly events (10 points)
    # --------------------------------------------------

    anomaly_section = re.search(
        r"Anomaly Events:(.*)$",
        final_report,
        re.DOTALL | re.IGNORECASE
    )

    num_anomalies = 0

    if anomaly_section:

        anomaly_text = anomaly_section.group(1)

        num_anomalies = len(
            re.findall(
                r"(?m)^Anomaly(?:\s+\d+)?\s*$",
                anomaly_text,
                re.IGNORECASE
            )
        )

    recurrence_component = min(
        num_anomalies / 5.0,
        1.0
    ) * 10.0
    # --------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------

    # severity_score = (max_danger / 4.0) * 100
    severity_score = (max_danger / 2.0) * 100

    exposure_score = (
        weighted_exposure / total_duration
    ) * 100

    score = round(
        0.75 * severity_score
        + 0.25 * exposure_score
    )

    return max(0, min(100, score))
'''
def get_danger_score_label(score):

    if score is None:
        return "NOT AVAILABLE"

    if score <= 20:
        return "VERY LOW"

    elif score <= 40:
        return "LOW"

    elif score <= 60:
        return "MODERATE"

    elif score <= 80:
        return "HIGH"

    else:
        return "CRITICAL"
'''
def extract_timeline_phases(final_report):
    """
    Extract structured timeline phases from the second LLM final report.
    """

    timeline_match = re.search(
        r"Timeline:(.*?)(?:Anomaly Events:|Overall Danger Score:|$)",
        final_report,
        re.DOTALL | re.IGNORECASE
    )

    if not timeline_match:
        return []

    timeline_text = timeline_match.group(1)

    phase_pattern = re.compile(
        r"Phase\s+\d+.*?"
        r"Start:\s*([\d.]+)\s*(?:seconds?)?.*?"
        r"End:\s*([\d.]+)\s*(?:seconds?)?.*?"
        r"Assessment:\s*(NORMAL|ANOMALOUS).*?"
        r"Danger Level:\s*(\d)",
        re.DOTALL | re.IGNORECASE
    )

    phases = []

    for phase_index, match in enumerate(
        phase_pattern.finditer(timeline_text),
        start=1
    ):
        danger_level = int(match.group(4))

        if danger_level not in VALID_DANGER_LEVELS:
            raise ValueError(
                f"Danger Level non valido nella fase "
                f"{phase_index}: {danger_level}"
            )

        phases.append({
            "phase": phase_index,
            "start_time": float(match.group(1)),
            "end_time": float(match.group(2)),
            "assessment": match.group(3).upper(),
            "danger_level": danger_level
        })

    return phases

def get_video_duration(video_path):
    """
    Restituisce la durata reale del video in secondi.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Impossibile aprire il video:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0:
        raise RuntimeError(
            f"FPS non validi per:\n{video_path}"
        )

    return frame_count / fps


def build_virtual_manifest():
    """
    Costruisce il manifest di un composite virtuale.

    I video originali NON vengono concatenati fisicamente.
    Ogni elemento di VIDEO_SEQUENCE viene trattato come una clip
    consecutiva dello stesso video logico.
    """
    segments = []
    current_time = 0.0

    for segment_index, video_name in enumerate(VIDEO_SEQUENCE):

        video_path = os.path.join(
            AI_VIDEO_DIR,
            video_name
        )

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video AI non trovato:\n{video_path}"
            )

        duration = get_video_duration(video_path)

        start_time = current_time
        end_time = start_time + duration

        segments.append({
            "segment_index": segment_index,

            # Mantenuto per compatibilità con evaluate_composite.py.
            "segment_file": video_name,

            # Nome del video AI originale, usato dalla Ground Truth.
            "source_file": video_name,

            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        })

        current_time = end_time

    manifest = {
        "video_id": VIDEO_ID,
        "composite_type": "virtual",
        "number_of_segments": len(segments),
        "total_duration": current_time,
        "segments": segments
    }

    # os.makedirs(COMPOSITE_DIR, exist_ok=True)
    os.makedirs(VIRTUAL_DIR, exist_ok=True)

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False
        )

    return manifest


def build_reports_text(clip_reports):
    """
    Costruisce il testo cronologico da passare al secondo LLM.
    """
    return "\n\n".join(
        (
            f"CLIP {report['clip_index']}\n"
            f"START: {report['start_time']:.2f} seconds\n"
            f"END: {report['end_time']:.2f} seconds\n"
            f"HOLMES REPORT: {report['prediction']}"
        )
        for report in clip_reports
    )


def run_second_llm(system_prompt, user_prompt, prompt_id, stage_name):
    """Esegue una singola chiamata al secondo LLM."""

    print("\n" + "=" * 70)
    print(stage_name)
    print("=" * 70)
    print("Modello:", OLLAMA_MODEL)
    print("Prompt model:", PROMPT_MODEL)
    print("Prompt ID:", prompt_id)

    chat_args = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": 8192,
            "num_predict": 2048
        },
        "keep_alive": "30m"
    }

    if PROMPT_MODEL.lower() == "qwen":
        chat_args["think"] = False

    response = ollama.chat(**chat_args)
    content = response["message"].get("content", "").strip()

    print("\n--- OLLAMA DEBUG ---")
    print("done:", response.get("done"))
    print("done_reason:", response.get("done_reason"))
    print("prompt_eval_count:", response.get("prompt_eval_count"))
    print("eval_count:", response.get("eval_count"))
    print("content length:", len(content))
    print("--------------------")

    if not content:
        raise RuntimeError(f"{stage_name}: il secondo modello ha restituito una risposta vuota.")

    return content


def analyze_with_second_llm_two_stage(clip_reports):
    """
    Esegue due chiamate al secondo LLM:
    1) Prompt 10: valutazione clip-level;
    2) Prompt 11: ricostruzione temporale usando Holmes + output Prompt 10.
    """

    reports_text = build_reports_text(clip_reports)

    system_prompt_10, user_prompt_10 = get_prompt(
        model=PROMPT_MODEL,
        prompt_id=PROMPT_STAGE1_ID,
        reports_text=reports_text
    )

    clip_evaluations = run_second_llm(
        system_prompt_10,
        user_prompt_10,
        PROMPT_STAGE1_ID,
        "SECONDO LLM - STAGE 1 - CLIP EVALUATION"
    )

    system_prompt_11, user_prompt_11 = get_prompt(
        model=PROMPT_MODEL,
        prompt_id=PROMPT_STAGE2_ID,
        reports_text=reports_text,
        clip_evaluations=clip_evaluations
    )

    final_report = run_second_llm(
        system_prompt_11,
        user_prompt_11,
        PROMPT_STAGE2_ID,
        "SECONDO LLM - STAGE 2 - TEMPORAL RECONSTRUCTION"
    )

    return clip_evaluations, final_report

def export_holmes_reports(clip_reports):
    """
    Salva un TXT neutro con le sole analisi Holmes-VAU,
    da poter passare anche a ChatGPT o Gemini.
    Non include i nomi dei file originali.
    """

    with open(
        HOLMES_EXPORT_PATH,
        "w",
        encoding="utf-8"
    ) as out:

        out.write("VIRTUAL COMPOSITE VIDEO REPORT\n\n")

        for report in clip_reports:
            out.write(
                f"CLIP {report['clip_index']}\n"
            )
            out.write(
                f"START: {report['start_time']:.2f} seconds\n"
            )
            out.write(
                f"END: {report['end_time']:.2f} seconds\n"
            )
            out.write(
                "HOLMES-VAU REPORT:\n"
            )
            out.write(
                report["prediction"].strip() + "\n\n"
            )

        out.write("=" * 80 + "\n")


# ============================================================
# CARICAMENTO HOLMES-VAU
# ============================================================

print("\nCaricamento Holmes-VAU...")

model, tokenizer, generation_config, sampler = load_model(
    mllm_path,
    sampler_path,
    device
)
# aggiunto per ridurre la ripetizione e i loop all'interno di holmes report
generation_config["repetition_penalty"] = 1.1

print("Holmes-VAU caricato.")


# ============================================================
# CARICAMENTO SEQUENZA E COSTRUZIONE COMPOSITE VIRTUALE
# ============================================================

print("\nCaricamento sequenza...")
print("Sequence ID:", SEQUENCE_ID)
print("Sequence file:", SEQUENCE_PATH)

VIDEO_SEQUENCE = load_video_sequence(SEQUENCE_PATH)
print("Video caricati:", len(VIDEO_SEQUENCE))

manifest = build_virtual_manifest()
segments = manifest["segments"]

print("\n" + "=" * 70)
print("VIRTUAL COMPOSITE")
print("=" * 70)

print(
    "Numero segmenti:",
    manifest["number_of_segments"]
)

print(
    "Durata totale:",
    f"{manifest['total_duration']:.3f}",
    "secondi"
)

print("\nSequenza:")

for segment in segments:
    print(
        f"CLIP {segment['segment_index']} | "
        f"{segment['source_file']} | "
        f"{segment['start_time']:.3f} - "
        f"{segment['end_time']:.3f}s"
    )


# ============================================================
# ANALISI DI OGNI SEGMENTO CON HOLMES-VAU
# ============================================================

clip_reports = []

for segment in segments:

    clip_index = segment["segment_index"]
    start_time = float(segment["start_time"])
    end_time = float(segment["end_time"])

    video_path = os.path.join(
        AI_VIDEO_DIR,
        segment["source_file"]
    )

    print("\n" + "-" * 70)
    print(
        f"CLIP {clip_index}: "
        f"{start_time:.2f}s - {end_time:.2f}s"
    )
    print(
        f"Source: {segment['source_file']}"
    )
    print("-" * 70)

    pred, history, frame_indices, anomaly_score = generate(
        video_path,
        holmes_prompt,
        model,
        tokenizer,
        generation_config,
        sampler,
        dense_sample_freq=dense_sample_freq,
        select_frames=select_frames,
        use_ATS=True
    )

    clip_report = {
        "clip_index": clip_index,
        "start_time": start_time,
        "end_time": end_time,
        "prediction": pred,
        "frame_indices": frame_indices,
        "anomaly_score": anomaly_score
    }

    clip_reports.append(clip_report)

    print("\nHolmesVAU:")
    print(pred)

    print("\nSelected frames:")
    print(frame_indices)


# ============================================================
# ESPORTAZIONE REPORT HOLMES
# ============================================================

export_holmes_reports(clip_reports)

print(
    f"\nReport Holmes esportati in:\n{HOLMES_EXPORT_PATH}"
)


# ============================================================
# ANALISI COMPLETA CON QWEN / LLAMA
# ============================================================

clip_evaluations, final_report = analyze_with_second_llm_two_stage(
    clip_reports
)

with open(CLIP_EVALUATIONS_PATH, "w", encoding="utf-8") as out:
    out.write(clip_evaluations + "\n")

print(f"\nClip-level evaluations salvate in:\n{CLIP_EVALUATIONS_PATH}")

timeline_phases = extract_timeline_phases(
    final_report
)

overall_danger_score = calculate_overall_danger_score(
    final_report
)
'''
danger_score_label = get_danger_score_label(
    overall_danger_score
)
if overall_danger_score is not None:

    score_text = (
        "\n\n"
        "Overall Danger Score:\n"
        f"{overall_danger_score} / 100\n"
        "\n"
        "Danger Score Category:\n"
        f"{danger_score_label}"
    )

else:

    score_text = (
        "\n\n"
        "Overall Danger Score:\n"
        "NOT AVAILABLE"
    )'''
if overall_danger_score is not None:

    score_text = (
        "\n\n"
        "Overall Danger Score:\n"
        f"{overall_danger_score} / 100"
    )

else:

    score_text = (
        "\n\n"
        "Overall Danger Score:\n"
        "NOT AVAILABLE"
    )
final_report_with_score = final_report + score_text

evaluation_data = {
    "video_id": VIDEO_ID,
    "composite_type": "virtual",
    "second_model": OLLAMA_MODEL,
    "prompt_model": PROMPT_MODEL,
    "prompt_stage1_id": PROMPT_STAGE1_ID,
    "prompt_stage2_id": PROMPT_STAGE2_ID,
    "clip_evaluations": clip_evaluations,
    "total_duration": manifest.get("total_duration"),
    "overall_danger_score": overall_danger_score,
    #"danger_score_category": danger_score_label,

    "segments": [],

    "timeline_phases": timeline_phases
}


for segment, report in zip(segments, clip_reports):

    evaluation_data["segments"].append({
        "segment_index": report["clip_index"],
        "segment_file": segment["segment_file"],
        "source_file": segment["source_file"],
        "start_time": report["start_time"],
        "end_time": report["end_time"],
        "holmes_report": report["prediction"],
        "holmes_anomaly_score": (
    report["anomaly_score"].tolist()
    if hasattr(report["anomaly_score"], "tolist")
    else report["anomaly_score"]
)
    })


with open(
    EVALUATION_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_data,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 70)
print(f"{OLLAMA_MODEL} FINAL REPORT")
print("=" * 70)
# print("final report: "+final_report)
print(final_report_with_score)
print(
    f"\nDati evaluation:\n"
    f"{EVALUATION_PATH}"
)

# ============================================================
# SALVATAGGIO RISULTATI COMPLETI
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as out:

    out.write("VIRTUAL COMPOSITE VIDEO ANALYSIS\n")
    out.write("=" * 80 + "\n\n")

    out.write(
        f"TOTAL DURATION: "
        f"{manifest.get('total_duration', 'N/A')} seconds\n"
    )

    out.write(
        f"NUMBER OF SEGMENTS: "
        f"{len(clip_reports)}\n"
    )

    out.write(
        f"SECOND MODEL: {OLLAMA_MODEL}\n"
    )

    out.write(
        f"PROMPT MODEL: {PROMPT_MODEL}\n"
    )

    out.write(
        f"PROMPT STAGE 1 ID: {PROMPT_STAGE1_ID}\n"
        f"PROMPT STAGE 2 ID: {PROMPT_STAGE2_ID}\n\n"
    )

    out.write("=" * 80 + "\n")
    out.write("HOLMES-VAU CLIP REPORTS\n")
    out.write("=" * 80 + "\n\n")

    for report in clip_reports:

        out.write(
            f"CLIP {report['clip_index']}\n"
        )

        out.write(
            f"START: {report['start_time']:.2f} seconds\n"
        )

        out.write(
            f"END: {report['end_time']:.2f} seconds\n"
        )

        out.write(
            f"SELECTED FRAMES: "
            f"{report['frame_indices']}\n"
        )

        out.write(
            "HOLMES-VAU REPORT:\n"
        )

        out.write(
            report["prediction"].strip()
            + "\n"
        )

        out.write(
            "\n" + "-" * 80 + "\n\n"
        )

    out.write("=" * 80 + "\n")
    out.write("SECOND-LEVEL LLM - STAGE 1 CLIP EVALUATIONS\n")
    out.write("=" * 80 + "\n\n")

    out.write(clip_evaluations + "\n")

    out.write("=" * 80 + "\n")
    out.write("SECOND-LEVEL LLM - STAGE 2 FINAL REPORT\n")
    out.write("=" * 80 + "\n\n")

    out.write(final_report + "\n")

    out.write("=" * 80 + "\n")
    out.write("SECOND-LEVEL LLM REPORT WITH SCORE\n")
    out.write("=" * 80 + "\n")

    out.write(final_report_with_score + "\n")


print("\n" + "=" * 70)
print("ANALISI COMPLETATA")
print("=" * 70)

print(
    f"Risultati completi:\n{RESULTS_PATH}"
)

print(
    f"\nReport Holmes per ChatGPT/Gemini:\n"
    f"{HOLMES_EXPORT_PATH}"
)

del history

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()