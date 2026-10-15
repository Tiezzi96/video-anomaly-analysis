import os
import sys

# Gestisce anche le copie temporanee create dal runner nella root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "inference"
    else SCRIPT_DIR
)

# Permette di trovare holmesvau e prompts anche avviando
# direttamente lo script contenuto nel package.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import shutil
import subprocess
import torch
from decord import VideoReader, cpu
from holmesvau.holmesvau_utils import load_model, generate
import ollama
from utils.prompts import get_prompt
import gc


OLLAMA_MODEL = "qwen3.5:9b"
# OLLAMA_MODEL = "llama3.1:8b"
TEMPERATURE = 0.0
FFMPEG_PATH = r"C:\Users\Bernardo\ffmpeg\ffmpeg-8.1.2-essentials_build\bin\ffmpeg.exe"

PROMPT_MODEL = "qwen"
PROMPT_STAGE1_ID = 12
PROMPT_STAGE2_ID = 13

device = torch.device("cuda:0")
mllm_path = os.path.join(
    PROJECT_ROOT, "ckpts", "HolmesVAU-2B"
)

sampler_path = os.path.join(
    PROJECT_ROOT, "holmesvau", "ATS", "anomaly_scorer.pth"
)

video_dir = os.path.join(
    PROJECT_ROOT, "dataset classificato"
)

temporary_clips_dir = os.path.join(
    PROJECT_ROOT, "temporary_clips_large_videos"
)
select_frames = 12
dense_sample_freq = 16
seconds_per_selected_frame = 2.0
clip_duration = select_frames * seconds_per_selected_frame
overlap_seconds = 4.0
keep_temporary_clips = False
'''
holmes_prompt = (
    "You are a professional video surveillance analysis software for company security. "
    "Provide a concise security report by summarizing the video and describing any anomaly events observed in it. "
    "Describe the scene, main subjects, actions, visible objects, movements, interactions, and important changes over time. "
    "Focus on safety-relevant or unusual events, including dangerous objects, blocked access or exits,"
    " fire or smoke, unusual people or animals, abnormal behavior, sudden environmental changes, people or objects in unexpected places,"
    " and situations that may create risk for people or property. "
    "If an anomaly event is visible, explain what happens, where it happens, who or what is involved, when it occurs,"
    " and what visual evidence supports considering it anomalous. "
    "If no safety-relevant anomaly is visible, briefly describe the visible scene and state that no relevant anomaly is observed. "
    "Do not call the scene normal or mundane if any unusual or safety-relevant event is visible. "
    "Do not repeat sentences."
)
'''
holmes_prompt = (
    "You are a professional video surveillance analysis software for company security. "
    "Provide a detailed security report describing the full visible scene and any anomaly events observed in the video. "
    "Describe the environment or setting, the main scene, all relevant people, their positions, movements, actions, interactions, "
    "and any important changes over time. "
    "Describe visible objects, their locations, their apparent use, and any changes in their position or state. "
    "Include relevant details about entrances, exits, doors, corridors, vehicles, furniture, equipment, containers, tools, obstacles, "
    "and other scene elements when visible. "
    "Describe how the scene evolves over time, including what is visible at the beginning, what changes during the clip, "
    "and what is visible at the end. "
    "Focus especially on safety-relevant or unusual events, including dangerous objects, blocked access or exits, "
    "fire or smoke, unusual people or animals, abnormal behavior, sudden environmental changes, people falling or collapsing, "
    "physical conflicts, suspicious actions, people or objects in unexpected places, and situations that may create risk for people or property. "
    "If an anomaly event is visible, explain in detail what happens, where it happens, who or what is involved, "
    "how the event develops over time, and what visible evidence supports considering it anomalous. "
    "If multiple anomaly events are visible, describe each one separately and in temporal order. "
    "If no safety-relevant anomaly is visible, still describe the visible scene, people, objects, actions, movements, "
    "and interactions in sufficient detail, then state that no relevant anomaly is observed. "
    "Do not call the scene normal or mundane if any unusual or safety-relevant event is visible. "
    "Do not infer hidden causes, intentions, identities, or events that are not directly supported by the visible content. "
    "Describe each event only once. "
    "Group visually similar consecutive moments into a single description instead of repeating the same observation. "
    "Do not repeatedly describe the same people, actions, or scene state across similar frames. "
    "When all relevant scene information and events have been described, end the report. "
    "Do not repeat sentences."
)


def check_ffmpeg():
    subprocess.run(
        [FFMPEG_PATH, "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def get_video_info(video_path):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps())
    if fps <= 0:
        raise ValueError(f"FPS non validi per: {video_path}")
    duration = total_frames / fps
    return total_frames, fps, duration


def is_large_video(video_path):
    total_frames, fps, duration = get_video_info(video_path)
    seconds_per_frame = duration / select_frames
    return {
        "split_required": seconds_per_frame > seconds_per_selected_frame,
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "seconds_per_selected_frame": seconds_per_frame,
    }


def split_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _, _, duration = get_video_info(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    step = clip_duration - overlap_seconds
    clips = []
    clip_index = 0
    start_time = 0.0

    while start_time < duration:
        end_time = min(start_time + clip_duration, duration)
        current_duration = end_time - start_time
        if current_duration < 1.0 and clips:
            break

        clip_path = os.path.join(output_dir, f"{video_name}_clip_{clip_index:04d}.mp4")
        command = [
            FFMPEG_PATH,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(current_duration),
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            clip_path,
        ]
        subprocess.run(command, check=True)

        clips.append({
            "clip_index": clip_index,
            "path": clip_path,
            "start_time": start_time,
            "end_time": end_time,
        })

        if end_time >= duration:
            break

        start_time += step
        clip_index += 1

    return clips

'''
def analyze_large_video_with_llm(clip_reports):
    system_prompt = """
You are the second-level language model in a video surveillance pipeline.

The first model, Holmes-VAU, analyzed consecutive and partially overlapping
clips of the same video.

For each clip, you receive:
- the global start time;
- the global end time;
- the Holmes-VAU textual report.

Your tasks:
- review Holmes-VAU's observations and judgment;
- reconstruct the complete video timeline;
- identify safety-relevant or contextual anomalies;
- estimate the approximate global start and end time of each anomaly;
- merge duplicate descriptions caused by overlapping clips.

Rules:
- Use only people, objects, actions, events, and locations explicitly described by Holmes-VAU.
- Do not invent events or exact timestamps.
- Do not automatically trust statements such as "normal", "ordinary", "common practice", or "no anomaly".
- Evaluate the described observations independently.
- Judge people, objects, animals, and actions according to the environment explicitly described by Holmes-VAU.
- If an anomaly appears in one clip only, use that clip interval as its approximate time range.
- If the same anomaly appears in consecutive overlapping clips, merge their intervals.
- Always provide a final decision.

Return exactly:
Holmes judgment: CORRECT or INCORRECT
Classification: NORMAL or ANOMALOUS
Anomaly type: type or NONE
Approximate start: seconds or NONE
Approximate end: seconds or NONE
Reason: one concise sentence
""".strip()

    reports_text = "\n\n".join(
        (
            f"CLIP {report['clip_index']}\n"
            f"GLOBAL START: {report['start_time']:.2f} seconds\n"
            f"GLOBAL END: {report['end_time']:.2f} seconds\n"
            f"HOLMES-VAU REPORT: {report['prediction']}"
        )
        for report in clip_reports
    )

    user_prompt = f"""
The following reports were produced by Holmes-VAU from consecutive clips of the same video:

{reports_text}

Review the complete timeline.

If an anomaly is present:
- identify the clip or clips in which it appears;
- estimate its approximate global start and end time;
- merge overlapping clips that describe the same event.

Do not use the video filename as evidence.
""".strip()

    chat_args = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": TEMPERATURE},
        "keep_alive": "30m",
    }

    if OLLAMA_MODEL.startswith("qwen"):
        chat_args["think"] = False

    response = ollama.chat(**chat_args)
    return response["message"].get("content", "").strip()
'''

def build_reports_text(clip_reports):
    """
    Costruisce il testo cronologico dei report Holmes-VAU.
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
    """
    Esegue una singola chiamata al secondo LLM.
    """

    print("\n" + "=" * 70)
    print(stage_name)
    print("=" * 70)
    print("Modello:", OLLAMA_MODEL)
    print("Prompt model:", PROMPT_MODEL)
    print("Prompt ID:", prompt_id)

    chat_args = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": 8192,
            "num_predict": 2048
        },
        "keep_alive": "30m"
    }

    if OLLAMA_MODEL.startswith("qwen"):
        chat_args["think"] = False

    response = ollama.chat(**chat_args)

    content = response["message"].get(
        "content",
        ""
    ).strip()

    if not content:
        raise RuntimeError(
            f"{stage_name}: il secondo modello ha restituito "
            "una risposta vuota."
        )

    return content


def analyze_large_video_with_llm_two_stage(clip_reports):
    """
    Esegue due chiamate al secondo LLM:
    1) Prompt 12: valutazione clip-level;
    2) Prompt 13: ricostruzione temporale usando Holmes + output Prompt 12.
    """

    reports_text = build_reports_text(
        clip_reports
    )

    system_prompt_12, user_prompt_12 = get_prompt(
        model=PROMPT_MODEL,
        prompt_id=PROMPT_STAGE1_ID,
        reports_text=reports_text
    )

    clip_evaluations = run_second_llm(
        system_prompt_12,
        user_prompt_12,
        PROMPT_STAGE1_ID,
        "SECONDO LLM - STAGE 1 - CLIP EVALUATION"
    )

    system_prompt_13, user_prompt_13 = get_prompt(
        model=PROMPT_MODEL,
        prompt_id=PROMPT_STAGE2_ID,
        reports_text=reports_text,
        clip_evaluations=clip_evaluations
    )

    final_report = run_second_llm(
        system_prompt_13,
        user_prompt_13,
        PROMPT_STAGE2_ID,
        "SECONDO LLM - STAGE 2 - TEMPORAL RECONSTRUCTION"
    )

    return clip_evaluations, final_report


model, tokenizer, generation_config, sampler = load_model(
    mllm_path,
    sampler_path,
    device,
)

# aggiunto per ridurre la ripetizione e i loop all'interno di holmes report
generation_config["repetition_penalty"] = 1.1


video_files = [
    os.path.join(video_dir, filename)
    for filename in os.listdir(video_dir)
    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]

check_ffmpeg()

with open(
    os.path.join(PROJECT_ROOT, "results_large_videos_prompt_12_13.txt"),
    "w",
    encoding="utf-8",
) as out, open(
    os.path.join(PROJECT_ROOT, "holmes_reports_large_prompt_12_13.txt"),
    "w",
    encoding="utf-8",
) as chat_out, open(
    os.path.join(PROJECT_ROOT, "clip_evaluations_large_prompt_12_13.txt"),
    "w",
    encoding="utf-8",
) as clip_eval_out:
    for video_id, video_path in enumerate(video_files, start=1):
        video_temp_dir = None

        try:
            video_info = is_large_video(video_path)

            if not video_info["split_required"]:
                print(
                    f"\nSKIPPED: {video_path} "
                    f"({video_info['duration']:.2f}s, non è un video grande)"
                )
                continue

            print(f"\n=== LARGE VIDEO: {video_path} ===")
            print("Total frames:", video_info["total_frames"])
            print("FPS:", video_info["fps"])
            print("Duration:", round(video_info["duration"], 2), "seconds")
            print(
                "Seconds represented by each selected frame:",
                round(video_info["seconds_per_selected_frame"], 2),
            )

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_temp_dir = os.path.join(temporary_clips_dir, video_name)
            clips = split_video(video_path, video_temp_dir)

            print("Number of clips:", len(clips))
            clip_reports = []

            chat_out.write(f"LONG VIDEO ID: {video_id}\n")
            chat_out.write(f"PROMPT STAGE 1 ID: {PROMPT_STAGE1_ID}\n")
            chat_out.write(f"PROMPT STAGE 2 ID: {PROMPT_STAGE2_ID}\n")
            chat_out.write(f"NUMBER OF CLIPS: {len(clips)}\n\n")

            for clip in clips:
                print(
                    f"\n--- Clip {clip['clip_index']}: "
                    f"{clip['start_time']:.2f}s - {clip['end_time']:.2f}s ---"
                )

                pred, history, frame_indices, anomaly_score = generate(
                    clip["path"],
                    holmes_prompt,
                    model,
                    tokenizer,
                    generation_config,
                    sampler,
                    dense_sample_freq=dense_sample_freq,
                    select_frames=select_frames,
                    use_ATS=True,
                )

                clip_report = {
                    "clip_index": clip["clip_index"],
                    "start_time": clip["start_time"],
                    "end_time": clip["end_time"],
                    "prediction": pred,
                    "frame_indices": frame_indices,
                }
                clip_reports.append(clip_report)

                print("HolmesVAU:", pred)

                chat_out.write(f"CLIP {clip['clip_index']}\n")
                chat_out.write(
                    f"START: {clip['start_time']:.2f} seconds\n"
                )
                chat_out.write(
                    f"END: {clip['end_time']:.2f} seconds\n"
                )
                chat_out.write("HOLMES-VAU REPORT:\n")
                chat_out.write(pred.strip() + "\n\n")

                out.write(f"VIDEO: {video_path}\n")
                out.write(
                    f"CLIP {clip['clip_index']}: "
                    f"{clip['start_time']:.2f}s - {clip['end_time']:.2f}s\n"
                )
                out.write(f"HOLMES: {pred}\n")
                out.write(f"FRAMES: {frame_indices}\n")
                out.write("\n" + "-" * 80 + "\n\n")

            chat_out.write("=" * 80 + "\n\n")
            
            clip_evaluations, final_report = (
                analyze_large_video_with_llm_two_stage(
                    clip_reports
                )
            )

            print(
                f"\n{OLLAMA_MODEL} STAGE 1 CLIP EVALUATIONS:"
            )
            print(clip_evaluations)

            print(
                f"\n{OLLAMA_MODEL} FINAL TIMELINE REPORT:"
            )
            print(final_report)

            clip_eval_out.write(
                f"VIDEO: {video_path}\n"
            )
            clip_eval_out.write(
                f"MODEL: {OLLAMA_MODEL}\n"
            )
            clip_eval_out.write(
                f"PROMPT STAGE 1 ID: {PROMPT_STAGE1_ID}\n"
            )
            clip_eval_out.write(
                "CLIP EVALUATIONS:\n"
            )
            clip_eval_out.write(
                clip_evaluations + "\n"
            )
            clip_eval_out.write(
                "\n" + "=" * 80 + "\n\n"
            )

            out.write(f"VIDEO: {video_path}\n")
            out.write(f"MODEL: {OLLAMA_MODEL}\n")
            out.write(
                f"PROMPT STAGE 1 ID: "
                f"{PROMPT_STAGE1_ID}\n"
            )
            out.write(
                f"PROMPT STAGE 2 ID: "
                f"{PROMPT_STAGE2_ID}\n"
            )
            out.write(
                "STAGE 1 CLIP EVALUATIONS:\n"
            )
            out.write(
                clip_evaluations + "\n\n"
            )
            out.write(
                f"FINAL REPORT:\n{final_report}\n"
            )
            out.write(
                "\n" + "=" * 80 + "\n\n"
            )

        except Exception as error:
            print(f"Errore su {video_path}: {error}")
            out.write(f"VIDEO: {video_path}\n")
            out.write(f"ERRORE: {error}\n")
            out.write("\n" + "=" * 80 + "\n\n")

        finally:
            if (
                video_temp_dir is not None
                and os.path.exists(video_temp_dir)
                and not keep_temporary_clips
            ):
                shutil.rmtree(video_temp_dir)
                print("Clip temporanee eliminate.")

del history

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()