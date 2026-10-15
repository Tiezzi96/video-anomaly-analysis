import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INFERENCE_SCRIPT = (
    BASE_DIR / "inference" / "inference_large_videos_single_clip_stage1.py"
)

EVALUATE_SCRIPT = (
    BASE_DIR / "evaluate" / "evaluate_large_video_two_stage_all_videos.py"
)

SINGLE_CM_SCRIPT = (
    BASE_DIR
    / "metrics"
    / "confusion_matrix_large_video_all_videos.py"
)

GLOBAL_CM_SCRIPT = (
    BASE_DIR
    / "metrics"
    / "global_confusion_matrix_large_video.py"
)
GLOBAL_ERROR_ANALYSIS_SCRIPT = (
    BASE_DIR / "error_analysis" / "global_error_analysis_large.py"
)
GLOBAL_F1_SCRIPT = (
    BASE_DIR / "metrics" / "global_clip_level_f1_score.py"
)


PROMPT_STAGE1_ID = 15
PROMPT_STAGE2_ID = 13

PROMPT_MODEL = "qwen"

EVALUATION_TYPE = "large_video_dataset"

RUN_GLOBAL_F1 = True

STOP_ON_ERROR = True


# ============================================================
# CONTROLLO FILE
# ============================================================

def check_scripts():

    scripts = [
        INFERENCE_SCRIPT,
        EVALUATE_SCRIPT,
        SINGLE_CM_SCRIPT,
        GLOBAL_CM_SCRIPT,
        GLOBAL_ERROR_ANALYSIS_SCRIPT,
    ]

    if RUN_GLOBAL_F1:
        scripts.append(GLOBAL_F1_SCRIPT)

    missing = [
        str(path)
        for path in scripts
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "File mancanti:\n"
            + "\n".join(missing)
        )


# ============================================================
# SOSTITUZIONE STRINGHE
# ============================================================

def replace_string_assignment(
    source,
    variable_name,
    value
):

    pattern = re.compile(
        rf'^(?P<prefix>\s*{re.escape(variable_name)}\s*=\s*)'
        rf'(?P<quote>["\'])'
        rf'(?P<value>.*?)'
        rf'(?P=quote)\s*$',
        re.MULTILINE,
    )

    replacement = (
        rf'\g<prefix>"{value}"'
    )

    modified, count = pattern.subn(
        replacement,
        source,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            f"Impossibile sostituire "
            f"{variable_name} nello script."
        )

    return modified


# ============================================================
# SOSTITUZIONE INTERI - PRIMA OCCORRENZA
# ============================================================

def replace_int_assignment(
    source,
    variable_name,
    value
):

    pattern = re.compile(
        rf'^(?P<prefix>\s*{re.escape(variable_name)}\s*=\s*)'
        rf'\d+\s*$',
        re.MULTILINE,
    )

    replacement = (
        rf'\g<prefix>{value}'
    )

    modified, count = pattern.subn(
        replacement,
        source,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            f"Impossibile sostituire "
            f"{variable_name} nello script."
        )

    return modified


# ============================================================
# SOSTITUZIONE INTERI - TUTTE LE OCCORRENZE
# ============================================================

def replace_all_int_assignments(
    source,
    variable_name,
    value
):

    pattern = re.compile(
        rf'^(?P<prefix>\s*{re.escape(variable_name)}\s*=\s*)'
        rf'\d+\s*$',
        re.MULTILINE,
    )

    replacement = (
        rf'\g<prefix>{value}'
    )

    modified, count = pattern.subn(
        replacement,
        source,
    )

    if count < 1:
        raise RuntimeError(
            f"Impossibile sostituire "
            f"{variable_name} nello script."
        )

    return modified


# ============================================================
# ESECUZIONE SCRIPT MODIFICATO TEMPORANEAMENTE
# ============================================================

def run_modified_script(
    script_path,
    label,
    string_replacements=None,
    int_replacements=None,
    all_int_replacements=None,
):

    print("\n" + "=" * 90)
    print(label)
    print(f"Script: {script_path.name}")

    if string_replacements:
        for variable, value in string_replacements.items():
            print(f"{variable} = {value}")

    if int_replacements:
        for variable, value in int_replacements.items():
            print(f"{variable} = {value}")

    if all_int_replacements:
        for variable, value in all_int_replacements.items():
            print(f"{variable} = {value}")

    print("=" * 90, flush=True)

    source = script_path.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # STRINGHE
    # --------------------------------------------------------

    if string_replacements:

        for variable_name, value in (
            string_replacements.items()
        ):

            source = replace_string_assignment(
                source,
                variable_name,
                value,
            )

    # --------------------------------------------------------
    # INTERI - PRIMA OCCORRENZA
    # --------------------------------------------------------

    if int_replacements:

        for variable_name, value in (
            int_replacements.items()
        ):

            source = replace_int_assignment(
                source,
                variable_name,
                value,
            )

    # --------------------------------------------------------
    # INTERI - TUTTE LE OCCORRENZE
    # --------------------------------------------------------

    if all_int_replacements:

        for variable_name, value in (
            all_int_replacements.items()
        ):

            source = replace_all_int_assignments(
                source,
                variable_name,
                value,
            )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".py",
            prefix=f"_runner_{script_path.stem}_",
            dir=BASE_DIR,
            delete=False,
        ) as tmp:

            tmp.write(source)

            temp_path = Path(
                tmp.name
            )

        env = os.environ.copy()

        # Evita blocchi causati da plt.show()
        env.setdefault(
            "MPLBACKEND",
            "Agg"
        )

        result = subprocess.run(
            [
                sys.executable,
                str(temp_path)
            ],
            cwd=BASE_DIR,
            env=env,
        )

        if result.returncode != 0:

            raise RuntimeError(
                f"{script_path.name} "
                f"terminato con codice "
                f"{result.returncode}"
            )

    finally:

        if temp_path is not None:

            try:
                temp_path.unlink()

            except FileNotFoundError:
                pass


# ============================================================
# GESTIONE ERRORI
# ============================================================

def handle_error(stage, exc):

    print("\n" + "!" * 90)

    print(
        f"ERRORE | Stage: {stage}"
    )

    print(exc)

    print(
        "!" * 90,
        flush=True
    )

    if STOP_ON_ERROR:
        raise exc


# ============================================================
# MAIN
# ============================================================

def main():

    check_scripts()

    print("\n" + "=" * 90)
    print(
        "LARGE VIDEO - "
        "SINGLE CLIP STAGE 1 - "
        "FULL PIPELINE"
    )
    print("=" * 90)

    print(
        f"Prompt Stage 1: "
        f"{PROMPT_STAGE1_ID}"
    )

    print(
        f"Prompt Stage 2: "
        f"{PROMPT_STAGE2_ID}"
    )

    print(
        f"Evaluation type: "
        f"{EVALUATION_TYPE}"
    )

    print("=" * 90)

    # ========================================================
    # FASE 1 - INFERENCE
    # ========================================================

    print("\n\n" + "#" * 90)
    print(
        "### FASE 1/6 - "
        "INFERENCE LARGE VIDEO SINGLE CLIP"
    )
    print("#" * 90, flush=True)

    try:

        run_modified_script(
            INFERENCE_SCRIPT,
            "INFERENCE LARGE VIDEO SINGLE CLIP",
            string_replacements={
                "PROMPT_MODEL": PROMPT_MODEL,
            },
            int_replacements={
                "PROMPT_STAGE1_ID":
                    PROMPT_STAGE1_ID,
                "PROMPT_STAGE2_ID":
                    PROMPT_STAGE2_ID,
            },
        )

    except Exception as exc:

        handle_error(
            "inference",
            exc
        )

    # ========================================================
    # FASE 2 - EVALUATE
    # ========================================================

    print("\n\n" + "#" * 90)
    print(
        "### FASE 2/6 - "
        "EVALUATE LARGE VIDEO"
    )
    print("#" * 90, flush=True)

    try:

        run_modified_script(
            EVALUATE_SCRIPT,
            "EVALUATE LARGE VIDEO",
            string_replacements={
                "PIPELINE_MODE":
                    "single_clip_stage1",
            },
        )

    except Exception as exc:

        handle_error(
            "evaluate",
            exc
        )

    # ========================================================
    # FASE 3 - CONFUSION MATRIX SINGOLE
    # ========================================================

    print("\n\n" + "#" * 90)
    print(
        "### FASE 3/6 - "
        "CONFUSION MATRIX SINGOLE"
    )
    print("#" * 90, flush=True)

    try:

        run_modified_script(
            SINGLE_CM_SCRIPT,
            "CONFUSION MATRIX SINGOLE",
            int_replacements={
                "PROMPT_STAGE1_ID":
                    PROMPT_STAGE1_ID,
                "PROMPT_STAGE2_ID":
                    PROMPT_STAGE2_ID,
            },
        )

    except Exception as exc:

        handle_error(
            "confusion_matrix_single",
            exc
        )

    # ========================================================
    # FASE 4 - GLOBAL CONFUSION MATRIX
    # ========================================================

    print("\n\n" + "#" * 90)
    print(
        "### FASE 4/6 - "
        "GLOBAL CONFUSION MATRIX"
    )
    print("#" * 90, flush=True)

    try:

        run_modified_script(
            GLOBAL_CM_SCRIPT,
            "GLOBAL CONFUSION MATRIX",
            int_replacements={
                "PROMPT_STAGE1_ID":
                    PROMPT_STAGE1_ID,
                "PROMPT_STAGE2_ID":
                    PROMPT_STAGE2_ID,
            },
        )

    except Exception as exc:

        handle_error(
            "global_confusion_matrix",
            exc
        )

    # ========================================================
    # FASE 5 - GLOBAL ERROR ANALYSIS
    # ========================================================

    print("\n\n" + "#" * 90)
    print(
        "### FASE 5/6 - "
        "GLOBAL ERROR ANALYSIS"
    )
    print("#" * 90, flush=True)

    try:

        run_modified_script(
            GLOBAL_ERROR_ANALYSIS_SCRIPT,
            "GLOBAL ERROR ANALYSIS",
            int_replacements={
                "PROMPT_STAGE1_ID":
                    PROMPT_STAGE1_ID,
                "PROMPT_STAGE2_ID":
                    PROMPT_STAGE2_ID,
            },
        )

    except Exception as exc:

        handle_error(
            "global_error_analysis",
            exc
        )

    # ========================================================
    # FASE 6 - GLOBAL F1
    # ========================================================

    if RUN_GLOBAL_F1:

        print("\n\n" + "#" * 90)
        print(
            "### FASE 6/6 - "
            "GLOBAL CLIP-LEVEL F1 SCORE"
        )
        print("#" * 90, flush=True)

        try:

            run_modified_script(
                GLOBAL_F1_SCRIPT,
                "GLOBAL CLIP-LEVEL F1 SCORE",
                string_replacements={
                    "EVALUATION_TYPE":
                        EVALUATION_TYPE,
                    "PROMPT_MODEL":
                        PROMPT_MODEL,
                },
                all_int_replacements={
                    "PROMPT_STAGE1_ID":
                        PROMPT_STAGE1_ID,
                    "PROMPT_STAGE2_ID":
                        PROMPT_STAGE2_ID,
                },
            )

        except Exception as exc:

            handle_error(
                "global_f1",
                exc
            )

    # ========================================================
    # FINE
    # ========================================================

    print("\n" + "=" * 90)
    print(
        "PIPELINE LARGE VIDEO "
        "SINGLE CLIP COMPLETATA"
    )
    print("=" * 90)

    print(
        f"Prompt utilizzati: "
        f"{PROMPT_STAGE1_ID} + "
        f"{PROMPT_STAGE2_ID}"
    )

    print(
        f"Evaluation type: "
        f"{EVALUATION_TYPE}"
    )


if __name__ == "__main__":
    main()