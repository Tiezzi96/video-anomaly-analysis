import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURAZIONE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modifica questi nomi solo se i tuoi script hanno nomi diversi.
INFERENCE_SCRIPT = "inference/inference_large_videos_two_stage.py"
EVALUATE_SCRIPT = "evaluate/evaluate_large_video_two_stage_all_videos.py"
GLOBAL_ERROR_ANALYSIS_SCRIPT = (
    "error_analysis/global_error_analysis_large.py"
)
SINGLE_CM_SCRIPT = (
    "metrics/confusion_matrix_large_video_all_videos.py"
)

GLOBAL_CM_SCRIPT = (
    "metrics/global_confusion_matrix_large_video.py"
)


# ============================================================
# STEP DELLA PIPELINE
# ============================================================

STEPS = [
    (
        "1/5 - INFERENCE SU TUTTI I LARGE VIDEO",
        PROJECT_ROOT / INFERENCE_SCRIPT
    ),
    (
        "2/5 - EVALUATE SU TUTTI I VIDEO VALUTABILI",
        PROJECT_ROOT / EVALUATE_SCRIPT
    ),
    (
        "3/5 - CONFUSION MATRIX SINGOLE",
        PROJECT_ROOT / SINGLE_CM_SCRIPT
    ),
    (
        "4/5 - CONFUSION MATRIX GLOBALE",
        PROJECT_ROOT / GLOBAL_CM_SCRIPT
    ),
    (
        "5/5 - ANALISI GLOBALE DEGLI ERRORI",
        PROJECT_ROOT / GLOBAL_ERROR_ANALYSIS_SCRIPT
    ),
]


# ============================================================
# ESECUZIONE DI UNO SCRIPT
# ============================================================

def run_script(label, script_path):

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script non trovato:\n{script_path}"
        )

    subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        cwd=str(PROJECT_ROOT),
        check=True
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("LARGE VIDEO - FULL PIPELINE")
    print("=" * 80)
    print("Project root:", PROJECT_ROOT)

    for label, script_path in STEPS:

        try:

            run_script(
                label,
                script_path
            )

        except subprocess.CalledProcessError as error:

            print("\n" + "=" * 80)
            print("PIPELINE INTERROTTA")
            print("=" * 80)
            print(
                f"Lo step '{label}' è terminato "
                f"con codice di errore {error.returncode}."
            )
            sys.exit(error.returncode)

        except Exception as error:

            print("\n" + "=" * 80)
            print("PIPELINE INTERROTTA")
            print("=" * 80)
            print(
                f"Errore nello step '{label}': {error}"
            )
            sys.exit(1)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETATA")
    print("=" * 80)
    print(
        "Inference, evaluate, analisi globale degli errori, "
        "confusion matrix singole e confusion matrix globale completate."
    )
