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

# Cambia questi nomi solo se nel tuo progetto i file hanno nomi diversi.
INFERENCE_SCRIPT = (
    BASE_DIR
    / "inference"
    / "inference_virtual_composite_sequences_two_stage.py"
)

EVALUATE_SCRIPT = (
    BASE_DIR
    / "evaluate"
    / "evaluate_composite_two_stage_error_analysis.py"
)
ERROR_ANALYSIS_SCRIPT = (
    BASE_DIR / "error_analysis" / "global_error_analysis_ai.py"
)
CONFUSION_SCRIPT = (
    BASE_DIR / "metrics" / "confusion_matrix.py"
)
GLOBAL_CONFUSION_SCRIPT = (
    BASE_DIR / "metrics" / "global_confusion_matrix.py"
)
GLOBAL_F1_SCRIPT = (
    BASE_DIR / "metrics" / "global_clip_level_f1_score.py"
)
# Sequenze da 01 a 10
SEQUENCE_IDS = [f"{i:02d}" for i in range(1, 11)]

# Se True, dopo la global confusion matrix esegue anche il calcolo F1 globale.
RUN_GLOBAL_F1 = True

# Se True, al primo errore interrompe tutto.
STOP_ON_ERROR = True


# ============================================================
# FUNZIONI
# ============================================================

def check_scripts():
    scripts = [
        INFERENCE_SCRIPT,
        EVALUATE_SCRIPT,
        ERROR_ANALYSIS_SCRIPT,
        CONFUSION_SCRIPT,
        GLOBAL_CONFUSION_SCRIPT,
    ]

    if RUN_GLOBAL_F1:
        scripts.append(GLOBAL_F1_SCRIPT)

    missing = [str(path) for path in scripts if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "File mancanti:\n" + "\n".join(missing)
        )


def replace_assignment(source, variable_name, value):
    """
    Sostituisce solo la prima assegnazione globale del tipo:
        VARIABLE = "..."
    senza modificare il file originale.
    """
    pattern = re.compile(
        rf'^(?P<prefix>{re.escape(variable_name)}\s*=\s*)'
        rf'(?P<quote>["\'])(?P<value>.*?)(?P=quote)\s*$',
        re.MULTILINE,
    )

    replacement = rf'\g<prefix>"{value}"'

    modified, count = pattern.subn(
        replacement,
        source,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            f"Impossibile sostituire {variable_name} in questo script."
        )

    return modified


def run_modified_script(script_path, variable_name, value, label):
    """
    Legge uno script, cambia una variabile nella copia temporanea,
    lo esegue e poi elimina automaticamente la copia.
    """
    print("\n" + "=" * 90)
    print(label)
    print(f"Script: {script_path.name}")
    print(f"{variable_name} = {value}")
    print("=" * 90, flush=True)

    source = script_path.read_text(encoding="utf-8")
    source = replace_assignment(
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
            temp_path = Path(tmp.name)

        env = os.environ.copy()

        # Evita che plt.show() blocchi il runner.
        # I grafici vengono comunque salvati su file dagli script.
        env.setdefault("MPLBACKEND", "Agg")

        result = subprocess.run(
            [sys.executable, str(temp_path)],
            cwd=BASE_DIR,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"{script_path.name} terminato con codice "
                f"{result.returncode}"
            )

    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def run_normal_script(script_path, label):
    print("\n" + "=" * 90)
    print(label)
    print(f"Script: {script_path.name}")
    print("=" * 90, flush=True)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} terminato con codice "
            f"{result.returncode}"
        )


def handle_error(sequence_id, stage, exc):
    print("\n" + "!" * 90)
    print(f"ERRORE | Sequenza {sequence_id} | Stage: {stage}")
    print(exc)
    print("!" * 90, flush=True)

    if STOP_ON_ERROR:
        raise exc


# ============================================================
# MAIN
# ============================================================

def main():
    check_scripts()

    print("=" * 90)
    print("PIPELINE 10 SEQUENZE - ESECUZIONE PER FASI")
    print("=" * 90)
    print("FASE 1: tutte le inference (01 -> 10)")
    print("FASE 2: tutte le evaluate (01 -> 10)")
    print("FASE 3: tutte le confusion matrix (01 -> 10)")
    print("FASE 4: evaluate global error analysis")
    print("FASE 5: global confusion matrix")
    if RUN_GLOBAL_F1:
        print("FASE 6: global clip-level F1 score")
    print("=" * 90)

    completed_inference = []
    completed_evaluate = []
    completed_confusion = []

    # ========================================================
    # FASE 1 - TUTTE LE INFERENZE
    # ========================================================

    print("\n\n" + "#" * 90)
    print("### FASE 1/6 - INFERENCE DI TUTTE LE SEQUENZE")
    print("#" * 90, flush=True)

    for sequence_id in SEQUENCE_IDS:
        try:
            run_modified_script(
                INFERENCE_SCRIPT,
                "SEQUENCE_ID",
                sequence_id,
                f"[INFERENCE] Sequenza {sequence_id}",
            )
            completed_inference.append(sequence_id)

        except Exception as exc:
            handle_error(sequence_id, "inference", exc)

    # Con STOP_ON_ERROR=False, valuta solo le sequenze
    # la cui inferenza è terminata correttamente.
    sequences_for_evaluation = completed_inference[:]

    # ========================================================
    # FASE 2 - TUTTE LE EVALUATE
    # ========================================================

    print("\n\n" + "#" * 90)
    print("### FASE 2/6 - EVALUATE DI TUTTE LE SEQUENZE")
    print("#" * 90, flush=True)

    for sequence_id in sequences_for_evaluation:
        try:
            run_modified_script(
                EVALUATE_SCRIPT,
                "VIDEO_ID",
                sequence_id,
                f"[EVALUATE] Sequenza {sequence_id}",
            )
            completed_evaluate.append(sequence_id)

        except Exception as exc:
            handle_error(sequence_id, "evaluate", exc)

    # Con STOP_ON_ERROR=False, genera la confusion matrix
    # solo per le evaluate completate correttamente.
    sequences_for_confusion = completed_evaluate[:]



    # ========================================================
    # FASE 3 - TUTTE LE CONFUSION MATRIX
    # ========================================================

    print("\n\n" + "#" * 90)
    print("### FASE 3/6 - CONFUSION MATRIX DI TUTTE LE SEQUENZE")
    print("#" * 90, flush=True)

    for sequence_id in sequences_for_confusion:
        try:
            run_modified_script(
                CONFUSION_SCRIPT,
                "VIDEO_ID",
                sequence_id,
                f"[CONFUSION MATRIX] Sequenza {sequence_id}",
            )
            completed_confusion.append(sequence_id)

        except Exception as exc:
            handle_error(sequence_id, "confusion_matrix", exc)

    # ========================================================
    # CONTROLLO PRIMA DELLE ANALISI GLOBALI
    # ========================================================

    if len(completed_evaluate) != len(SEQUENCE_IDS):
        missing = [
            sequence_id
            for sequence_id in SEQUENCE_IDS
            if sequence_id not in completed_evaluate
        ]

        raise RuntimeError(
            "Non tutte le evaluate sono state completate. "
            "Le analisi globali non vengono eseguite per evitare "
            "metriche calcolate su un insieme incompleto.\n"
            f"Sequenze mancanti: {missing}"
        )

    # ========================================================
    # FASE 4 - ERROR GLOBAL ANALYSIS
    # ========================================================

    print("\n\n" + "#" * 90)
    print("### FASE 4/6 -  GLOBAL ERROR ANALYSIS")
    print("#" * 90, flush=True)

    run_normal_script(
        ERROR_ANALYSIS_SCRIPT,
        " GLOBAL ERROR ANALYSIS",
    )
    # ========================================================
    # FASE 5 - GLOBAL CONFUSION MATRIX
    # ========================================================

    print("\n\n" + "#" * 90)
    print("### FASE 5/6 - GLOBAL CONFUSION MATRIX")
    print("#" * 90, flush=True)

    run_normal_script(
        GLOBAL_CONFUSION_SCRIPT,
        "GLOBAL CONFUSION MATRIX",
    )

    # ========================================================
    # FASE 6 - GLOBAL F1
    # ========================================================

    if RUN_GLOBAL_F1:
        print("\n\n" + "#" * 90)
        print("### FASE 6/6 - GLOBAL CLIP-LEVEL F1 SCORE")
        print("#" * 90, flush=True)

        run_normal_script(
            GLOBAL_F1_SCRIPT,
            "GLOBAL CLIP-LEVEL F1 SCORE",
        )

    # ========================================================
    # RIEPILOGO
    # ========================================================

    print("\n" + "=" * 90)
    print("PIPELINE COMPLETATA")
    print("=" * 90)
    print(f"Inference completate: {len(completed_inference)} / {len(SEQUENCE_IDS)}")
    print(f"Evaluate completate:  {len(completed_evaluate)} / {len(SEQUENCE_IDS)}")
    print(f"Confusion completate: {len(completed_confusion)} / {len(SEQUENCE_IDS)}")
    print(f"Sequence inference: {completed_inference}")
    print(f"Sequence evaluate:  {completed_evaluate}")
    print(f"Sequence confusion: {completed_confusion}")


if __name__ == "__main__":
    main()
