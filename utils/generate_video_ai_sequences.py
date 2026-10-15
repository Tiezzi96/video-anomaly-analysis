import random
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SEQUENCES_DIR = os.path.join(
    PROJECT_ROOT,
    "sequenze AI"
)

os.makedirs(SEQUENCES_DIR, exist_ok=True)
video_ai_list = [
    "01-medico in camice attraversa sala di attesa di corsa.mp4",
    "02-persona sviene mentre aspetta in sala di attesa.mp4",
    "03-persona sviene mentre aspetta in sala di attesa quasi vuota.mp4",
    "05-troppe persone in sala di attesa.mp4",
    "06-barella-lasciata-davanti-uscita-emergenza.mp4",
    "07-persone-in-fuga-verso-uscita-emergenza.mp4",
    "08-persona sviene mentre aspetta in sala di attesa i presenti accorrono.mp4",
    "09-persona abbandona borsa.mp4",
    "010-fumo in ospedale.mp4",
    "10-la sala diventa affollata.mp4",
    "11-signora lascia borsetta sulla sedia.mp4",
    "12-terremoto in un ufficio.mp4",
    "13-blackout in ospedale.mp4",
    "14 bis-Fumo in sala.mp4",
    "14-Incendio per sovraccarico corrente.mp4",
    "19- Presenza di animali in una sala di attesa.mp4",
    "21- Persona con bandiera.mp4",
    "21-bis-Persona con bandiera.mp4",
    "video 23.mp4",
    "video 24.mp4"
]
# genera sequenze random di lunghezza k con ripetizione
video_ai_sequence = random.choices(video_ai_list,k=100)
print(video_ai_sequence)
# "x" impedisce che sequenze con lo stesso nome siano sovrascritte lanciando FileExistError
# usa "w" se voui che file con lo stesso nome siano sovrascritti
with open(
    os.path.join(SEQUENCES_DIR, "video_ai_sequence_large.txt"),
    "x",
    encoding="utf-8"
) as f:
    f.write("\n".join(video_ai_sequence))