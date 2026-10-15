'''
GROUND_TRUTH = {
    "01-medico in camice attraversa sala di attesa di corsa.mp4": 1,
    "02-persona sviene mentre aspetta in sala di attesa.mp4": 3,
    "03-persona sviene mentre aspetta in sala di attesa quasi vuota.mp4": 3,
    "05-troppe persone in sala di attesa.mp4": 2,
    "06-barella-lasciata-davanti-uscita-emergenza.mp4": 2,
    "07-persone-in-fuga-verso-uscita-emergenza.mp4": 3,
    "08-persona sviene mentre aspetta in sala di attesa i presenti accorrono.mp4": 3,
    "09-persona abbandona borsa.mp4": 2,
    "10-la sala diventa affollata.mp4": 2,
    "11-signora lascia borsetta sulla sedia.mp4": 1,
    "12-terremoto in un ufficio.mp4": 3,
    "13-blackout in ospedale.mp4": 3,
    "14 bis-Fumo in sala.mp4": 3,
    "14-Incendio per sovraccarico corrente.mp4": 4,
    "010-fumo in ospedale.mp4": 3,
    "19- Presenza di animali in una sala di attesa.mp4": 1,
    "21- Persona con bandiera.mp4": 1,
    "21-bis-Persona con bandiera.mp4": 1,
    "video 23.mp4": 0,
    "video 24.mp4": 0,
}
'''
GROUND_TRUTH = {
    "01-medico in camice attraversa sala di attesa di corsa.mp4": 1,
    "02-persona sviene mentre aspetta in sala di attesa.mp4": 2,
    "03-persona sviene mentre aspetta in sala di attesa quasi vuota.mp4": 2,
    "05-troppe persone in sala di attesa.mp4": 1,
    "06-barella-lasciata-davanti-uscita-emergenza.mp4": 1,
    "07-persone-in-fuga-verso-uscita-emergenza.mp4": 2,
    "08-persona sviene mentre aspetta in sala di attesa i presenti accorrono.mp4": 2,
    "09-persona abbandona borsa.mp4": 1,
    "10-la sala diventa affollata.mp4": 1,
    "11-signora lascia borsetta sulla sedia.mp4": 1,
    "12-terremoto in un ufficio.mp4": 2,
    "13-blackout in ospedale.mp4": 2,
    "14 bis-Fumo in sala.mp4": 2,
    "14-Incendio per sovraccarico corrente.mp4": 2,
    "010-fumo in ospedale.mp4": 2,
    "19- Presenza di animali in una sala di attesa.mp4": 1,
    "21- Persona con bandiera.mp4": 1,
    "21-bis-Persona con bandiera.mp4": 1,
    "video 23.mp4": 0,
    "video 24.mp4": 0,
}

GROUND_TRUTH_DESCRIPTIONS = {
    "01-medico in camice attraversa sala di attesa di corsa.mp4":
        ("Un video di un ambiente ospedaliero in cui diverse "
         "persone sono in attesa di accedere a servizi, alcune in piedi, altre sedute. "
         "Ad un certo istante del video un medico, riconoscibile perché indossa un camice verde, "
         "attraversa la sala di corsa. La scena deve essere rappresentata come osservata da una "
         "videocamera di sicurezza montata a circa tre metri da terra"),
    "02-persona sviene mentre aspetta in sala di attesa.mp4":
        (" un video di un ambiente ospedaliero in cui diverse persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute. Ad un certo istante una delle persone in piedi si accascia a terra"
         " e le persone nelle sue vicinanze si avvicinano per chiedere come sta"),
    "03-persona sviene mentre aspetta in sala di attesa quasi vuota.mp4":
        ("un video di un ambiente ospedaliero in cui diverse persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute. Ad un certo istante una delle persone in piedi si accascia a terra"
         " mentre le altre se ne sono andate"),
    "05-troppe persone in sala di attesa.mp4":
        ("un video di un ambiente ospedaliero in cui molte persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute. Le persone sono così tante da non poter vedere il pavimento della sala."),
    "06-barella-lasciata-davanti-uscita-emergenza.mp4":
        (" un video di un ambiente ospedaliero con una sala di attesa dove sono disposte diverse file di sedie"
         " e degli sportelli di accettazione. Nella sala è anche visibile la porta di una uscita di emergenza."
         " Ad un certo punto del video si vedono degli operatori sanitari che spingono una barella senza nessuno sopra."
         " Gli operatori lasciano la barella di fronte all'uscita di emergenza e vanno via attraversando la sala"),
    "07-persone-in-fuga-verso-uscita-emergenza.mp4":
        (" un video di una sala di attesa dove sono disposte diverse file di sedie e degli sportelli di accettazione."
         " Alcune persone sono sedute ed altre aspettano in piedi. Nella sala è visibile la porta di una uscita "
         "di emergenza. Ad un certo punto del video tutte le persone scappano verso l'uscita di emergenza."),
    "08-persona sviene mentre aspetta in sala di attesa i presenti accorrono.mp4":
        ("un video di un corridoio dove sono disposte due file di sedie. Alcune persone sono sedute"
         " ed altre aspettano in piedi. Ad un certo punto del video una delle persone in piedi cade per terra."),
    "09-persona abbandona borsa.mp4":
        ("Siamo in una sala di attesa dove sono disposte diverse file di sedie e degli sportelli di accettazione."
         " Alcune persone sono sedute ed altre aspettano in piedi. Ad un certo punto nella sala entra una persona"
         " con una borsa. La persona lascia la borsa vicino ad una sedia e poi va via dalla sala."),
    "10-la sala diventa affollata.mp4":
        ("Siamo in una sala di attesa dove sono disposte diverse file di sedie e degli sportelli di accettazione. "
         "Alcune persone sono sedute ed altre aspettano in piedi. Pian piano nella sala entrano nuove persone "
         "che rimangono in piedi in attesa sino a che la sala è piena di persone."),
    "11-signora lascia borsetta sulla sedia.mp4":
        ("Siamo in una sala di attesa dove sono disposte diverse file di sedie e degli sportelli di accettazione."
         " Alcune persone sono sedute ed altre aspettano in piedi."
         " Ad un certo punto nella sala entra una signora vestita di rosso con una borsetta."
         " La signora lascia la borsetta su una sedia e poi va via dalla sala."),
    "12-terremoto in un ufficio.mp4":
        ("Un video di videosorveglianza di un ambiente lavorativo, un ufficio all'interno di un edificio,"
         " in cui molte persone stanno lavorando sedute alle loro scrivanie davanti al loro PC"
         " e con documenti vari, altre stanno parlando in piedi. Ad un certo punto si verifica un terremoto forte,"
         " cadono schermi del pc, tremano i tavoli e cadono altri oggetti. I presenti sono presi dal terrore"
         " alcuni corrono all'uscita, altri si rifugiano sotto i banchi"),
    "13-blackout in ospedale.mp4":
        ("Un video di un ambiente ospedaliero in cui alcune persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute. Ad un certo punto si verifica un blackout, le luci si spengono"
         " e si accendono quelle di emergenza"),
    "14 bis-Fumo in sala.mp4":
        ("Un video di un ambiente ospedaliero in cui alcune persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute. Ad un certo punto una striscia di fumo appare da una luce del soffitto."
         " Improvvisamente una delle persone in sala si accorge del fumo scatta in piedi e indica il fumo gridando"
         " ‘attenzione c’è del fumo’ le altre persone si girano tutte verso il punto indicato dalla persona in piedi,"
         " eventualmente una persona va a prendere un estintore e cerca di domare il principio di incendio. "),
    "14-Incendio per sovraccarico corrente.mp4":
        ("Un video di videosorveglianza in un ufficio bancario in cui alcune persone sono in attesa di accedere"
         " a servizi, alcune in piedi, altre sedute. Ad un certo punto si verifica un incendio causato da"
         " un sovraccarico di una presa elettrica di un computer."
         " La gente si dirige verso l'uscita di sicurezza spaventata"),
    "010-fumo in ospedale.mp4":
        ("Un video di un ambiente ospedaliero in cui alcune persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute. Ad un certo punto, dalla finestra, si nota che dal piano superiore proviene"
         " del fumo. La finestra viene oscurata. Le persone in sala risultano preoccupate, così come gli infermieri"),
    "19- Presenza di animali in una sala di attesa.mp4":
        ("Un video di un ambiente ospedaliero in cui alcune persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute. A un certo punto entra una signora con un cane al guinzaglio"
         " e si mette a sedere."),
    "21- Persona con bandiera.mp4":
        ("Un video di un ambiente ospedaliero in cui alcune persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute. A un certo punto una persona entra nella sala con una bandiera"
         " e la sventola magari cantando una canzone, tutte le persone in sala si voltano a guardare la persona"
         " con la bandiera ma non si spostano, restano cioè a sedere dove erano "),
    "21-bis-Persona con bandiera.mp4":
        ("Un video di un ambiente ospedaliero in cui alcune persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute. A un certo punto una persona entra nella sala con una bandiera"
         " e la sventola magari cantando una canzone, tutte le persone in sala si voltano a guardare la persona"
         " con la bandiera ma non si spostano, restano cioè a sedere dove erano "),
    "video 23.mp4":
        ("Un video di un ambiente ospedaliero in cui alcune persone sono in attesa di accedere a servizi,"
         " alcune in piedi, altre sedute, nella sala esiste una porta con scritto uscita di sicurezza."
         " A un certo punto alcune persone entrano nella sala e si fermano in piedi a parlare davanti alla porta"
         " con la scritta uscita di sicurezza, le persone in sala non si accorgono di nulla e continuano a fare"
         " quello che facevano prima. "),
    "video 24.mp4":
        (" Un video in cui è rappresentato un ambiente ospedaliero con alcune persone in attesa di accedere a servizi."
         " Alcune persone sono in piedi, altre sedute. Nella sala è visibile una porta con scritto uscita di emergenza."
         " A un certo punto una persona che era seduta si alza e si dirige verso un desk informativo.  "),
}

'''
DANGER_LABELS = {
    0: "NONE",
    1: "LOW",
    2: "MODERATE",
    3: "HIGH",
    4: "CRITICAL",
}
'''
DANGER_LABELS = {
    0: "NONE",
    1: "WARNING",
    2: "DANGER",
}

array = [
    [1, 1, 1, 1, 1, 2, 1, 2, 0, 1],  # sequence_01
    [2, 2, 0, 1, 2, 1, 1, 2, 1, 1],  # sequence_02
    [2, 2, 1, 1, 1, 2, 1, 2, 0, 2],  # sequence_03
    [1, 2, 1, 2, 1, 0, 1, 2, 1, 2],  # sequence_04
    [2, 1, 2, 1, 1, 2, 2, 1, 1, 1],  # sequence_05
    [2, 1, 2, 2, 2, 1, 1, 2, 1, 0],  # sequence_06
    [2, 2, 2, 2, 2, 1, 2, 1, 2, 1],  # sequence_07
    [2, 2, 0, 1, 2, 1, 1, 1, 2, 2],  # sequence_08
    [1, 0, 1, 1, 1, 2, 2, 0, 1, 1],  # sequence_09
    [1, 1, 1, 2, 2, 2, 1, 1, 2, 2],  # sequence_10
]