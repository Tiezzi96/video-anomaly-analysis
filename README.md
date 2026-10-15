# Analisi di anomalie video con HolmesVAU e LLM

Progetto basato su **HolmesVAU**, esteso con pipeline per analizzare video lunghi e sequenze di video generati con AI.

HolmesVAU produce descrizioni delle clip, successivamente elaborate da un secondo modello tramite Ollama per classificare le anomalie e ricostruirne l’evoluzione temporale.

Sono disponibili una modalità two stage e una variante con valutazione delle singole clip.

## Struttura

- `inference/`: analisi dei video e generazione dei report.
- `evaluate/`: confronto delle predizioni con la ground truth.
- `metrics/`: confusion matrix, precision, recall e F1.
- `error_analysis/`: analisi aggregata degli errori.
- `runners/`: esecuzione delle pipeline complete.
- `utils/`: prompt, ground truth e strumenti di supporto.
- `holmesvau/` e `internvl_chat/`: componenti del progetto originale.

## Preparazione

È necessario un ambiente Python con le dipendenze del progetto, PyTorch con supporto CUDA, Ollama e i modelli utilizzati.

```bash
pip install -r requirements.txt
```

I pesi di [HolmesVAU-2B](https://huggingface.co/ppxin321/HolmesVAU-2B) devono essere disponibili in `ckpts/HolmesVAU-2B`.

Prima dell’esecuzione, verificare negli script:

- modello Ollama e ID dei prompt;
- percorsi dei dataset e delle annotazioni;
- percorso di FFmpeg per le pipeline dei video lunghi.

## Esecuzione

Dalla cartella principale, con l’ambiente Python attivo:

```bash
# Sequenze AI: two stage
python runners/run_all_sequences.py

# Sequenze AI: single clip stage 1
python runners/run_all_sequences_single_clip_stage1.py

# Video lunghi: two stage
python runners/run_large_video_full_pipeline.py

# Video lunghi: single clip stage 1
python runners/run_large_video_single_clip_full_pipeline.py
```

Le sequenze AI sono definite dai file presenti in `sequenze AI/`. I video corrispondenti si trovano in `Video Generati AI/`.

I video lunghi e le relative annotazioni sono nella cartella `dataset classificato/`.

## Risultati

Gli output principali vengono salvati in:

- `composite_video_output/`: risultati delle sequenze AI.
- `large_video_evaluation/`: valutazioni e metriche dei video lunghi.
- File TXT nella cartella principale: report di inferenza dei video lunghi.

Le cartelle di output, i checkpoint e i file temporanei sono esclusi dal versionamento tramite `.gitignore`.

## Riferimenti e licenza

Questo lavoro deriva da **Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity**.

- [Articolo originale](https://arxiv.org/abs/2412.06171)
- [Modello HolmesVAU-2B](https://huggingface.co/ppxin321/HolmesVAU-2B)

Il codice originale mantiene le attribuzioni e la licenza MIT riportate nel file `LICENSE`.