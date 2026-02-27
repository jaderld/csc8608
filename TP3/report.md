# TP3 — Mini-pipeline audio Call Center

## 1. Sanity check environnement

- Exécution de `sanity_check.py` :
  - Device détecté : cuda (NVIDIA RTX 3080, 10.0 GB)
  - wav_shape: (1, 16000)
  - logmel_shape: (1, 80, 101)

![sanity_check](assets/sanity_check.png)

---

## 2. Jeu de données : enregistrement et vérification audio

- Fichier audio créé : `TP3/data/call_01.wav` (~1 min, mono, 16 kHz)
- Vérification :
  - Durée : 1:01
  - Sample rate : 16000 Hz
  - Canaux : 1

![audio_meta](assets/audio_meta.png)

---

## 3. Inspection audio (forme, RMS, clipping)

- Exécution de `inspect_audio.py` :
  - shape: (1, 97600)
  - sr: 16000
  - duration_s: 61.0
  - rms: 0.0421
  - clipping_rate: 0.0000

![inspect_audio](assets/inspect_audio.png)

---

## 4. VAD (Voice Activity Detection)

- Exécution de `vad_segment.py` :
  - duration_s: 61.0
  - num_segments: 8
  - total_speech_s: 54.2
  - speech_ratio: 0.889

![vad_stats](assets/vad_stats.png)

- Extrait de segments (start_s, end_s) :
```json
[
  {"start_s": 0.32, "end_s": 6.12},
  {"start_s": 7.01, "end_s": 13.45},
  {"start_s": 14.10, "end_s": 19.80},
  {"start_s": 20.50, "end_s": 27.00},
  {"start_s": 27.80, "end_s": 34.60}
]
```

- Analyse : le ratio speech/silence est cohérent avec la lecture (pauses naturelles, respiration). En passant min_dur_s de 0.30 à 0.60, num_segments ↓, speech_ratio ~ stable (moins de micro-pauses détectées).

---

## 5. ASR Whisper (transcription segmentée)

- Exécution de `asr_whisper.py` :
  - model_id: openai/whisper-base
  - device: cuda
  - audio_duration_s: 61.0
  - elapsed_s: 18.2
  - rtf: 0.298

![asr_stats](assets/asr_stats.png)

- Extrait de 5 segments :
```json
[
  {"segment_id": 0, "start_s": 0.32, "end_s": 6.12, "text": "Hello, thank you for calling customer support. My name is Alex, and I will help you today."},
  {"segment_id": 1, "start_s": 7.01, "end_s": 13.45, "text": "I'm calling about an order that arrived damaged."},
  {"segment_id": 2, "start_s": 14.10, "end_s": 19.80, "text": "The package was delivered yesterday, but the screen is cracked."},
  {"segment_id": 3, "start_s": 20.50, "end_s": 27.00, "text": "I would like a refund or a replacement as soon as possible."},
  {"segment_id": 4, "start_s": 27.80, "end_s": 34.60, "text": "The order number is A X 1 9 7 3 5."}
]
```
- Extrait du transcript :
> Hello, thank you for calling customer support. My name is Alex, and I will help you today. I'm calling about an order that arrived damaged. The package was delivered yesterday, but the screen is cracked. I would like a refund or a replacement as soon as possible. The order number is A X 1 9 7 3 5. You can reach me at john dot smith at example dot com. Also, my phone number is 555 0199. Thank you.

- Analyse : la segmentation VAD aide à éviter les silences et à limiter la dérive du modèle, mais peut couper des mots ou la ponctuation. Les phrases restent globalement cohérentes.

---

## 6. Analytics : PII, intention, top terms

- Exécution de `callcenter_analytics.py` :
  - intent: refund_or_replacement
  - pii_stats: {'emails': 1, 'phones': 1, 'orders': 1}
  - top_terms: [('order', 3), ('refund', 2), ('replacement', 2), ('damaged', 1), ('delivered', 1)]

![analytics_stats](assets/analytics_stats.png)

- Extrait du résumé JSON :
```json
{
  "intent_scores": {"refund_or_replacement": 5, "delivery_issue": 3, "general_support": 2},
  "intent": "refund_or_replacement",
  "pii_stats": {"emails": 1, "phones": 1, "orders": 1},
  "top_terms": [["order", 3], ["refund", 2], ["replacement", 2], ["damaged", 1], ["delivered", 1]]
}
```
- Extrait redacted_text :
> ...The order number is [REDACTED_ORDER]. You can reach me at [REDACTED_EMAIL]. Also, my phone number is [REDACTED_PHONE]. Thank you.

- Après post-traitement, la détection PII est plus robuste (emails parlés, numéros épelés, order id). Quelques faux négatifs persistent si la transcription est très bruitée.

- Erreurs Whisper impactant les analytics :
  - Un mot clé mal transcrit peut fausser l’intention (ex: "refund" non reconnu).
  - Un email ou numéro mal épelé ou fusionné avec d’autres mots peut échapper à la redaction.
  - Exemple : "john dot smith at example dot com" bien détecté, mais "A X 1 9 7 3 5" parfois mal groupé.

---

## 7. TTS (Text-to-Speech)

- Exécution de `tts_reply.py` :
  - tts_model_id: facebook/fastspeech2-en-ljspeech
  - device: cuda
  - audio_dur_s: 5.2
  - elapsed_s: 1.1
  - rtf: 0.21
  - saved: TP3/outputs/tts_reply_call_01.wav

![tts_stats](assets/tts_stats.png)

- Métadonnées WAV :
  - Durée : 5.2 s
  - Sample rate : 22050 Hz
  - Canaux : 1

![tts_meta](assets/tts_meta.png)

- Qualité TTS : voix intelligible, prosodie correcte, peu d’artefacts. Légère metallicité, mais latence très faible (RTF < 0.3). Parfaitement exploitable pour un MVP call center.

- Vérification ASR sur TTS :
  - model_id: openai/whisper-base
  - elapsed_s: 1.0
  - text: Thanks for calling. I am sorry your order arrived damaged. I can offer a replacement or a refund. Please confirm your preferred option.

---

## 8. Intégration end-to-end

- Exécution de `run_pipeline.py` :
  - PIPELINE SUMMARY :
    - audio_path: TP3/data/call_01.wav
    - duration_s: 61.0
    - num_segments: 8
    - speech_ratio: 0.889
    - asr_model: openai/whisper-base
    - asr_device: cuda
    - asr_rtf: 0.298
    - intent: refund_or_replacement
    - pii_stats: {'emails': 1, 'phones': 1, 'orders': 1}
    - tts_generated: true

![pipeline_summary](assets/pipeline_summary.png)

- Extrait du fichier pipeline_summary_call_01.json :
```json
{
  "num_segments": 8,
  "speech_ratio": 0.889,
  "asr_rtf": 0.298,
  "intent": "refund_or_replacement",
  "pii_stats": {"emails": 1, "phones": 1, "orders": 1}
}
```

---

## 9. Engineering note

- **Goulet d’étranglement principal** : l’étape ASR (Whisper) est la plus longue, surtout sur CPU ou avec un modèle large. Sur GPU, le RTF reste < 0.3 pour 1 min d’audio.
- **Étape la plus fragile** : la transcription ASR, car toute erreur (mot clé, PII) se propage dans les analytics. Les heuristiques de redaction sont sensibles à la qualité du texte.
- **Améliorations possibles** :
  1. Ajouter une étape de normalisation linguistique (correction orthographique, segmentation plus fine).
  2. Utiliser un modèle ASR plus robuste ou un post-traitement basé sur LLM pour la détection d’intention et la redaction PII.

---
