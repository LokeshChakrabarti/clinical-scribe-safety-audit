# clinical-scribe-safety-audit
Testing if open-source speech AI fails when transcribing look-alike/sound-alike drug names under realistic hospital background noise.

Standard software testing evaluates transcription models using an overall Word Error Rate (WER). However, in a high-stakes clinical environment like an Emergency Department, an overall 98% accuracy score is dangerously misleading if the remaining 2% error involves swapping look-alike/sound-alike (LASA) drug names that could cause fatal prescription or administration errors.

## What this script does
1. Generates base audio by converting standard clinical handover scripts containing high-risk, LASA medical terms into clean audio files using Google Text-to-Speech (gTTS).
2. Simulates noisy clinical environments by progressively overlaying uniform white noise mathematically onto the waveforms.
3. Feeds the noisy files directly into OpenAI's Whisper model (whisper-tiny) to generate a new transcript, evaluates the WER, and checks if a target LASA drug has been swapped with its alternative. The script overrides standard pass metrics to flag a `CRITICAL SAFETY FAILURE` if this occurs.

## Setup and Libraries
To run the audit locally, you need Python installed alongside these libraries:
```bash
pip install numpy soundfile gtts transformers torch jiwer
```

## Running the Script
Simply paste the main `audit.py` script into your project folder and run it via terminal:
```bash
python audit.py
```

## Findings
1. When noise injection levels increase to represent chaotic environments, standard WERs go up, but the critical risk is the model hallucinating phonemically similar words.
2. At the highest white noise level, the AI could not make out any words at all, only repeating the white noise which caused a massive collapse.

## Project Limits
This script injects continuous Gaussian white noise as a mathematical baseline. In a true hospital setting, background interference consists of discrete, sharp acoustic spikes (such as monitor alarms, moving trolleys, or shouting). A natural next step for this project would be testing audio files overlaid with real hospital sound logs.
