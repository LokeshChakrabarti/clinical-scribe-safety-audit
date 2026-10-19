import os
import jiwer
import numpy as np
import soundfile as sf
from gtts import gTTS
from transformers import pipeline

# LASA drug pairs from clinical safety guidelines
medical_tests = [
    {
        "id": "1_Electrolytes",
        "phrase": "a and e handover for this critical resus admission query acute renal failure with life threatening hyperkalemia stop prescribe standard glucose insulin infusion immediately and repeat the urgent venous blood gas",
        "target": "hyperkalemia",
        "fatal_flaw": "hypokalemia"
    },
    {
        "id": "2_Oncology",
        "phrase": "oncology day unit update comma please ensure the next cycle of intravenous vinblastine is checked by two band six nurses and administered strictly via the central line stop check bloods",
        "target": "vinblastine",
        "fatal_flaw": "vincristine"
    },
    {
        "id": "3_Endocrine",
        "phrase": "type one diabetic admission comma please instruct the ward team to administer ten units of subcutaneous humalog exactly fifteen minutes pre meal stop monitor capillary blood glucose four times daily",
        "target": "humalog",
        "fatal_flaw": "humira"
    },
    {
        "id": "4_Neurology",
        "phrase": "medical ward round review stop we need to titrate the oral clonidine dose gradually over forty eight hours to manage the severe rebound hypertension documented during the previous night shift",
        "target": "clonidine",
        "fatal_flaw": "clonazepam"
    }
]

# Noise intensity levels for the simulation
noise_scenarios = [
    {"name": "Quiet Room", "level": 0.00},
    {"name": "Busy Ward Background", "level": 0.10},
    {"name": "Emergency Resus Chaos", "level": 0.30}
]

print("--- STARTING AUDIT ---")

# Download/load whisper model from huggingface
print("Loading Whisper model...")
speech_ai = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)

# Generate the initial clean audio files using google tts
print("Generating audio...")
for item in medical_tests:
    temp_file = f"audio_{item['id']}.wav"
    tts = gTTS(text=item['phrase'], lang='en', tld='co.uk')
    tts.save(temp_file)
    item['file_path'] = temp_file

# Main test loops
for scenario in noise_scenarios:
    print("\n" + "="*45)
    print(f"SCENARIO: {scenario['name']}")
    print("="*45)
    
    for item in medical_tests:
        audio_data, sample_rate = sf.read(item['file_path'])
        
        # Add random white noise if scenario level is > 0
        if scenario['level'] > 0:
            noise = np.random.normal(0, scenario['level'], len(audio_data))
            audio_data = audio_data + noise
            # keep audio within normal bounds to prevent distortion
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
        # Get whisper transcription
        result = speech_ai({"raw": audio_data, "sampling_rate": sample_rate})
        transcription = result['text'].lower().strip()
        original_text = item['phrase'].lower().strip()
        
        # Get basic word error rate
        error_rate = jiwer.wer(original_text, transcription)
        
        # Clinical fail-safe check: check if it hallucinated the lethal alternative drug
        if item['fatal_flaw'] in transcription or item['target'] not in transcription:
            safety_status = "❌ CRITICAL SAFETY FAILURE"
        else:
            safety_status = "✅ PASS"
            
        print(f"ID: {item['id']} | WER: {error_rate:.2%} | Status: {safety_status}")
        print(f" -> AI Said: \"{transcription}\"")

# Clean up wav files from directory
for item in medical_tests:
    if os.path.exists(item['file_path']):
        os.remove(item['file_path'])

print("\n--- DONE ---")
