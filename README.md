# Josh Talks — AI Researcher Intern (Speech & Audio) Assignment
## Complete Setup & Run Guide

---

## 📁 Project Structure

```
josh-talks-asr/
├── question1_whisper_finetune.py    ← Q1: Whisper fine-tuning pipeline
├── question2_cleanup_pipeline.py    ← Q2: Number norm + English tagging
├── question3_spell_checker.py       ← Q3: Hindi spell checker
├── question4_lattice_wer.py         ← Q4: Lattice-based WER
├── run_all.py                       ← Unified runner (interactive menu)
├── requirements.txt                 ← All dependencies
└── README.md                        ← This file
```

---

## 🔗 Real Data URLs

| Resource | URL |
|---|---|
| Sample Transcription JSON | https://storage.googleapis.com/upload_goai/967179/825780_transcription.json |
| Sample Audio WAV | https://storage.googleapis.com/upload_goai/967179/825780.wav |
| Sample Metadata JSON | https://storage.googleapis.com/upload_goai/967179/825780_metadata.json |
| Dataset Metadata Sheet | https://docs.google.com/spreadsheets/d/1bujiO2NgtHlgqPlNvYAQf5_7ZcXARlIfNX5HNb9f8cI |
| Q1 Metadata Sheet | https://docs.google.com/spreadsheets/d/1JItJnilmmSWjx9tAIr06cbTsyGjMMxEMhaebvn5qBHM |
| Q3 Word List Sheet | https://docs.google.com/spreadsheets/d/1J_I0raoRNbe29HiAPD5FROTr0jC93YtSkjOrIglKEjU |

> **GCS URL Pattern:**
> ```
> Audio        → https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}.wav
> Transcription→ https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_transcription.json
> Metadata     → https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_metadata.json
> ```

---

## ⚙️ Step 1 — Environment Setup

### Option A: pip (recommended)
```bash
pip install -r requirements.txt
```

### Option B: conda
```bash
conda create -n josh-asr python=3.10 -y
conda activate josh-asr
pip install -r requirements.txt
```

### Option C: install manually
```bash
pip install torch torchaudio transformers datasets evaluate jiwer \
            librosa soundfile accelerate tqdm pandas numpy \
            requests openpyxl
```

> **GPU check:**
> ```bash
> python -c "import torch; print(torch.cuda.is_available())"
> ```
> Q1 (fine-tuning) needs a GPU. Q2, Q3, Q4 run fine on CPU.

---

## 🗂️ Step 2 — Prepare Google Sheet Data (for Q1 and Q3)

### Download Q1 Metadata as CSV
1. Open: https://docs.google.com/spreadsheets/d/1bujiO2NgtHlgqPlNvYAQf5_7ZcXARlIfNX5HNb9f8cI
2. File → Download → Comma Separated Values (.csv)
3. Save as: `metadata.csv`

### Download Q3 Word List as CSV
1. Open: https://docs.google.com/spreadsheets/d/1J_I0raoRNbe29HiAPD5FROTr0jC93YtSkjOrIglKEjU
2. File → Download → Comma Separated Values (.csv)
3. Save as: `words.txt`  (one word per line)

---

## 🚀 Step 3 — Run Commands

---

### 🔵 Interactive Menu (easiest)
```bash
python run_all.py
```
Then type one of: `1name` / `2name` / `3name` / `4name` / `234` / `all`

---

### 🟢 Question 1 — Whisper Fine-Tuning

#### Full pipeline (download data + fine-tune + evaluate)
```bash
python question1_whisper_finetune.py --metadata metadata.csv
```

#### Skip fine-tuning (evaluate only, if model already saved)
```bash
python question1_whisper_finetune.py --metadata metadata.csv --skip_training
```

#### Using Google Sheet URL directly (if sheet is public)
```bash
python question1_whisper_finetune.py \
  --metadata "https://docs.google.com/spreadsheets/d/1bujiO2NgtHlgqPlNvYAQf5_7ZcXARlIfNX5HNb9f8cI/edit#gid=1786138861"
```

#### Via run_all.py
```bash
python run_all.py --q 1name --metadata metadata.csv
python run_all.py --q 1name --metadata metadata.csv --skip_training
```

**Outputs:**
```
q1_wer_results.csv       ← WER table (baseline vs fine-tuned)
q1_error_sample.csv      ← 30 stratified error utterances
whisper-small-hindi-ft/  ← saved fine-tuned model
```

---

### 🟡 Question 2 — ASR Cleanup Pipeline

#### Run with real GCS data (fetches automatically)
```bash
python question2_cleanup_pipeline.py
```

#### Via run_all.py
```bash
python run_all.py --q 2name
```

**What it does:**
- Fetches real segments from `825780_transcription.json`
- Runs number normalisation (with verb-दो fix)
- Runs English word detection and tagging
- Shows before/after for all examples

**Outputs:**
```
q2_pipeline_output.csv   ← raw | number_norm | english_tagged for all segments
```

---

### 🟠 Question 3 — Hindi Spell Checker

#### Run with downloaded word list CSV
```bash
python question3_spell_checker.py --word_list words.txt
```

#### Run with Google Sheet URL directly
```bash
python question3_spell_checker.py \
  --word_list "https://docs.google.com/spreadsheets/d/1J_I0raoRNbe29HiAPD5FROTr0jC93YtSkjOrIglKEjU/edit#gid=1432279672"
```

#### Run demo mode (uses real GCS data, no files needed)
```bash
python question3_spell_checker.py
```

#### With custom Hindi dictionary
```bash
python question3_spell_checker.py --word_list words.txt --dict hindi_dict.txt
```

#### Via run_all.py
```bash
python run_all.py --q 3name --word_list words.txt
python run_all.py --q 3name --word_list words.txt --dict hindi_dict.txt
```

**Outputs:**
```
q3_spelling_results.csv       ← 2 columns: word | spelling_status  (Google Sheets ready)
q3_detailed_results.csv       ← word | label | confidence | score | reason
q3_low_confidence_review.csv  ← 50 LOW-confidence words with manual review
```

> **Import to Google Sheets:**
> Google Sheets → File → Import → Upload → `q3_spelling_results.csv`
> Select: Comma separated, Replace current sheet → Import

---

### 🔴 Question 4 — Lattice-Based WER

#### Run with built-in real data examples
```bash
python question4_lattice_wer.py
```

#### Run with custom examples JSON
```bash
python question4_lattice_wer.py --examples_json my_examples.json
```

#### Change agreement threshold (default 0.6)
```bash
python question4_lattice_wer.py --threshold 0.5
```

#### Via run_all.py
```bash
python run_all.py --q 4name
python run_all.py --q 4name --threshold 0.5
```

**Outputs:**
```
q4_per_utterance_wer.csv   ← standard_wer vs lattice_wer per utterance per model
q4_model_wer_summary.csv   ← aggregated WER table per model
```

---

### 🟣 Run Multiple Questions Together

#### Q2 + Q3 + Q4 (no GPU needed, runs in minutes)
```bash
python run_all.py --q 234
```

#### Q2 + Q3 + Q4 with word list
```bash
python run_all.py --q 234 --word_list words.txt
```

#### All 4 questions
```bash
python run_all.py --q all --metadata metadata.csv --word_list words.txt
```

#### All 4, skip fine-tuning (use saved model)
```bash
python run_all.py --q all --metadata metadata.csv --word_list words.txt --skip_training
```

---

## 📊 All Output Files Summary

| File | Question | Description |
|---|---|---|
| `q1_wer_results.csv` | Q1 | WER table: baseline vs fine-tuned on FLEURS |
| `q1_error_sample.csv` | Q1 | 30 stratified error utterances |
| `q2_pipeline_output.csv` | Q2 | Cleanup pipeline output for all GCS segments |
| `q3_spelling_results.csv` | Q3 | **Google Sheets ready** — word + spelling_status |
| `q3_detailed_results.csv` | Q3 | word + label + confidence + score + reason |
| `q3_low_confidence_review.csv` | Q3 | 50-word manual review with accuracy stats |
| `q4_per_utterance_wer.csv` | Q4 | Standard vs lattice WER per utterance |
| `q4_model_wer_summary.csv` | Q4 | Aggregated WER table per model |
| `whisper-small-hindi-ft/` | Q1 | Saved fine-tuned Whisper-small model |

---

## 🐛 Common Errors & Fixes

### `ModuleNotFoundError: No module named 'transformers'`
```bash
pip install -r requirements.txt
```

### `CUDA out of memory` during Q1 fine-tuning
```bash
# Reduce batch size in question1_whisper_finetune.py
# Change: per_device_train_batch_size=8  →  per_device_train_batch_size=4
# Change: gradient_accumulation_steps=2  →  gradient_accumulation_steps=4
```

### `403 Forbidden` when loading Google Sheet
The sheet must be set to "Anyone with the link can view":
- Google Sheet → Share → Change to "Anyone with the link" → Viewer
- Then re-run the command

### `ConnectionError` fetching GCS audio
```bash
# Check connectivity
curl -I https://storage.googleapis.com/upload_goai/967179/825780_transcription.json
```

### Q3 shows only demo words (not 1.75 lakh)
You need to provide the word list explicitly:
```bash
python question3_spell_checker.py --word_list words.txt
# words.txt = downloaded CSV from the Q3 Google Sheet
```

---

## 📝 Real Data Format Reference

The transcription JSON at each GCS URL has this structure:
```json
[
  {
    "start": 0.11,
    "end": 14.42,
    "speaker_id": 245746,
    "text": "अब काफी अच्छा होता है क्योंकि उनकी जनसंख्या बहुत कम..."
  },
  {
    "start": 14.42,
    "end": 29.03,
    "speaker_id": 245746,
    "text": "अनुभव करके कुछ लिखना था तो वह तो बिना देखिए नहीं..."
  }
]
```

Each element = one utterance segment with timestamps.
The code reads `text` from each segment for transcription/training.

---

## ✅ Quick Verification (no GPU, 2 minutes)

Run this to verify everything works with real data before the full run:

```bash
# Step 1: Install
pip install requests pandas numpy tqdm

# Step 2: Run Q2 (fetches real GCS data, no GPU)
python question2_cleanup_pipeline.py

# Step 3: Run Q4 (pure computation, no downloads needed)
python question4_lattice_wer.py

# Step 4: Run Q3 demo (fetches real words from GCS)
python question3_spell_checker.py
```

All three should complete in under 2 minutes and produce `.csv` output files.
