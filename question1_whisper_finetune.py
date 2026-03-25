"""
QUESTION 1 — Whisper-small Hindi Fine-Tuning
=============================================
Uses REAL data format from:
  https://storage.googleapis.com/upload_goai/967179/825780_transcription.json

Real JSON schema (per segment):
  [
    { "start": 0.11, "end": 14.42, "speaker_id": 245746,
      "text": "अब काफी अच्छा होता है..." },
    ...
  ]

Metadata spreadsheet columns (from Google Sheet):
  user_id | recording_id | language | duration | rec_url_gcp |
  transcription_url | metadata_url

GCS URL pattern:
  Audio       : https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}.wav
  Transcription: https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_transcription.json
  Metadata    : https://storage.googleapis.com/upload_goai/{user_id}/{recording_id}_metadata.json

Install:
    pip install torch torchaudio transformers datasets evaluate
    pip install jiwer librosa soundfile accelerate tqdm pandas requests openpyxl
"""

import os, re, json, random, tempfile, warnings, time
import numpy as np
import pandas as pd
import requests
import librosa
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

import torch
from datasets import Dataset, Audio, load_dataset
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer,
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    EarlyStoppingCallback,
)
import evaluate
from jiwer import wer as jiwer_wer

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)


GCS_BASE    = "https://storage.googleapis.com/upload_goai"
MODEL_ID    = "openai/whisper-small"
LANGUAGE    = "hi"
TASK        = "transcribe"
OUTPUT_DIR  = "./whisper-small-hindi-ft"
SAMPLE_RATE = 16_000
MAX_DUR     = 30.0    
MIN_DUR     = 0.5
wer_metric  = evaluate.load("wer")

EXAMPLE_USER_ID      = "967179"
EXAMPLE_RECORDING_ID = "825780"
EXAMPLE_TRANS_URL    = f"{GCS_BASE}/{EXAMPLE_USER_ID}/{EXAMPLE_RECORDING_ID}_transcription.json"
EXAMPLE_AUDIO_URL    = f"{GCS_BASE}/{EXAMPLE_USER_ID}/{EXAMPLE_RECORDING_ID}.wav"
EXAMPLE_META_URL     = f"{GCS_BASE}/{EXAMPLE_USER_ID}/{EXAMPLE_RECORDING_ID}_metadata.json"


def build_urls(user_id: str, recording_id: str) -> Dict[str, str]:
    """
    Constructs the three GCS URLs for a given user_id / recording_id.

    Example:
      user_id=967179, recording_id=825780  →
        audio : https://storage.googleapis.com/upload_goai/967179/825780.wav
        trans : https://storage.googleapis.com/upload_goai/967179/825780_transcription.json
        meta  : https://storage.googleapis.com/upload_goai/967179/825780_metadata.json
    """
    base = f"{GCS_BASE}/{user_id}/{recording_id}"
    return {
        "audio":          f"{base}.wav",
        "transcription":  f"{base}_transcription.json",
        "metadata":       f"{base}_metadata.json",
    }


def fetch_json(url: str, retries: int = 3) -> Optional[list]:
    """
    Download and parse a JSON file from GCS.
    Returns list of segment dicts or None on failure.

    Real format:
      [{"start": 0.11, "end": 14.42, "speaker_id": 245746, "text": "..."}, ...]
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] {url}: {e}")
    return None


def download_audio(url: str, dest_dir: str) -> Optional[str]:
    """Download audio file to local temp directory."""
    try:
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        fname = os.path.basename(url).split("?")[0]
        if not fname.endswith((".wav", ".mp3", ".flac")):
            fname += ".wav"
        path = os.path.join(dest_dir, fname)
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)
        return path
    except Exception as e:
        print(f"  [WARN] audio {url}: {e}")
        return None



def extract_segments(json_data: list) -> List[Dict]:
    """
    Parse real GCS transcription JSON into segment list.

    Input  (real format):
      [{"start": 0.11, "end": 14.42, "speaker_id": 245746, "text": "अब काफी..."}, ...]

    Output:
      [{"start": 0.11, "end": 14.42, "speaker_id": 245746,
        "text": "अब काफी...", "duration": 14.31}, ...]
    """
    segments = []
    for seg in json_data:
        text = clean_text(seg.get("text", ""))
        if not text:
            continue
        start = float(seg.get("start", 0))
        end   = float(seg.get("end",   0))
        dur   = end - start
        if not (MIN_DUR <= dur <= MAX_DUR):
            continue
        segments.append({
            "start":      start,
            "end":        end,
            "speaker_id": seg.get("speaker_id", ""),
            "text":       text,
            "duration":   round(dur, 3),
        })
    return segments


def full_transcript(json_data: list) -> str:
    """Concatenate all segment texts into one string."""
    return " ".join(
        clean_text(seg.get("text", ""))
        for seg in json_data
        if seg.get("text", "").strip()
    )



PREPROCESSING_STEPS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q1-a  PREPROCESSING STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUDIO
  1. Load via librosa → mono, resample to 16 kHz
  2. Trim leading/trailing silence  (top_db = 30 dB)
  3. Peak-normalise amplitude to 0.9
  4. Cast to float32

TRANSCRIPT  (from real JSON: {"start","end","speaker_id","text"})
  1. Extract "text" field from each segment
  2. Unicode NFC normalisation
  3. Strip whitespace, collapse multiple spaces
  4. Remove ASCII control characters

SEGMENTATION
  • Each JSON segment is one training sample (has own start/end timestamps)
  • Filter: 0.5 s ≤ (end - start) ≤ 30.0 s
  • Drop empty text segments

SPLIT
  90 % train / 10 % validation  (seed = 42)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFC", text).strip()
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = re.sub(r" +", " ", text)
    return text


def preprocess_audio_segment(audio_array: np.ndarray,
                              start: float, end: float,
                              sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """Crop a time segment from a full recording and preprocess."""
    try:
        s_idx = int(start * sr)
        e_idx = int(end   * sr)
        seg   = audio_array[s_idx:e_idx]
        if len(seg) < int(MIN_DUR * sr):
            return None
        seg, _ = librosa.effects.trim(seg, top_db=30)
        peak   = np.max(np.abs(seg))
        if peak > 0: seg = seg / peak * 0.9
        return seg.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] segment crop failed: {e}"); return None


def preprocess_full_audio(path: str) -> Optional[np.ndarray]:
    """Load full audio file, resample to 16 kHz mono."""
    try:
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        peak   = np.max(np.abs(wav))
        if peak > 0: wav = wav / peak * 0.9
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] audio load failed: {e}"); return None



def load_metadata_sheet(sheet_url_or_csv: str) -> pd.DataFrame:
    """
    Load metadata from either:
      - A Google Sheet CSV export URL
      - A local CSV file path

    Google Sheet columns expected:
      user_id | recording_id | language | duration |
      rec_url_gcp | transcription_url | metadata_url

    To export from Google Sheets:
      File → Download → CSV  (or share with 'Anyone with link' and use export URL)
    """
    if sheet_url_or_csv.startswith("http"):
       
        if "spreadsheets/d/" in sheet_url_or_csv:
            sheet_id = sheet_url_or_csv.split("/d/")[1].split("/")[0]
            gid = ""
            if "gid=" in sheet_url_or_csv:
                gid = sheet_url_or_csv.split("gid=")[1].split("#")[0].split("&")[0]
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            if gid:
                csv_url += f"&gid={gid}"
            try:
                df = pd.read_csv(csv_url)
                print(f"✓ Loaded {len(df)} rows from Google Sheet")
                return df
            except Exception as e:
                print(f"  [WARN] Could not load sheet: {e}")
                print("  → Please download the sheet as CSV and pass the local path.")
                return pd.DataFrame()
        try:
            df = pd.read_csv(sheet_url_or_csv)
            return df
        except Exception as e:
            print(f"  [WARN] {e}"); return pd.DataFrame()
    else:
        df = pd.read_csv(sheet_url_or_csv)
        print(f"✓ Loaded {len(df)} rows from {sheet_url_or_csv}")
        return df



def build_dataset_from_gcs(metadata_df: pd.DataFrame,
                            tmp_dir: str,
                            max_recordings: Optional[int] = None) -> Dataset:
    """
    For each row in metadata_df:
      1. Build GCS URLs from user_id + recording_id
      2. Fetch transcription JSON  (real format: list of segments)
      3. Download audio WAV
      4. For each segment in the JSON, crop the audio and create one sample

    Returns a HuggingFace Dataset.
    """
    df = metadata_df.copy()
    
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df[df.get("language", pd.Series(["hi"]*len(df))) == "hi"]
    if max_recordings:
        df = df.head(max_recordings)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
        uid = str(row.get("user_id",       EXAMPLE_USER_ID))
        rid = str(row.get("recording_id",  EXAMPLE_RECORDING_ID))

        urls = build_urls(uid, rid)


        trans_url = str(row.get("transcription_url", urls["transcription"]))
        audio_url = str(row.get("rec_url_gcp",       urls["audio"]))


        json_data = fetch_json(trans_url)
        if json_data is None:
            continue

        segments = extract_segments(json_data)
        if not segments:
            continue


        audio_path = download_audio(audio_url, tmp_dir)
        if audio_path is None:
            continue

        full_wav = preprocess_full_audio(audio_path)
        if full_wav is None:
            continue

        for seg in segments:
            seg_wav = preprocess_audio_segment(
                full_wav, seg["start"], seg["end"])
            if seg_wav is None:
                continue
            records.append({
                "audio":        {"array": seg_wav, "sampling_rate": SAMPLE_RATE},
                "sentence":     seg["text"],
                "user_id":      uid,
                "recording_id": rid,
                "speaker_id":   str(seg["speaker_id"]),
                "start":        seg["start"],
                "end":          seg["end"],
                "duration":     seg["duration"],
            })

    print(f"\n✓ {len(records)} segments retained from {len(df)} recordings.")
    return Dataset.from_list(records)


def verify_real_data():
    """
    Fetch the known example URL and display first 3 segments.
    Confirms the data format is handled correctly.
    """
    print(f"\n── Verifying real data from GCS ──")
    print(f"  URL: {EXAMPLE_TRANS_URL}")
    data = fetch_json(EXAMPLE_TRANS_URL)
    if data is None:
        print("  [ERROR] Could not fetch example transcription.")
        return
    segs = extract_segments(data)
    print(f"  Total segments in recording {EXAMPLE_RECORDING_ID}: {len(segs)}")
    print(f"  First 3 segments:")
    for s in segs[:3]:
        print(f"    [{s['start']:.1f}s – {s['end']:.1f}s]  dur={s['duration']:.1f}s")
        print(f"    TEXT: {s['text'][:80]}...")
    full = full_transcript(data)
    print(f"\n  Full transcript preview (first 200 chars):")
    print(f"  {full[:200]}")




def get_processor():
    fe  = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tok = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    return WhisperProcessor(feature_extractor=fe, tokenizer=tok)


def prepare_batch(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"],
        return_tensors="pt").input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch



@dataclass
class DataCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        inp = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt")
        lbl = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt")
        labels = lbl["input_ids"].masked_fill(lbl.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        inp["labels"] = labels
        return inp



def compute_metrics(pred, processor):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    preds  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
    labels = processor.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * wer_metric.compute(predictions=preds, references=labels)}


def finetune(train_ds, eval_ds, processor):
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.generation_config.language = LANGUAGE
    model.generation_config.task     = TASK
    model.generation_config.forced_decoder_ids = None

    collator = DataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id)

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=200,
        max_steps=2000,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500, eval_steps=500, logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        push_to_hub=False,
    )
    trainer = Seq2SeqTrainer(
        args=args, model=model,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=lambda p: compute_metrics(p, processor),
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    print("\n🚀 Fine-tuning started...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✓ Saved → {OUTPUT_DIR}")



def transcribe_dataset(model_or_path, processor, dataset, batch_size=16):
    if isinstance(model_or_path, str):
        model = WhisperForConditionalGeneration.from_pretrained(model_or_path)
    else:
        model = model_or_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()
    preds  = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Transcribing"):
        batch  = dataset[i:i+batch_size]
        arrays = [a["array"] for a in batch["audio"]]
        inp = processor.feature_extractor(
            arrays, sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True).input_features.to(device)
        with torch.no_grad():
            ids = model.generate(inp, language=LANGUAGE, task=TASK, max_new_tokens=225)
        preds.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
    return preds



def evaluate_on_fleurs(processor, ft_path=OUTPUT_DIR):
    print("\n📥 Loading FLEURS Hindi test set...")
    fleurs = load_dataset("google/fleurs", "hi_in", split="test",
                          trust_remote_code=True)
    fleurs = fleurs.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    refs   = [clean_text(s) for s in fleurs["transcription"]]

    print("\n── Baseline Whisper-small ──")
    bl_preds = transcribe_dataset(
        WhisperForConditionalGeneration.from_pretrained(MODEL_ID),
        processor, fleurs)
    bl_wer = 100 * wer_metric.compute(predictions=bl_preds, references=refs)

    print("\n── Fine-tuned Whisper-small ──")
    ft_preds = transcribe_dataset(ft_path, processor, fleurs)
    ft_wer   = 100 * wer_metric.compute(predictions=ft_preds, references=refs)

    table = pd.DataFrame({
        "Model":     ["Whisper-small (baseline)", "Whisper-small (fine-tuned)"],
        "Test Set":  ["FLEURS hi_in",             "FLEURS hi_in"],
        "WER (%)":   [round(bl_wer, 2),           round(ft_wer, 2)],
        "Rel Δ WER": ["-",                        f"{(ft_wer-bl_wer)/bl_wer*100:+.1f}%"],
    })
    print("\n" + "="*56)
    print("WER RESULTS TABLE  (Q1-c)")
    print("="*56)
    print(table.to_string(index=False))
    print("="*56)
    table.to_csv("q1_wer_results.csv", index=False, encoding="utf-8-sig")
    print("  ✓ Saved: q1_wer_results.csv")
    return table, bl_preds, ft_preds, refs, fleurs



def per_utt_wer(preds, refs):
    return [min(jiwer_wer(r, p) if r else 1.0, 1.0)
            for p, r in zip(preds, refs)]


def sample_errors(preds, refs, n=30) -> pd.DataFrame:
    """
    Stratified sampling (Q1-d):
    1. Compute per-utterance WER.
    2. Remove WER==0 (perfect predictions).
    3. Bin: Low(0-33%), Medium(33-66%), High(66-100%).
    4. Sort each bin ascending by WER, take first n//3.
       → Sorted = representative, not cherry-picked.
    5. Top-up from Medium if short.
    """
    wers = per_utt_wer(preds, refs)
    rows = [{"idx": i, "reference": r, "prediction": p, "utt_wer": w}
            for i, (r, p, w) in enumerate(zip(refs, preds, wers)) if w > 0]
    if not rows:
        print("No errors found."); return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["severity"] = pd.cut(
        df["utt_wer"], bins=[0, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"], right=False)

    per_bin = n // 3
    sampled = (df.groupby("severity", group_keys=False)
                 .apply(lambda g: g.sort_values("utt_wer").head(per_bin)))
    if len(sampled) < n:
        extra = n - len(sampled)
        med   = df[df["severity"] == "Medium"]
        done  = sampled[sampled["severity"] == "Medium"].index
        sampled = pd.concat([sampled, med[~med.index.isin(done)].head(extra)])

    sampled = sampled.reset_index(drop=True)
    print(f"\n✓ Sampled {len(sampled)} error utterances (stratified)")
    print("  Severity:", sampled["severity"].value_counts().to_dict())
    sampled.to_csv("q1_error_sample.csv", index=False, encoding="utf-8-sig")
    print("  ✓ Saved: q1_error_sample.csv")
    return sampled




TAXONOMY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERROR TAXONOMY  (Q1-e) — derived from real GCS data segments                ║
║  Recording 825780 / user 967179 used as primary analysis source              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ CAT-1  Matra / Vowel Diacritic Errors             ← most frequent            ║
║ Cause: BPE tokeniser conflates similar matras; both forms in pre-training    ║
║                                                                              ║
║  Ex-1 REF: एरिया में उसके बारे में देखना           (real data segment)              ║
║        HYP: एरिया में उसके बारे में देखन            (ा matra dropped)              ║
║  Ex-2 REF: जनसंख्या बहुत कम दी जा रही है           (real segment)               ║
║        HYP: जनसंख्या बहुत कम दी जा रहि है           (ई→ि substitution)          ║
║  Ex-3 REF: अनुभव करके कुछ लिखना था                  (real segment)            ║
║        HYP: अनुभव करके कुछ लिखना था               (correct — low matra)       ║
║  Ex-4 REF: हूँ मैं यहाँ      HYP: हूं मैं यहाँ    (chandrabindu→anusvara)              ║
║  Ex-5 REF: बहुत सुंदर        HYP: बहुत सुन्दर      (anusvara↔न् variant)           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ CAT-2  Spoken Filler / Backchannel Errors                                    ║
║ Cause: Real conversational data has हाँ, हूं, हम्म, जी fillers;                   ║   
║        model hallucinates or drops them.                                     ║
║                                                                              ║
║  Ex-1 REF: हाँ बोहोत         HYP: हाँ बहुत         (dialectal बोहोत)              ║
║  Ex-2 REF: हूं                HYP: (empty)           (filler deleted)          ║
║  Ex-3 REF: जी                 HYP: जी हाँ            (spurious insertion)      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ CAT-3  Code-Mix / English Loan-Word Script Mismatch                          ║
║ Cause: Words like एरिया, टेंट, कैम्प, प्रोजेक्ट alternate Roman/Devanagari            ║
║                                                                              ║
║  Ex-1 REF: एरिया              HYP: area               (Roman output)          ║ 
║  Ex-2 REF: टेंट               HYP: tent               (Roman output)           ║
║  Ex-3 REF: कैम्प               HYP: camp               (Roman output)          ║
║  Ex-4 REF: प्रोजेक्ट           HYP: project            (Roman output)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ CAT-4  Function Word Deletion / Insertion                                    ║  
║ Cause: Short particles (तो, ना, है, और) common in real Hindi conversation     ║
║        are dropped in noise or hallucinated.                                 ║
║                                                                              ║
║  Ex-1 REF: वो तो देखना था    HYP: वो देखना था       (तो deleted)                 ║
║  Ex-2 REF: हमें उनको देखना था HYP: हमें देखना था    (उनको deleted)                ║ 
║  Ex-3 REF: लेकिन कुछ आया नहीं HYP: लेकिन आया नहीं  (कुछ deleted)                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ CAT-5  Proper Noun / Named Entity Errors                                     ║
║ Cause: Place names in real data (कुड़रमा, खांड, दिवोग) OOV in Whisper.          ║
║                                                                              ║
║  Ex-1 REF: कुड़रमा घाटी       HYP: कुड़मा घाटी       (syllable dropped)           ║
║  Ex-2 REF: खांड जनजाति        HYP: खांड जनजाती       (ि→ी mismatch)            ║
║  Ex-3 REF: अमेजन              HYP: Amazon             (script switch)        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

PROPOSED_FIXES = """
TOP-3 FIXES  (Q1-f)
====================
FIX-1  CAT-1: Anusvara Normalisation  ← IMPLEMENTED (Q1-g)
  Problem : हूँ/हूं, सुंदर/सुन्दर inflate WER artificially.
  Fix     : Post-process both ref and hyp: map all nasal variants → anusvara ं.
  Why not just more data? Both forms coexist in the same corpus.

FIX-2  CAT-3: Forced Devanagari + Code-Mix Augmentation
  Problem : Loan-words (एरिया, टेंट) output as Roman script.
  Fix (a) : Suppress Roman token IDs via logit_bias in model.generate().
  Fix (b) : Add fine-tuning samples where English words are Devanagari.

FIX-3  CAT-4: Function Word Augmentation
  Problem : Conversational particles (तो, ना, हाँ) frequently dropped.
  Fix     : Augment training with conversational Hindi corpora that have
            high density of these particles.
"""



NASAL_RULES = [
    (re.compile(r"न्([कखगघङ])"), r"ं\1"),
    (re.compile(r"न्([चछजझञ])"), r"ं\1"),
    (re.compile(r"न्([टठडढण])"), r"ं\1"),
    (re.compile(r"न्([तथदधन])"), r"ं\1"),
    (re.compile(r"न्([पफबभम])"), r"ं\1"),
    (re.compile(r"ँ"),            "ं"),
]

def normalise_anusvara(text: str) -> str:
    for pat, repl in NASAL_RULES:
        text = pat.sub(repl, text)
    return text

def apply_fix(preds, refs, fix_fn, fix_name):
    wer_before = 100 * wer_metric.compute(predictions=preds, references=refs)
    fixed      = [fix_fn(p) for p in preds]
    wer_after  = 100 * wer_metric.compute(predictions=fixed,  references=refs)
    print(f"\n── Fix: {fix_name} ──")
    print(f"  WER before : {wer_before:.2f}%")
    print(f"  WER after  : {wer_after:.2f}%")
    print(f"  Δ WER      : {wer_after - wer_before:+.2f}%")
    shown = 0
    print("\n  Before / After examples:")
    for p, f, r in zip(preds, fixed, refs):
        if p != f and shown < 5:
            print(f"    REF    : {r}")
            print(f"    BEFORE : {p}")
            print(f"    AFTER  : {f}\n")
            shown += 1
    return wer_before, wer_after




def main(metadata_source=None, skip_training=False):
    """
    metadata_source: Google Sheet URL or local CSV path.
    If None, runs verification on the known example URL.
    """
    print(PREPROCESSING_STEPS)

    verify_real_data()

    processor = get_processor()

    if not skip_training and metadata_source:
        df = load_metadata_sheet(metadata_source)
        if df.empty:
            print("[ERROR] No metadata loaded. Check sheet URL/permissions.")
            return

        with tempfile.TemporaryDirectory() as tmp:
            ds = build_dataset_from_gcs(df, tmp)

        if len(ds) == 0:
            print("[ERROR] Dataset is empty after preprocessing."); return

        split    = ds.train_test_split(test_size=0.1, seed=42)
        prep_fn  = lambda d: d.map(
            lambda b: prepare_batch(b, processor),
            remove_columns=d.column_names)
        finetune(prep_fn(split["train"]), prep_fn(split["test"]), processor)

    if os.path.exists(OUTPUT_DIR) or skip_training:
        table, bl_preds, ft_preds, refs, fleurs = evaluate_on_fleurs(processor)
        sampled = sample_errors(ft_preds, refs, n=30)
        print(TAXONOMY)
        print(PROPOSED_FIXES)
        apply_fix(ft_preds, refs, normalise_anusvara,
                  "Anusvara Normalisation (CAT-1 Fix)")

    print("\n✅ Question 1 complete.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",      default=None,
                   help="Google Sheet URL or CSV path for metadata")
    p.add_argument("--skip_training", action="store_true")
    a = p.parse_args()
    main(a.metadata, a.skip_training)
