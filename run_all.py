"""
run_all.py — Unified Runner for All 4 Questions
================================================
Real data URLs used throughout:
  Transcription : https://storage.googleapis.com/upload_goai/967179/825780_transcription.json
  Audio         : https://storage.googleapis.com/upload_goai/967179/825780.wav
  Metadata      : https://storage.googleapis.com/upload_goai/967179/825780_metadata.json

Google Sheets (provide via --metadata / --word_list):
  Dataset metadata : https://docs.google.com/spreadsheets/d/1bujiO2NgtHlgqPlNvYAQf5_7ZcXARlIfNX5HNb9f8cI
  Q1 metadata      : https://docs.google.com/spreadsheets/d/1JItJnilmmSWjx9tAIr06cbTsyGjMMxEMhaebvn5qBHM
  Q3 word list     : https://docs.google.com/spreadsheets/d/1J_I0raoRNbe29HiAPD5FROTr0jC93YtSkjOrIglKEjU

Usage:
    python run_all.py               # interactive menu (choose 1name/2name/3name/4name)
    python run_all.py --q 1name     # Q1 only
    python run_all.py --q 2name     # Q2 only
    python run_all.py --q 3name     # Q3 only
    python run_all.py --q 4name     # Q4 only
    python run_all.py --q 234       # Q2+Q3+Q4 (no GPU needed)
    python run_all.py --q all       # all 4

Optional flags:
    --metadata   <Google Sheet URL or CSV>   for Q1 dataset metadata
    --word_list  <Google Sheet URL or CSV>   for Q3 word list
    --skip_training                          skip Q1 fine-tuning
    --threshold  0.6                         lattice agreement threshold (Q4)

File layout:
    1name: question1_whisper_finetune.py
    2name: question2_cleanup_pipeline.py
    3name: question3_spell_checker.py
    4name: question4_lattice_wer.py
"""

import sys, argparse

MENU = """
╔══════════════════════════════════════════════════════════════════╗
║      Josh Talks ASR Assignment — Question Runner                ║
╠══════════════════════════════════════════════════════════════════╣
║  1name  →  Q1: Whisper Fine-Tuning + WER + Error Taxonomy      ║
║  2name  →  Q2: Number Normalisation + English Word Tagging      ║
║  3name  →  Q3: Hindi Spell Checker (1.75 lakh words)            ║
║  4name  →  Q4: Lattice-Based WER Evaluation                     ║
║  234    →  Q2 + Q3 + Q4 combined (no GPU required)             ║
║  all    →  All 4 questions                                      ║
╚══════════════════════════════════════════════════════════════════╝

Real data:  https://storage.googleapis.com/upload_goai/967179/825780_transcription.json
"""

NAME_MAP = {
    "1name":"1", "2name":"2", "3name":"3", "4name":"4",
    "1":"1",     "2":"2",     "3":"3",     "4":"4",
    "234":"234", "all":"1234",
}


def interactive_menu() -> str:
    print(MENU)
    while True:
        choice = input("Enter choice (1name / 2name / 3name / 4name / 234 / all): ").strip()
        mapped = NAME_MAP.get(choice.lower())
        if mapped: return mapped
        print(f"  Invalid: '{choice}'. Try again.")


def run_q1(args):
    print("\n" + "▓"*65)
    print("▓  1name: Q1 — Whisper-small Fine-Tuning Pipeline")
    print("▓"*65)
    from question1_whisper_finetune import main as q1
    q1(metadata_source=args.metadata, skip_training=args.skip_training)


def run_q2(_args):
    print("\n" + "▓"*65)
    print("▓  2name: Q2 — ASR Cleanup (Number Norm + English Tagging)")
    print("▓"*65)
    from question2_cleanup_pipeline import main as q2
    q2()


def run_q3(args):
    print("\n" + "▓"*65)
    print("▓  3name: Q3 — Hindi Spell Checker")
    print("▓"*65)
    from question3_spell_checker import main as q3
    q3(word_source=args.word_list, dict_path=args.dict)


def run_q4(args):
    print("\n" + "▓"*65)
    print("▓  4name: Q4 — Lattice-Based WER")
    print("▓"*65)
    from question4_lattice_wer import main as q4
    q4(examples=None, threshold=args.threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Josh Talks ASR Assignment — unified runner",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--q",            default=None,
        help="1name | 2name | 3name | 4name | 234 | all")
    parser.add_argument("--metadata",     default=None,
        help="Google Sheet URL or CSV path for Q1 metadata")
    parser.add_argument("--word_list",    default=None,
        help="Google Sheet URL or .txt path for Q3 word list")
    parser.add_argument("--dict",         default=None,
        help="Hindi dictionary file for Q3")
    parser.add_argument("--skip_training",action="store_true",
        help="Skip Q1 fine-tuning (use already-saved model)")
    parser.add_argument("--threshold",    type=float, default=0.6,
        help="Model agreement threshold for Q4 (default 0.6)")

    args = parser.parse_args()

    qs = NAME_MAP.get((args.q or "").lower(), args.q)
    if not qs:
        qs = interactive_menu()

    print(f"\n  Running: {qs}\n")
    if "1" in qs: run_q1(args)
    if "2" in qs: run_q2(args)
    if "3" in qs: run_q3(args)
    if "4" in qs: run_q4(args)

    print("\n" + "="*65)
    print("  ALL SELECTED QUESTIONS COMPLETE ✅")
    print("="*65)


if __name__ == "__main__":
    main()
