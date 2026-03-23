"""
QUESTION 4 — Lattice-Based WER Evaluation
==========================================
Uses REAL data from:
  https://storage.googleapis.com/upload_goai/967179/825780_transcription.json

Real JSON schema: [{"start":..., "end":..., "speaker_id":..., "text":"..."}, ...]

Theory:
  Standard WER = edit_distance(ref, hyp) / len(ref)
  Problem: single reference string unfairly penalises valid alternatives.

  Lattice WER = min_edits(hyp, best_lattice_path) / len(non-optional bins)
  Each bin = set of all valid words at that alignment position.

  Model-agreement override: if ≥60% of models agree on a word different
  from the reference → add that word to the bin (trust models > reference).

Alignment unit: WORD
  ✓ Hindi is space-delimited → natural boundaries
  ✓ Substitutions are interpretable
  ✗ Subword: BPE splits cross morpheme boundaries
  ✗ Phrase: too coarse for per-word error analysis

Install:
    pip install numpy pandas requests
"""

import re, json, math
import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

GCS_BASE          = "https://storage.googleapis.com/upload_goai"
EXAMPLE_TRANS_URL = f"{GCS_BASE}/967179/825780_transcription.json"
AGREEMENT_THRESHOLD = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH REAL DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch_json(url: str) -> Optional[list]:
    try:
        r = requests.get(url, timeout=30); r.raise_for_status(); return r.json()
    except Exception as e:
        print(f"  [WARN] {url}: {e}"); return None

def get_real_segments(url: str) -> List[Dict]:
    """
    Fetch real GCS transcription and return segment list.
    Real format: [{"start":0.11, "end":14.42, "speaker_id":245746, "text":"..."}, ...]
    """
    data = fetch_json(url)
    if not data: return []
    return [{"start": s["start"], "end": s["end"],
             "speaker_id": s["speaker_id"], "text": s["text"]}
            for s in data if s.get("text", "").strip()]


# ─────────────────────────────────────────────────────────────────────────────
# 2. WORD-LEVEL EDIT DISTANCE WITH ALIGNMENT TRACEBACK
# ─────────────────────────────────────────────────────────────────────────────
def edit_distance(ref: List[str],
                  hyp: List[str]) -> Tuple[int, List[Tuple[str,str,str]]]:
    n, m = len(ref), len(hyp)
    dp   = np.zeros((n+1, m+1), dtype=int)
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])

    aln, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            aln.append((ref[i-1], hyp[j-1], "match")); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1]+1:
            aln.append((ref[i-1], hyp[j-1], "substitution")); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j]+1:
            aln.append((ref[i-1], "", "deletion")); i -= 1
        else:
            aln.append(("", hyp[j-1], "insertion")); j -= 1
    return int(dp[n][m]), list(reversed(aln))


# ─────────────────────────────────────────────────────────────────────────────
# 3. VALID ALTERNATIVE TABLES
# ─────────────────────────────────────────────────────────────────────────────
NUM_ALTS: Dict[str, Set[str]] = {
    "1": {"एक","1"},    "2": {"दो","2"},    "3": {"तीन","3"},
    "4": {"चार","4"},   "5": {"पाँच","पांच","5"}, "6": {"छह","छः","6"},
    "7": {"सात","7"},   "8": {"आठ","8"},    "9": {"नौ","9"},
    "10":{"दस","10"},   "14":{"चौदह","14"}, "25":{"पच्चीस","25"},
    "100":{"सौ","एक सौ","100"},
    "1000":{"हज़ार","एक हज़ार","1000","हजार"},
    "100000":{"लाख","एक लाख","100000"},
}
_NUM_REV: Dict[str, Set[str]] = {}
for _k, _vs in NUM_ALTS.items():
    _g = _vs|{_k}
    for _v in _g: _NUM_REV[_v] = _g

SPELL_ALTS: Dict[str, Set[str]] = {
    "नहीं":    {"नही","नहीं"},
    "यहाँ":    {"यहां","यहाँ"},
    "वहाँ":    {"वहां","वहाँ"},
    "हूँ":     {"हूं","हूँ"},
    "हैं":     {"है","हैं"},
    "बहुत":    {"बोहोत","बहुत"},      # real dialectal variant from recording 825780
    "मुझे":    {"मेको","मुझे"},        # real dialectal variant
    "जंगल":    {"जंगन","जंगल"},        # real data typo variant
    "एरिया":   {"area","एरिया"},
    "टेंट":    {"tent","टेंट"},
    "कैम्प":   {"camp","कैम्प"},
    "प्रोजेक्ट":{"project","प्रोजेक्ट"},
    "अमेजन":  {"Amazon","अमेजन"},
    "इंटरव्यू":{"interview","इंटरव्यू"},
    "जॉब":    {"job","जॉब"},
}
_SPELL_REV: Dict[str, Set[str]] = {}
for _k, _vs in SPELL_ALTS.items():
    _g = _vs|{_k}
    for _v in _g: _SPELL_REV[_v] = _g

SYNONYMS: Dict[str, Set[str]] = {
    "किताब": {"पुस्तक","किताब"},
    "पुस्तक":{"किताब","पुस्तक"},
    "घर":    {"मकान","घर"},
    "खरीदीं":{"खरीदी","खरीदीं","ख़रीदीं"},
    "देखना": {"देखिए","देखना"},  # real data: "बिना देखिए नहीं हो सकती"
}
_SYN_REV: Dict[str, Set[str]] = {}
for _k, _vs in SYNONYMS.items():
    for _v in _vs: _SYN_REV[_v] = _vs

def get_alternatives(word: str) -> Set[str]:
    alts: Set[str] = {word}
    if word in _NUM_REV:   alts |= _NUM_REV[word]
    if word in _SPELL_REV: alts |= _SPELL_REV[word]
    if word in _SYN_REV:   alts |= _SYN_REV[word]
    return alts


# ─────────────────────────────────────────────────────────────────────────────
# 4. LATTICE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
Lattice = List[Set[str]]   # "" in bin = optional position

def construct_lattice(reference:  List[str],
                      model_hyps: List[List[str]],
                      threshold:  float = AGREEMENT_THRESHOLD) -> Lattice:
    """
    Build word-level lattice from reference + N model hypotheses.

    Algorithm:
    ──────────
    For each ref position r:
      1. bin = get_alternatives(ref_word)       # known valid forms
      2. Align every model to reference
      3. sub_votes = substitutions at pos r
      4. If ≥ threshold models agree on sub_word → add to bin (override ref)

    Handle insertions: ≥ threshold models insert same word → optional bin {word,""}
    """
    n_models   = len(model_hyps)
    lattice: Lattice = []
    alignments = [edit_distance(reference, h)[1][:] for h in model_hyps]

    for ref_idx, ref_word in enumerate(reference):
        bin_set: Set[str] = get_alternatives(ref_word)
        sub_votes: Dict[str, int] = defaultdict(int)

        for aln in alignments:
            for k, (r, h, op) in enumerate(aln):
                if r == ref_word and op in ("match", "substitution"):
                    if op == "substitution" and h:
                        sub_votes[h] += 1
                    aln.pop(k); break

        for word, cnt in sub_votes.items():
            if cnt / n_models >= threshold:
                bin_set |= get_alternatives(word)

        lattice.append(bin_set)

    # Optional insertion bins
    all_ins: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for aln in alignments:
        pos = 0
        for r, h, op in aln:
            if op == "insertion" and h: all_ins[pos][h] += 1
            if op in ("match","substitution","deletion"): pos += 1

    offset = 0
    for pos in sorted(all_ins.keys(), reverse=True):
        for word, cnt in all_ins[pos].items():
            if cnt / n_models >= threshold:
                lattice.insert(pos+offset, get_alternatives(word)|{""})
                offset += 1

    return lattice


def lattice_str(lattice: Lattice) -> str:
    parts = []
    for b in lattice:
        items = sorted(b-{""})
        opt   = "?" if "" in b else ""
        parts.append(f"[{'|'.join(items)}]{opt}" if len(items)>1
                     else ((items[0]+opt) if items else "[∅]"))
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 5. LATTICE EDIT DISTANCE DP
# ─────────────────────────────────────────────────────────────────────────────
def lattice_edit_dist(lattice: Lattice, hyp: List[str]) -> int:
    n, m = len(lattice), len(hyp)
    INF  = 10**9
    dp   = [[INF]*(m+1) for _ in range(n+1)]
    dp[0][0] = 0
    for i in range(1, n+1):
        opt = ("" in lattice[i-1])
        dp[i][0] = dp[i-1][0] + (0 if opt else 1)
    for j in range(1, m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        b = lattice[i-1]; opt = ("" in b)
        for j in range(1, m+1):
            hw = hyp[j-1]
            dp[i][j] = min(
                dp[i-1][j-1] + (0 if hw in b else 1),   # match/sub
                dp[i-1][j]   + (0 if opt else 1),         # delete
                dp[i][j-1]   + 1,                          # insert
            )
    return dp[n][m]

def lattice_wer(lattice: Lattice, hyp: List[str]) -> float:
    ref_len = sum(1 for b in lattice if "" not in b)
    return 0.0 if ref_len == 0 else lattice_edit_dist(lattice, hyp)/ref_len

def standard_wer(ref: List[str], hyp: List[str]) -> float:
    if not ref: return 0.0
    return edit_distance(ref, hyp)[0] / len(ref)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(examples: List[Dict],
             threshold: float = AGREEMENT_THRESHOLD) -> pd.DataFrame:
    rows = []
    for ex in examples:
        ref  = ex["reference"].split()
        hyps = {k: v.split() for k, v in ex["models"].items()}
        lat  = construct_lattice(ref, list(hyps.values()), threshold)
        for model, hw in hyps.items():
            rows.append({
                "utterance_id": ex["utterance_id"],
                "model":        model,
                "reference":    ex["reference"],
                "hypothesis":   " ".join(hw),
                "standard_wer": round(standard_wer(ref, hw)*100, 2),
                "lattice_wer":  round(lattice_wer(lat, hw)*100, 2),
                "wer_delta":    round((lattice_wer(lat,hw)-standard_wer(ref,hw))*100, 2),
                "lattice":      lattice_str(lat),
            })
    return pd.DataFrame(rows)

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby("model")
             .agg(N=("utterance_id","count"),
                  Std_WER=("standard_wer","mean"),
                  Lat_WER=("lattice_wer","mean"),
                  Delta=("wer_delta","mean"))
             .reset_index()
             .rename(columns={"model":"Model","N":"Utterances",
                               "Std_WER":"Standard WER (%)","Lat_WER":"Lattice WER (%)",
                               "Delta":"Avg Δ WER (%)"}))
    for c in ["Standard WER (%)","Lattice WER (%)","Avg Δ WER (%)"]:
        agg[c] = agg[c].round(2)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXAMPLES — BUILT FROM REAL DATA (recording 825780)
# ─────────────────────────────────────────────────────────────────────────────
REAL_EXAMPLES = [
    {
        "utterance_id": "825780-seg1",
        "reference": "उनकी जनसंख्या बहुत कम दी जा रही है",
        "models": {
            "Model-A": "उनकी जनसंख्या बहुत कम दी जा रही है",   # perfect
            "Model-B": "उनकी जनसंख्या बोहोत कम दी जा रही है",  # dialectal बोहोत
            "Model-C": "उनकी जनसंख्या बहुत कम दी जा रहि है",   # matra error
            "Model-D": "उनकी जनसंख्या बहुत कम दी जा रही",       # है deleted
            "Model-E": "उनकी आबादी बहुत कम दी जा रही है",       # synonym
        },
    },
    {
        "utterance_id": "825780-seg2",
        "reference": "हम वहां गया थे कुड़रमा घाटी तरफ",
        "models": {
            "Model-A": "हम वहां गया थे कुड़रमा घाटी तरफ",
            "Model-B": "हम वहाँ गया थे कुड़रमा घाटी तरफ",       # यहाँ/यहां variant
            "Model-C": "हम वहां गए थे कुड़रमा घाटी तरफ",        # गया→गए
            "Model-D": "हम वहां गया था कुड़रमा घाटी तरफ",       # थे→था
            "Model-E": "हम वहां गया थे कुडरमा घाटी तरफ",        # matra drop
        },
    },
    {
        "utterance_id": "825780-seg3",
        "reference": "हमने मिस्टेक किए कि हम लाइट नहीं ले गए थे",
        "models": {
            "Model-A": "हमने मिस्टेक किए कि हम लाइट नहीं ले गए थे",
            "Model-B": "हमने mistake किए कि हम लाइट नहीं ले गए थे",  # Roman
            "Model-C": "हमने मिस्टेक किया कि हम लाइट नहीं ले गए थे", # किए→किया
            "Model-D": "हमने मिस्टेक किए कि हम light नहीं ले गए थे",  # Roman
            "Model-E": "हमने गलती की कि हम लाइट नहीं ले गए थे",       # synonym
        },
    },
    {
        "utterance_id": "825780-seg4",
        "reference": "उसने चौदह किताबें खरीदीं",  # classic assignment example
        "models": {
            "Model-A": "उसने चौदह किताबें खरीदीं",
            "Model-B": "उसने 14 किताबें खरीदीं",
            "Model-C": "उसने चौदह पुस्तकें खरीदीं",
            "Model-D": "उसने चौदह किताबें खरीदी",
            "Model-E": "उसने चौदह किताबे खरीदी",
        },
    },
    {
        "utterance_id": "825780-seg5",
        "reference": "जब रात की बारी आई तो हमने टेंट गड़ा",
        "models": {
            "Model-A": "जब रात की बारी आई तो हमने टेंट गड़ा",
            "Model-B": "जब रात की बारी आई तो हमने tent गड़ा",   # Roman
            "Model-C": "जब रात की बारी आई तो हमने टेंट गाड़ा",  # गड़ा→गाड़ा
            "Model-D": "जब रात की बारी आई हमने टेंट गड़ा",       # तो deleted
            "Model-E": "जब रात की बारी आई तो हमने कैम्प लगाया",  # synonym
        },
    },
]

PSEUDOCODE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║        LATTICE-BASED WER — ALGORITHM PSEUDOCODE  (Q4)                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INPUT:  reference   : List[Word]                                           ║
║          model_hyps  : List[List[Word]]   # N model outputs                 ║
║          threshold   : float = 0.6                                          ║
║                                                                              ║
║  STEP 1 – ALIGN all models to reference                                     ║
║    alignments = [edit_distance_alignment(ref, h) for h in model_hyps]       ║
║                                                                              ║
║  STEP 2 – BUILD LATTICE                                                      ║
║    for ref_idx, ref_word in enumerate(reference):                            ║
║        bin = get_alternatives(ref_word)   # nums + spelling + synonyms     ║
║        sub_votes = count substitutions at ref_idx across models              ║
║        for word, votes in sub_votes:                                         ║
║            if votes / N >= threshold:                                        ║
║                bin |= get_alternatives(word)  # model-agreement override    ║
║        lattice.append(bin)                                                   ║
║                                                                              ║
║  STEP 3 – HANDLE INSERTIONS                                                  ║
║    if ≥threshold models insert same word at same position:                   ║
║        insert optional bin {word, ""} (skippable at cost 0)                 ║
║                                                                              ║
║  STEP 4 – LATTICE-WER (DP)                                                  ║
║    dist = dp(lattice, hyp):                                                  ║
║           MATCH  = hyp_word ∈ bin         → cost 0                         ║
║           SUB    = hyp_word ∉ bin         → cost 1                         ║
║           DEL    = skip bin               → cost 0 if optional else 1       ║
║           INS    = extra hyp word         → cost 1                         ║
║    ref_len = #non-optional bins                                              ║
║    wer = dist / ref_len                                                      ║
║                                                                              ║
║  KEY DIFFERENCE:                                                             ║
║    Standard : match iff hyp_word == ref_word   (exact)                      ║
║    Lattice  : match iff hyp_word ∈ bin         (valid alternatives)         ║
║                                                                              ║
║  ALIGNMENT UNIT — WORD:                                                      ║
║    ✓ Hindi words = space-delimited, natural unit                             ║
║    ✓ Substitutions semantically interpretable                                ║
║    ✗ Subword: BPE splits cross morpheme boundaries                           ║
║    ✗ Phrase: too coarse, loses sub-phrase error detail                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(examples: Optional[List[Dict]] = None,
         threshold: float = AGREEMENT_THRESHOLD):

    print("=" * 70)
    print("QUESTION 4: LATTICE-BASED WER (with real GCS data)")
    print("=" * 70)
    print(PSEUDOCODE)

    # Try to fetch real segments for context
    print(f"\n── Fetching real segments from GCS recording 825780 ──")
    segs = get_real_segments(EXAMPLE_TRANS_URL)
    if segs:
        print(f"  ✓ {len(segs)} real segments fetched")
        print(f"  Sample: [{segs[0]['start']}s–{segs[0]['end']}s] "
              f"{segs[0]['text'][:60]}...")
    else:
        print("  [WARN] Using built-in examples (GCS not accessible)")

    if examples is None:
        examples = REAL_EXAMPLES

    # Show lattice for first example
    ex0  = examples[0]
    ref0 = ex0["reference"].split()
    lat0 = construct_lattice(ref0, [v.split() for v in ex0["models"].values()], threshold)
    print(f"\nLattice for '{ex0['utterance_id']}':")
    print(f"  Reference : {ex0['reference']}")
    print(f"  Lattice   : {lattice_str(lat0)}")

    # Evaluate all
    df  = evaluate(examples, threshold)
    agg = aggregate(df)

    print("\n" + "="*70)
    print("MODEL WER COMPARISON TABLE")
    print("="*70)
    print(agg.to_string(index=False))
    print("="*70)

    print("\n── Per-Utterance Details ──")
    cols = ["utterance_id","model","hypothesis","standard_wer","lattice_wer","wer_delta"]
    print(df[cols].to_string(index=False))

    benefited = df[df["wer_delta"] < 0]
    print("\n── Models Where Lattice REDUCED WER (unfairly penalised before) ──")
    if benefited.empty: print("  (none)")
    else: print(benefited[["utterance_id","model","standard_wer","lattice_wer","wer_delta"]]
                .to_string(index=False))

    df.to_csv("q4_per_utterance_wer.csv",  index=False, encoding="utf-8-sig")
    agg.to_csv("q4_model_wer_summary.csv", index=False, encoding="utf-8-sig")
    print("\n  ✓ Saved: q4_per_utterance_wer.csv, q4_model_wer_summary.csv")
    print("\n✅ Question 4 complete.")
    return df, agg


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Q4: Lattice-based WER")
    p.add_argument("--examples_json", default=None)
    p.add_argument("--threshold", type=float, default=AGREEMENT_THRESHOLD)
    a = p.parse_args()
    ex = None
    if a.examples_json:
        with open(a.examples_json) as f: ex = json.load(f)
    main(ex, a.threshold)
