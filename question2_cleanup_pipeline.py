"""
QUESTION 2 — Hindi ASR Cleanup Pipeline
=========================================
Uses REAL data from:
  https://storage.googleapis.com/upload_goai/967179/825780_transcription.json

Real JSON schema:
  [{"start": 0.11, "end": 14.42, "speaker_id": 245746, "text": "..."}, ...]

Operations:
  a) Number Normalisation  — with context-aware verb-दो detection (BUG FIXED)
  b) English Word Detection — Roman + Devanagari loan-words tagged [EN]...[/EN]

BUG FIX:
  "एक हज़ार रुपये दो" was producing "1000 रुपये 2"  ❌
  Now correctly produces "1000 रुपये दो"             ✓
  Rule: दो after currency/unit/pronoun = VERB, not numeral.

Install:
    pip install pandas tqdm requests
"""

import re, json, time
from typing import List, Tuple, Dict, Optional, Set
import pandas as pd
import requests
from tqdm import tqdm


GCS_BASE             = "https://storage.googleapis.com/upload_goai"
EXAMPLE_USER_ID      = "967179"
EXAMPLE_RECORDING_ID = "825780"
EXAMPLE_TRANS_URL    = f"{GCS_BASE}/{EXAMPLE_USER_ID}/{EXAMPLE_RECORDING_ID}_transcription.json"


def fetch_real_transcription(url: str) -> Optional[list]:
    """Fetch real GCS transcription JSON."""
    try:
        r = requests.get(url, timeout=30); r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] {url}: {e}"); return None


def get_real_texts(url: str) -> List[str]:
    """
    Extract list of text strings from real GCS JSON.
    Real format: [{"start":..., "end":..., "speaker_id":..., "text":"..."}, ...]
    """
    data = fetch_real_transcription(url)
    if not data: return []
    return [seg["text"] for seg in data if seg.get("text", "").strip()]



UNIT_MAP: Dict[str, int] = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
}
TENS_MAP: Dict[str, int] = {
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28,
    "उनतीस": 29, "तीस": 30, "इकतीस": 31, "बत्तीस": 32,
    "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36,
    "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39, "चालीस": 40,
    "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48,
    "उनचास": 49, "पचास": 50, "इक्यावन": 51, "बावन": 52,
    "तिरपन": 53, "चौवन": 54, "पचपन": 55, "छप्पन": 56,
    "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59, "साठ": 60,
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68,
    "उनहत्तर": 69, "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72,
    "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75, "छिहत्तर": 76,
    "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79, "अस्सी": 80,
    "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88,
    "नवासी": 89, "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92,
    "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95, "छियानवे": 96,
    "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}
SCALE_MAP: Dict[str, int] = {
    "सौ": 100, "हज़ार": 1_000, "हजार": 1_000,
    "लाख": 100_000, "करोड़": 10_000_000,
    "करोड": 10_000_000, "अरब": 1_000_000_000,
}
ALL_NUM: Dict[str, int] = {**UNIT_MAP, **TENS_MAP, **SCALE_MAP}

CURRENCY_WORDS: Set[str] = {
    "रुपये","रुपए","रूपये","रूपए","रुपया","पैसे","पैसा",
    "डॉलर","यूरो","पौंड",
    "किलो","ग्राम","लीटर","मीटर","फ़ीट","इंच",
    "खाना","पानी","चाय","कॉफ़ी","दूध",
}
VERB_DO_BEFORE: Set[str] = CURRENCY_WORDS | {
    "मुझे","हमें","उसे","इसे","उन्हें","आप","तुम","मेको",
    "कृपया","ज़रा","please",
}

IDIOM_PATTERNS = [
    re.compile(r"दो[-–]चार"),   re.compile(r"चार[-–]पाँच"),
    re.compile(r"पाँच[-–]सात"), re.compile(r"एक[-–]दो"),
    re.compile(r"दो[-–]तीन"),   re.compile(r"सात[-–]आठ"),
    re.compile(r"नौ[-–]दस"),    re.compile(r"दो\s+टूक"),
    re.compile(r"दो\s+राय"),
]

def _is_idiom(text: str, start: int, end: int) -> bool:
    snippet = text[max(0, start-6): end+10]
    return any(p.search(snippet) for p in IDIOM_PATTERNS)

def _is_verb_do(tokens: List[str], idx: int) -> bool:
    """
    Returns True if "दो" at position idx is a VERB (imperative of देना).

    Rules:
      R1: Preceded by currency/unit/pronoun word → VERB
          "रुपये दो", "पानी दो", "मुझे दो"
      R2: Sentence-final with no preceding scale word → VERB
          "ले लो दो" — borderline, treated as verb
      R3: Precedes a scale word (हज़ार, सौ) → NUMERAL
          "दो हज़ार", "दो सौ"
      R4: Part of compound numeral (surrounded by other numbers) → NUMERAL
    """
    prev = tokens[idx-1] if idx > 0 else ""
    nxt  = tokens[idx+1] if idx+1 < len(tokens) else ""

   
    if nxt in SCALE_MAP:
        return False
 
    if prev in SCALE_MAP:
        return False
 
    if prev in VERB_DO_BEFORE:
        return True
 
    stripped = nxt.strip("।?!.,")
    if (stripped == "" or idx == len(tokens)-1) and prev not in ALL_NUM:
        return True
    return False

def words_to_number(tokens: List[str]) -> Optional[int]:
    if not tokens: return None
    result, current = 0, 0
    for tok in tokens:
        if tok in UNIT_MAP or tok in TENS_MAP:
            current += ALL_NUM[tok]
        elif tok in SCALE_MAP:
            scale = SCALE_MAP[tok]
            if scale >= 1_000:
                result += (current if current else 1) * scale; current = 0
            else:
                current = (current if current else 1) * scale
        else:
            return None
    return result + current

def tokenise(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

def normalise_numbers(text: str) -> str:
    """
    Context-aware number normalisation.
    KEY: checks _is_verb_do() before converting any "दो".
    """
    triples = tokenise(text)
    if not triples: return text
    words = [t for t, _, _ in triples]

    parts, prev_end, i = [], 0, 0
    while i < len(triples):
        tok, start, end = triples[i]
        if tok not in ALL_NUM:
            i += 1; continue
        if _is_idiom(text, start, end):
            i += 1; continue
        if tok == "दो" and _is_verb_do(words, i):
            i += 1; continue

   
        num_toks = [tok]
        j = i + 1
        while j < len(triples):
            nxt = triples[j][0]
            if nxt == "दो" and _is_verb_do(words, j): break
            if nxt in ALL_NUM: num_toks.append(nxt); j += 1
            else: break

        converted, best_j = None, i + 1
        for k in range(len(num_toks), 0, -1):
            val = words_to_number(num_toks[:k])
            if val is not None:
                converted = val; best_j = i + k; break

        parts.append(text[prev_end:start])
        if converted is not None:
            parts.append(str(converted))
            prev_end = triples[best_j-1][2]; i = best_j
        else:
            parts.append(tok); prev_end = end; i += 1

    parts.append(text[prev_end:])
    return "".join(parts)




DEVANAGARI_LOANWORDS: Set[str] = {
   
    "एरिया","टेंट","कैम्प","प्रोजेक्ट","मिस्टेक","अमेजन",

    "कंप्यूटर","कम्प्यूटर","मोबाइल","इंटरनेट","वेबसाइट","ऐप",
    "सॉफ्टवेयर","हार्डवेयर","ब्राउज़र","सर्वर","डेटा","लैपटॉप",
    "टैबलेट","स्क्रीन","कीबोर्ड","चार्जर","कैमरा","वीडियो",
    "ऑडियो","स्पीकर","माइक्रोफोन",

    "इंटरव्यू","जॉब","ऑफिस","मीटिंग","प्रोजेक्ट","टीम",
    "मैनेजर","बॉस","सैलरी","बोनस","ट्रेनिंग","प्रेजेंटेशन",
    "रिपोर्ट","डेडलाइन","टारगेट","बजट","क्लाइंट","कस्टमर",

    "मैसेज","ईमेल","चैट","व्हाट्सएप","फेसबुक","इंस्टाग्राम",
    "ट्विटर","यूट्यूब","गूगल",

    "प्रॉब्लम","सॉल्यूशन","आइडिया","प्लान","टाइम","डेट",
    "पार्टी","फिल्म","मूवी","शॉपिंग","मार्केट","होटल",
    "रेस्टोरेंट","कैफे","पिज़्ज़ा","बर्गर",

    "स्कूल","कॉलेज","यूनिवर्सिटी","क्लास","टेस्ट","एग्जाम",
    "बैंक","लोन","पेमेंट","फ्लाइट","टैक्सी","बाइक","कार","ट्रेन","बस",
}

ROMAN_RE = re.compile(r"[A-Za-z]")

def detect_english_words(text: str) -> Tuple[str, List[str]]:
    """
    Tag English words (Roman or known Devanagari transliteration).
    Returns (tagged_text, list_of_detected_English_words).
    """
    tokens   = text.split()
    tagged   = []
    detected = []
    for tok in tokens:
        core = re.sub(r"^[।,\.!\?\"\']+|[।,\.!\?\"\']+$", "", tok)
        if not core:
            tagged.append(tok); continue
        if bool(ROMAN_RE.search(core)) or core in DEVANAGARI_LOANWORDS:
            tagged.append(f"[EN]{tok}[/EN]")
            detected.append(core)
        else:
            tagged.append(tok)
    return " ".join(tagged), detected




def cleanup_pipeline(raw_asr: str) -> Dict[str, str]:
    stage1 = normalise_numbers(raw_asr)
    stage2, detected = detect_english_words(stage1)
    return {"raw": raw_asr, "number_norm": stage1,
            "english_tagged": stage2, "english_words": detected}




NUMBER_EXAMPLES = [
   
    ("मेरी उम्र पच्चीस साल है",
     "मेरी उम्र 25 साल है", None),
    ("यहाँ तीन सौ चौवन लोग थे",
     "यहाँ 354 लोग थे", None),
    ("एक हज़ार रुपये दो",
     "1000 रुपये दो",
     "KEY FIX: दो=VERB(give). Rule R1: 'रुपये' precedes दो → skip conversion."),
    ("मुझे दो रुपये दो",
     "मुझे 2 रुपये दो",
     "First दो: numeral (no currency before it). "
     "Second दो: verb (रुपये precedes)."),
    ("छः सात आठ किलोमीटर में नौ बजे है",
     "6 7 8 किलोमीटर में 9 बजे है",
     "From real data segment: time/distance context → convert."),

    ("दो-चार बातें कर लो",
     "दो-चार बातें कर लो",
     "Idiom: दो-चार = a few → must NOT convert."),
    ("उसने दो टूक बात कही",
     "उसने दो टूक बात कही",
     "Idiom: दो टूक = bluntly → must NOT convert."),
    ("दो हज़ार चौबीस में",
     "2024 में",
     "Year: दो→2, हज़ार→1000, चौबीस→24 → 2024. Judgment: year context, convert."),
]

def run_number_demo():
    print("\n" + "="*65)
    print("PART A — NUMBER NORMALISATION (with verb-दो BUG FIX)")
    print("="*65)
    print("""
Context-aware verb-दो rule:
  R1: दो after currency/unit/pronoun → VERB (रुपये दो, पानी दो, मुझे दो)
  R2: दो sentence-final, no prior numeral → VERB
  R3: दो before scale word → NUMERAL (दो हज़ार, दो सौ)
  Idiom: दो-चार, दो टूक → NEVER convert
""")
    all_ok = True
    for inp, expected, note in NUMBER_EXAMPLES:
        result = normalise_numbers(inp)
        ok = result == expected
        if not ok: all_ok = False
        print(f"  {'✓' if ok else '✗'} INPUT   : {inp}")
        print(f"      OUTPUT  : {result}")
        print(f"      EXPECTED: {expected}")
        if not ok: print(f"      *** MISMATCH ***")
        if note:   print(f"      NOTE    : {note}")
        print()
    print(f"  {'All tests passed ✓' if all_ok else 'Some tests FAILED ✗'}")

def run_english_demo():
    print("\n" + "="*65)
    print("PART B — ENGLISH WORD DETECTION")
    print("="*65)
    examples = [
        
        "जो एरिया में उधर की एरिया में उसके बारे में देखना",
        "हम वहां गया थे कुड़रमा घाटी तरफ पर दिवोग काफी जंगली एरिया है",
        "टेंट गड़ा और रहा तो जब पता जैसी रात हुआ",
        "आसपास की एरिया मतलब कुछ एरिया में थोड़ा पता है आग लहरा देना चाहिए",
        "हमने मिस्टेक किए कि हम लाइट नहीं ले गए थे",
        "और फिर अमेजन का जंगन होता है ना",
       
        "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
        "यह problem solve नहीं हो रहा",
        "कल meeting है office में",
        "वह बहुत अच्छा इंसान है",       
    ]
    for s in examples:
        tagged, found = detect_english_words(s)
        print(f"  INPUT   : {s}")
        print(f"  TAGGED  : {tagged}")
        print(f"  DETECTED: {found if found else '(none)'}")
        print()

def run_real_data_pipeline():
    """Run pipeline on actual GCS data from recording 825780."""
    print("\n" + "="*65)
    print("PIPELINE ON REAL GCS DATA (recording 825780 / user 967179)")
    print("="*65)
    print(f"  Fetching: {EXAMPLE_TRANS_URL}")

    texts = get_real_texts(EXAMPLE_TRANS_URL)
    if not texts:
        print("  [WARN] Could not fetch real data — showing demo mode")
        texts = [
            "जो जन जाती पाई जाती है उधर कि उधर की एरिया में",
            "हम वहां गया थे कुड़रमा घाटी तरफ पर दिवोग काफी जंगली एरिया है",
            "हमने मिस्टेक किए कि हम लाइट नहीं ले गए थे",
            "जब हम रहने के लिए गए थे नातो चाहते के साथ जैसे हम वहाँ पहले एंटर किये थे",
            "छः सात आठ किलोमीटर में नौ बजे है नौ उसके बाद",
        ]
    else:
        print(f"  ✓ Fetched {len(texts)} real segments")

    results = []
    for text in texts:
        out = cleanup_pipeline(text)
        results.append(out)
        print(f"\n  RAW   : {out['raw'][:80]}")
        print(f"  NORMED: {out['number_norm'][:80]}")
        print(f"  TAGGED: {out['english_tagged'][:80]}")
        if out['english_words']:
            print(f"  EN    : {out['english_words']}")

    df = pd.DataFrame(results)
    df.to_csv("q2_pipeline_output.csv", index=False, encoding="utf-8-sig")
    print(f"\n  ✓ Saved {len(df)} rows → q2_pipeline_output.csv")
    return df


def main():
    print("=" * 65)
    print("QUESTION 2: ASR CLEANUP PIPELINE (with real GCS data)")
    print("=" * 65)
    run_number_demo()
    run_english_demo()
    run_real_data_pipeline()
    print("\n✅ Question 2 complete.")


if __name__ == "__main__":
    main()
