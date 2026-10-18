"""Summarize benchmark results: median timings and word error rate per config.

usage: report.py <results.jsonl>... [--by key1,key2] [--notes dir]
"""
import json, re, statistics, sys, unicodedata
from collections import defaultdict

import jiwer


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[„“”\"'’‚‘«»]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation to spaces, keeps umlauts/digits
    return " ".join(text.split())


def main():
    args = sys.argv[1:]
    by = ["tag", "audio"]
    notes = "notes"
    files = []
    i = 0
    while i < len(args):
        if args[i] == "--by":
            by = args[i + 1].split(","); i += 2
        elif args[i] == "--notes":
            notes = args[i + 1]; i += 2
        else:
            files.append(args[i]); i += 1

    groups = defaultdict(list)
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if "stats" not in r or r.get("error"):
                print("skipping failed run:", r.get("cfg"), r.get("error"), file=sys.stderr)
                continue
            groups[tuple(str(r.get(k, r.get("cfg", {}).get(k, ""))) for k in by)].append(r)

    cols = ["n", "total_s", "per_audio_min", "encode_s", "decode_ms_tok", "first_text_s", "wer_%", "big_after"]
    print(" | ".join(by + cols))
    print("|".join(["---"] * (len(by) + len(cols))))
    for key, runs in sorted(groups.items()):
        med = lambda xs: statistics.median(xs) if xs else float("nan")
        total = med([r["stats"]["total_ms"] / 1000 for r in runs])
        audio_s = runs[0]["stats"]["audio_s"]
        enc = med([r["whisper"]["encode"][0] / 1000 for r in runs if "encode" in r.get("whisper", {})])
        dec = med([r["whisper"]["decode"][0] / max(1, r["whisper"]["decode"][1]) for r in runs if "decode" in r.get("whisper", {})])
        first = med([r["stats"]["first_text_ms"] / 1000 for r in runs])
        ref = normalize(json.load(open(f"{notes}/{runs[0]['audio'].rsplit('.', 1)[0]}.json"))["reference"])
        wer = med([100 * jiwer.wer(ref, normalize(r.get("text", "")) or "-") for r in runs])
        big = med([r.get("big_c_after", 0) for r in runs])
        row = [len(runs), f"{total:.1f}", f"{total / audio_s * 60:.1f}", f"{enc:.1f}", f"{dec:.1f}", f"{first:.1f}", f"{wer:.1f}", f"{big:.0f}"]
        print(" | ".join(list(key) + [str(x) for x in row]))


if __name__ == "__main__":
    main()
