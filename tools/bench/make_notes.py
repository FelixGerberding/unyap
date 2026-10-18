# Build test "voice notes" from FLEURS test clips: concatenated sentences with
# short pauses, plus their reference transcripts.
import json, subprocess, sys, wave
import numpy as np
import pyarrow.parquet as pq

data_dir, out_dir = sys.argv[1], sys.argv[2]
specs = {"de_de": [("de_12s", 12), ("de_60s", 60), ("de_120s", 120)], "en_us": [("en_60s", 60)]}
PAUSE = np.zeros(int(0.6 * 16000), dtype=np.int16)

for lang, notes in specs.items():
    t = pq.read_table(f"{data_dir}/fleurs_{lang}.parquet", columns=["id", "audio", "raw_transcription", "num_samples"])
    rows, seen = [], set()
    for i in range(t.num_rows):
        cid = t["id"][i].as_py()
        if cid in seen:
            continue  # same sentence read by another speaker
        seen.add(cid)
        rows.append(i)
    pos = 0
    for name, target in notes:
        pcm, refs = [], []
        dur = 0.0
        while dur < target - 3:
            i = rows[pos]; pos += 1
            raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
                                 input=t["audio"][i].as_py()["bytes"], capture_output=True, check=True).stdout
            a = np.frombuffer(raw, dtype=np.int16)
            if dur > 0 and dur + len(a) / 16000 > target + 8:
                continue
            if pcm:
                pcm.append(PAUSE)
            pcm.append(a)
            refs.append(t["raw_transcription"][i].as_py())
            dur += len(a) / 16000 + 0.6
        with wave.open(f"{out_dir}/{name}.wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(np.concatenate(pcm).tobytes())
        json.dump({"lang": lang[:2], "reference": " ".join(refs), "seconds": round(dur, 1)},
                  open(f"{out_dir}/{name}.json", "w"), ensure_ascii=False)
        print(name, round(dur, 1), "s,", len(refs), "sentences")
