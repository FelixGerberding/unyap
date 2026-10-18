# Transcription benchmark

Measures speed and accuracy of the native pipeline on a real phone, using the
`benchmark` build type (release-equivalent, installs as "Unyap Bench" next to
the other builds) and its `BenchActivity`.

## Setup

```bash
# Python deps (any venv)
pip install pyarrow numpy jiwer

# Test notes: FLEURS test sentences (CC BY 4.0) stitched into ~12-120 s
# "voice notes" with reference transcripts, encoded as Ogg Opus.
mkdir -p data notes models
curl -L -o data/fleurs_de_de.parquet "https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/de_de/test/0000.parquet"
curl -L -o data/fleurs_en_us.parquet "https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/en_us/test/0000.parquet"
python make_notes.py data notes
for f in notes/*.wav; do ffmpeg -y -i "$f" -c:a libopus -b:a 24k -ac 1 "${f%.wav}.ogg"; done

# Models to compare
curl -L -o models/ggml-small-q8_0.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin

# The app fetches notes/ and models/ from this server through adb reverse
# (files adb pushes into the app's external dir are invisible to the app).
python -m http.server 8765 --bind 127.0.0.1 &
adb reverse tcp:8765 tcp:8765

./gradlew installBenchmark   # from the repo root
adb shell svc power stayon true   # throttling behind a dark screen skews results
```

## Running

```bash
echo '[{"tag":"default","audio":"de_60s.ogg"},
       {"tag":"q4","audio":"de_60s.ogg","model":"ggml-small-q4_0.bin"}]' > plan.json
python run.py plan.json results.jsonl --reps 3      # timing: waits for cooldown
python run.py plan.json results.jsonl --no-cooldown # accuracy only: much faster
python report.py results.jsonl --by tag,audio --notes notes
```

Greedy decoding is deterministic, so one run per config is enough for word
error rate; timings need cooldown and repetitions. With four notes (~1100
words), WER differences below ~3 points are noise: changing almost anything
moves the chunk cut points, which shifts errors around.

## Findings (Pixel 8, 2026-10)

Defaults: small q8_0, 5 threads, armv8.6 CPU variant, 6-12 s chunks, encoder
context +4 s, VAD with level normalization, libopus decoding. About 5.5 s per
minute of audio, first text after ~1 s, pooled WER 8.0%.

| Change | Result |
|---|---|
| armv9.0 (SVE2) CPU variant, ggml's own pick | encoder ~3x slower than armv8.6 |
| armv8.2 / armv8.0 variants (older phones) | same text; 1.05x / 2.2x slower |
| 4 / 6 / 8 / 9 threads (vs 5) | 13% slower / 75% / 52% / 43% slower |
| MediaCodec Opus decoding (vs libopus) | 2.5-7 s per minute of audio (vs 0.08 s) |
| VAD without level normalization | dropped quiet speech, English WER 27% |
| context padding 2 s / 6 s (vs 4 s) | WER 13.2% / 10.3% |
| 8 s / 20 s max chunks (vs 12 s) | WER 21.6% / 13.7% |
| no chunking (30 s windows) | WER 23.9%, first text after ~5 s |
| small q4_0 | 6% faster, WER 9.5% |
| small q5_0 / q5_1 | no i8mm path: encoder ~3x slower |
| base q8_0 | WER 34%, hallucination loops |
| KleidiAI (ggml at the time) | wrong output |
| -O2 vs -O3, debuggable vs not | no measurable difference |
