"""Drive BenchActivity on the phone over adb.

usage: run.py <plan.json> <results.jsonl> [--reps N] [--no-cooldown]
       run.py --builds <builds.json> --out <results.jsonl> [--reps N]

plan.json is a list of configs: {"tag", "audio", "model", "threads", "chunkMax",
"chunkMin", "ctxMin", "ctxPad", "vad", "timestamps", "vadThreshold", "vadPad",
"vadDrop", "vadAgc", "variant", "pkg"}; missing keys use app defaults (see
BenchActivity and the tuning struct in jni_bridge.cpp). Configs are interleaved across repetitions so thermal drift spreads
evenly. Before each run we wait for the phone to cool down.
"""
import json, re, subprocess, sys, time

DEV = ["adb"]  # pick the device with ANDROID_SERIAL if more than one is connected
RESULTS = "/sdcard/Android/data/{pkg}/files/bench_results.jsonl"
MAX_BIG_C, MAX_SKIN_C, MIN_GAP_S, MAX_WAIT_S = 45.0, 35.0, 15, 300


def sh(*args, check=True):
    return subprocess.run(DEV + ["shell", *args], capture_output=True, text=True, check=check).stdout


def temps():
    out = sh("dumpsys", "thermalservice")
    hal = out.split("Current temperatures from HAL:")[1].split("Current cooling")[0]
    t = {name: float(v) for v, name in re.findall(r"mValue=([\d.]+), mType=-?\d+, mName=([\w-]+)", hal)}
    return t.get("BIG", 0.0), t.get("VIRTUAL-SKIN", 0.0)


NO_COOLDOWN = "--no-cooldown" in sys.argv  # accuracy-only runs: timings will be warm


def cool_down():
    if NO_COOLDOWN:
        return temps()
    time.sleep(MIN_GAP_S)
    start = time.time()
    while True:
        big, skin = temps()
        if (big <= MAX_BIG_C and skin <= MAX_SKIN_C) or time.time() - start > MAX_WAIT_S:
            return big, skin
        time.sleep(5)


def whisper_timings():
    """Totals from whisper_print_timings in logcat, e.g. {"encode": [ms, runs]}."""
    log = subprocess.run(DEV + ["logcat", "-d", "-s", "whisper:I", "WhisperJNI:I"], capture_output=True, text=True).stdout
    out = {}
    for name, ms, runs in re.findall(r"print_timings:\s+(\w+) time =\s+([\d.]+) ms /\s+(\d+) runs", log):
        out[name] = [float(ms), int(runs)]
    m = re.search(r"fallbacks =\s+(\d+) p /\s+(\d+) h", log)
    if m:
        out["fallbacks"] = int(m.group(1)) + int(m.group(2))
    out["chunks"] = re.findall(r"WhisperJNI: (chunk .*)", log)
    out["vad"] = re.findall(r"WhisperJNI: VAD speech (.*)", log)
    return out


def n_lines(pkg):
    out = sh("cat", RESULTS.format(pkg=pkg), check=False)
    return [l for l in out.splitlines() if l.strip()]


def run_one(cfg):
    pkg = cfg.get("pkg", "codes.felix.whisp.bench")
    before = len(n_lines(pkg))
    big, skin = cool_down()
    sh("am", "force-stop", pkg)
    subprocess.run(DEV + ["logcat", "-c"], check=False)
    sh("input", "keyevent", "KEYCODE_WAKEUP")
    cmd = ["am", "start", "-W", "-n", f"{pkg}/codes.felix.whisp.BenchActivity",
           "--es", "audio", cfg["audio"], "--es", "tag", cfg.get("tag", "")]
    for key, flag in [("model", "--es"), ("variant", "--es"), ("threads", "--ei"), ("ctxMin", "--ei"),
                      ("ctxPad", "--ei"), ("chunkMax", "--ef"), ("chunkMin", "--ef"), ("vad", "--ez"),
                      ("timestamps", "--ez"), ("vadThreshold", "--ef"), ("vadPad", "--ei"), ("vadDrop", "--ez"), ("vadAgc", "--ez")]:
        if key in cfg:
            val = cfg[key]
            cmd += [flag, key, str(val).lower() if isinstance(val, bool) else str(val)]
    sh(*cmd)
    deadline = time.time() + 900
    while time.time() < deadline:
        time.sleep(3)
        lines = n_lines(pkg)
        if len(lines) > before:
            rec = json.loads(lines[-1])
            rec.update({"big_c_before": big, "skin_c_before": skin, "cfg": cfg})
            rec["big_c_after"], rec["skin_c_after"] = temps()
            rec["whisper"] = whisper_timings()
            return rec
    return {"cfg": cfg, "error": "timeout"}


def install(apk):
    subprocess.run(DEV + ["install", "-r", apk], capture_output=True, check=True)


def main_builds():
    """--builds plan.json: [{"apk": path, "build": name, "configs": [...]}], interleaved per rep."""
    plan = json.load(open(sys.argv[sys.argv.index("--builds") + 1]))
    out = sys.argv[sys.argv.index("--out") + 1]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1
    for rep in range(reps):
        for b in plan:
            install(b["apk"])
            for cfg in b["configs"]:
                rec = run_one(cfg)
                rec.update({"rep": rep, "build": b["build"]})
                with open(out, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st = rec.get("stats") or {}
                print(f"rep{rep} {b['build']} {cfg.get('tag')}: total={st.get('total_ms')} "
                      f"decode_audio={rec.get('audio_decode_ms')} {rec.get('error', '')}", flush=True)


def main():
    if "--builds" in sys.argv:
        return main_builds()
    plan = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1
    total = len(plan) * reps
    i = 0
    for rep in range(reps):
        for cfg in plan:
            i += 1
            rec = run_one(cfg)
            rec["rep"] = rep
            with open(out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st = rec.get("stats") or {}
            print(f"[{i}/{total}] {cfg.get('tag')} {cfg['audio']}: total={st.get('total_ms')}ms "
                  f"enc={(rec.get('whisper') or {}).get('encode')} prefill={st.get('prefill_ms')} first={st.get('first_text_ms')} "
                  f"big={rec.get('big_c_before')}->{rec.get('big_c_after')} {rec.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
