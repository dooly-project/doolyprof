#!/usr/bin/env python3
"""GPU validation of the signature fields through the real CLI (run-e2e.py).

Each scenario profiles with a small sweep into a fresh SQLite DB and asserts what the
comparator shared versus re-profiled. Scenarios run in parallel, one per GPU.

  kv     : Llama-3.1-8B bf16 KV cache, then --kv-cache-dtype fp8 (FLASHINFER).
           Expect: 2nd run profiles only Attention; 2 Attention signatures; 2 profiled_models rows.
  dtype  : Llama-3.1-8B --dtype bfloat16, then --dtype float16.
           Expect: 2nd run re-profiles the 12 floating-point ops and shares the 3 integer/float32 ops.
  overlap: Llama-3.1-8B + Command R7B in one run (FLASH_ATTN).
           Expect: 7 shared signatures incl. Command R7B's global attention; its sliding-window attention separate.
  softcap: Gemma-2-2b (FLASHINFER) smoke test of the impl-contract path on a soft-capped model.

Usage: python doolyprof/tests/validate_signature_fields_gpu.py --workdir /tmp/x --gpus 0 1 2 3 [--scenarios kv dtype overlap softcap]
"""
import argparse, json, os, sqlite3, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "..", "run", "run-e2e.py")

def e2e(workdir, label, models, backend, db, gpu, dtype="bfloat16", kv="auto", maxlen=256, maxbs=2):
    log = os.path.join(workdir, f"{label}.log")
    cmd = [sys.executable, E2E, "--models", *models, "--attention-backend", backend, "--dtype", dtype,
           "--kv-cache-dtype", kv, "--max-batch-size", str(maxbs), "--max-seq-len", str(maxlen), "--tp", "1",
           "--gpu", str(gpu), "--db-path", db, "--workspace-dir", os.path.join(workdir, f"ws_{label}")]
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    txt = open(log, errors="replace").read()
    plan = [int(x) for x in __import__("re").findall(r"Modules to Profile: (\d+)", txt)]
    return {"rc": rc, "log": log, "modules_to_profile": plan, "secs": round(time.time() - t0)}

def q(db, sql, *a):
    c = sqlite3.connect(db); r = c.execute(sql, a).fetchall(); c.close(); return r

def check(cond, msg, out):
    out.append(("PASS" if cond else "FAIL", msg)); return cond

def scenario_kv(workdir, gpu, out):
    db = os.path.join(workdir, "kv.db"); os.path.exists(db) and os.remove(db)
    r1 = e2e(workdir, "kv_bf16", ["meta-llama/Llama-3.1-8B"], "FLASHINFER", db, gpu)
    r2 = e2e(workdir, "kv_fp8", ["meta-llama/Llama-3.1-8B"], "FLASHINFER", db, gpu, kv="fp8")
    check(r1["rc"] == 0 and r2["rc"] == 0, f"kv: both runs exit 0 ({r1['secs']}s, {r2['secs']}s)", out)
    check(r1["modules_to_profile"] == [15], f"kv: first run profiles 15 modules, got {r1['modules_to_profile']}", out)
    check(r2["modules_to_profile"] == [1], f"kv: fp8 run profiles only Attention, got {r2['modules_to_profile']}", out)
    attn = q(db, "select signature_hash, signature_json from signatures where operation_name='Attention'")
    check(len(attn) == 2, f"kv: 2 Attention signatures, got {len(attn)}", out)
    check(q(db, "select count(*) from signatures")[0][0] == 16, "kv: 16 signatures total (14 shared)", out)
    rows = q(db, "select kv_cache_dtype, dtype from profiled_models order by model_id")
    check(rows == [("auto", "bfloat16"), ("fp8", "bfloat16")], f"kv: profiled_models rows keyed by kv_cache_dtype, got {rows}", out)
    if len(attn) == 2:
        s0, s1 = json.loads(attn[0][1]), json.loads(attn[1][1])
        check(s0[:2] == s1[:2] and s0[3] == s1[3] and s0[2] != s1[2], "kv: Attention signatures differ only in state hash", out)

def scenario_dtype(workdir, gpu, out):
    db = os.path.join(workdir, "dtype.db"); os.path.exists(db) and os.remove(db)
    r1 = e2e(workdir, "dt_bf16", ["meta-llama/Llama-3.1-8B"], "FLASHINFER", db, gpu, dtype="bfloat16")
    r2 = e2e(workdir, "dt_fp16", ["meta-llama/Llama-3.1-8B"], "FLASHINFER", db, gpu, dtype="float16")
    check(r1["rc"] == 0 and r2["rc"] == 0, f"dtype: both runs exit 0 ({r1['secs']}s, {r2['secs']}s)", out)
    check(r2["modules_to_profile"] == [12], f"dtype: float16 run re-profiles the 12 float ops, got {r2['modules_to_profile']}", out)
    dts = dict(q(db, "select json_extract(signature_json, '$[3]') d, count(*) from signatures group by d order by d"))
    non_float = {k: v for k, v in dts.items() if k not in ("bfloat16", "float16")}
    # 12 floating-point ops per model dtype; the integer casts (token ids) and the float32 argmax
    # do not depend on the model dtype and must appear once.
    check(dts.get("bfloat16") == 12 and dts.get("float16") == 12 and sum(non_float.values()) == 3
          and set(non_float) <= {"float32", "int32", "int64"}, f"dtype: per-op dtype fields {dts}", out)
    shared = q(db, "select count(*) from (select operation_id from model_operations group by operation_id having count(distinct model_id)=2)")[0][0]
    check(shared == 3, f"dtype: 3 signatures shared across model dtypes (int/float32 ops), got {shared}", out)

def scenario_overlap(workdir, gpu, out):
    db = os.path.join(workdir, "overlap.db"); os.path.exists(db) and os.remove(db)
    r = e2e(workdir, "overlap", ["meta-llama/Llama-3.1-8B", "CohereLabs/c4ai-command-r7b-12-2024"], "FLASH_ATTN", db, gpu)
    check(r["rc"] == 0, f"overlap: run exits 0 ({r['secs']}s)", out)
    check(r["modules_to_profile"] == [15, 10], f"overlap: plan 15 + 10 modules, got {r['modules_to_profile']}", out)
    shared = q(db, "select count(*) from (select operation_id from model_operations group by operation_id having count(distinct model_id)=2)")[0][0]
    check(shared == 7, f"overlap: 7 shared signatures, got {shared}", out)
    attn = q(db, """select s.signature_hash, group_concat(pm.model_name||':'||mo.count) from signatures s
                     join model_operations mo on mo.operation_id=s.operation_id join profiled_models pm on pm.model_id=mo.model_id
                     where s.operation_name='Attention' group by s.signature_hash""")
    holders = sorted(h for _, h in attn)
    check(len(attn) == 2 and any("Llama-3.1-8B:32" in h and "command-r7b-12-2024:8" in h for h in holders)
          and any(h == "CohereLabs/c4ai-command-r7b-12-2024:24" for h in holders),
          f"overlap: global attention shared (32+8), sliding-window separate (24); got {holders}", out)

def scenario_softcap(workdir, gpu, out):
    db = os.path.join(workdir, "softcap.db"); os.path.exists(db) and os.remove(db)
    r = e2e(workdir, "softcap", ["google/gemma-2-2b"], "FLASHINFER", db, gpu)
    check(r["rc"] == 0, f"softcap: Gemma-2-2b run exits 0 ({r['secs']}s)", out)
    attn = q(db, "select count(*) from signatures where operation_name='Attention'")[0][0]
    check(attn == 2, f"softcap: Gemma-2 keeps its 2 attention variants (sliding/global), got {attn}", out)

SCENARIOS = {"kv": scenario_kv, "dtype": scenario_dtype, "overlap": scenario_overlap, "softcap": scenario_softcap}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workdir", required=True); ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS)); a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    results = {}; threads = []
    for i, name in enumerate(a.scenarios):
        out = results.setdefault(name, []); gpu = a.gpus[i % len(a.gpus)]
        t = threading.Thread(target=lambda n=name, g=gpu, o=out: SCENARIOS[n](a.workdir, g, o)); t.start(); threads.append(t)
    for t in threads: t.join()
    fails = 0
    for name, out in results.items():
        print(f"== {name}")
        for status, msg in out:
            print(f"   {status} {msg}"); fails += status == "FAIL"
    print(f"\n{fails} failure(s)"); sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
