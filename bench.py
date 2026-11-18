"""Fake callers phone your voice agent; plain code scores every call.

  python bench.py                              every scenario, on a clean line and a noisy phone line
  python bench.py --conditions all             every condition in scenarios.yaml (clean, phone, noisy-phone, fast)
  python bench.py --only changes-mind          one scenario
  python bench.py --text                       skip the audio: tests the agent's decisions only (much faster)

Each call ends PASS, FAIL (the agent broke a rule), KNOWN (a failure the scenario marks as a known trade-off) or
ERROR (the lab broke, e.g. the AI model could not load: not the agent's fault, run it again).
Writes out/scorecard.md and out/scorecard.csv, plus each call's transcript and audio under out/runs/.
"""
import argparse
import csv
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import yaml

import callers
import checks
import phoneline


def load_env(path=".env"):
    """KEY=value lines from .env into the environment (no extra package needed); # comments are ignored."""
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = re.sub(r"(^|\s)#.*$", "", line).strip()
            if "=" in line:
                key, value = line.split("=", 1)
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ.setdefault(key.strip(), value)


def run_one(scenario: dict, condition_name: str, condition: dict, agent, started: datetime, out: Path,
            text_only: bool = False, llm=callers.ollama_chat, voices=None) -> dict:
    """One fake call: the caller speaks, the line is degraded, the agent hears it with its own ears and answers."""
    if voices is None and not text_only:
        import voices  # the callers' voices (Kokoro), loaded only when audio is used
    folder = out / "runs" / f"{scenario['id']}--{condition_name}"
    shutil.rmtree(folder, ignore_errors=True)  # a fresh calendar and no stale audio from an earlier run
    folder.mkdir(parents=True, exist_ok=True)
    call = agent.new_call(started, folder / "agent")
    caller = callers.make_caller(scenario, llm)
    history, agent_lines, errors, seconds, lab = [("agent", call.greeting)], [call.greeting], [], [], []
    for turn in range(1, checks.MAX_TURNS + 2):
        try:
            line = caller.next_line(agent_lines[-1], history)
        except Exception as err:  # the AI playing the caller failed: a lab problem
            lab.append(f"the AI caller failed: {str(err)[:80]}")
            break
        if not line:
            break
        t0 = time.perf_counter()
        if text_only:
            heard = line
        else:
            speed = scenario.get("speed", 1.0) * condition.get("speed", 1.0)
            samples, rate = phoneline.degrade(voices.speak(line, scenario.get("voice", "bf_alice"), speed),
                                              voices.SR, condition, seed=turn)
            wav = folder / f"caller-{turn}.wav"
            voices.save_wav(wav, samples, rate)
            heard = call.hear(wav)
            errors.append(checks.wer(line, heard))
        reply = call.reply(heard)
        seconds.append(time.perf_counter() - t0)
        history += [("caller said", line), ("agent heard", heard), ("agent", reply)]
        agent_lines.append(reply)
        if call.done:
            break
    if getattr(call, "infra_errors", 0):
        lab.append(f"{call.infra_errors} of the agent's AI calls failed (model not available)")
    result = {"outcome": call.outcome, "booked": call.booked, "agent_lines": agent_lines,
              "turns": sum(1 for who, _ in history if who == "caller said")}
    fails = checks.check(scenario.get("expect", {}), result, started)
    critical = any(f.startswith("CRITICAL") for f in fails)  # a missed emergency is the agent's, lab error or not
    status = ("FAIL" if critical else "ERROR" if lab else "PASS" if not fails
              else "KNOWN" if scenario.get("known_issue") else "FAIL")
    notes = fails + lab if critical else lab or fails
    if status == "KNOWN":
        notes = fails + [f"known: {scenario['known_issue']}"]
    (folder / "transcript.txt").write_text(
        "\n".join(f"{who.upper():12} {text}" for who, text in history) + f"\n\n{status}"
        + (": " + "; ".join(notes) if notes else "") + "\n", encoding="utf-8")
    return {"scenario": scenario["id"], "condition": condition_name, "status": status, "pass": status == "PASS",
            "fails": notes, "outcome": call.outcome or "hung up",
            "booked": call.booked.isoformat(timespec="minutes") if call.booked else "", "turns": result["turns"],
            "misheard": round(sum(errors) / len(errors), 2) if errors else None,
            "seconds_per_turn": round(sum(seconds) / len(seconds), 1) if seconds else None}


def scorecard(rows: list[dict], out: Path) -> str:
    """A grid of scenario x condition, the failures spelled out, and the averages per condition."""
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    ids = list(dict.fromkeys(r["scenario"] for r in rows))
    cell = {(r["scenario"], r["condition"]): r for r in rows}
    show = {"PASS": "PASS", "FAIL": "**FAIL**", "KNOWN": "known", "ERROR": "error"}
    lines = ["| scenario | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for i in ids:
        lines.append(f"| {i} | " + " | ".join(show[cell[(i, c)]["status"]] if (i, c) in cell else "-" for c in conds)
                     + " |")
    count = {s: sum(r["status"] == s for r in rows) for s in show}
    critical = sum(r["status"] == "FAIL" and any(f.startswith("CRITICAL") for f in r["fails"]) for r in rows)
    scored = len(rows) - count["ERROR"]
    lines += ["", f"{count['PASS']}/{scored} calls passed, {count['FAIL']} failed ({critical} critical: an "
                  f"emergency not flagged), {count['KNOWN']} known trade-off"
              + (f", {count['ERROR']} not scored (lab error: run again)" if count["ERROR"] else "") + ".", ""]
    for c in conds:
        rs = [r for r in rows if r["condition"] == c and r["status"] != "ERROR"]
        heard = [r["misheard"] for r in rs if r["misheard"] is not None]
        speed = [r["seconds_per_turn"] for r in rs if r["seconds_per_turn"]]
        lines.append(f"- {c}: {sum(r['pass'] for r in rs)}/{len(rs)} passed"
                     + (f", {round(100 * sum(heard) / len(heard))}% of words misheard" if heard else "")
                     + (f", {sum(speed) / len(speed):.1f} s per turn" if speed else ""))
    notes = [r for r in rows if r["status"] != "PASS"]
    if notes:
        lines += ["", "Not passed:"] + [f"- {r['scenario']} ({r['condition']}, {r['status']}): {'; '.join(r['fails'])}"
                                        for r in notes]
    text = "\n".join(lines) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.md").write_text(text, encoding="utf-8")
    with open(out / "scorecard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["scenario"])
        w.writeheader()
        for r in rows:
            w.writerow(r | {"fails": " | ".join(r["fails"])})
    return text


def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="scenarios.yaml")
    ap.add_argument("--conditions", default="clean,noisy-phone", help="comma list, or 'all'")
    ap.add_argument("--only", help="comma list of scenario ids")
    ap.add_argument("--text", action="store_true", help="no audio: test the agent's decisions only")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.scenarios, encoding="utf-8"))
    started = datetime.fromisoformat(spec["call_time"])
    conds = spec["conditions"] if args.conditions == "all" else {c: spec["conditions"][c] for c in args.conditions.split(",")}
    if args.text:
        conds = {"text": {}}
    scenarios = [s for s in spec["scenarios"] if not args.only or s["id"] in args.only.split(",")]
    from agents.dental import DentalAgent
    rows = run_all(scenarios, conds, DentalAgent(), started, Path(args.out), args.text)
    print("\n" + scorecard(rows, Path(args.out)))


def run_all(scenarios, conds, agent, started, out: Path, text_only=False, **kw) -> list[dict]:
    """Every scenario under every condition. One crash in the lab (voice, ears...) costs one call, not the run,
    and the scorecard is saved after every call."""
    rows = []
    for s in scenarios:
        for name, cond in conds.items():
            try:
                r = run_one(s, name, cond, agent, started, out, text_only, **kw)
            except Exception as err:
                r = {"scenario": s["id"], "condition": name, "status": "ERROR", "pass": False,
                     "fails": [f"the bench crashed: {type(err).__name__}: {str(err)[:100]}"], "outcome": "-",
                     "booked": "", "turns": 0, "misheard": None, "seconds_per_turn": None}
            rows.append(r)
            scorecard(rows, out)
            extra = f" | misheard {round(100 * r['misheard'])}%" if r["misheard"] is not None else ""
            print(f"{r['status']:<5}  {s['id']:<26} {name:<12} {r['outcome']:<8} {r['booked']:<17}{extra}"
                  + ("" if r["pass"] else f"\n       -> {'; '.join(r['fails'])}"), flush=True)
    return rows


if __name__ == "__main__":
    main()
