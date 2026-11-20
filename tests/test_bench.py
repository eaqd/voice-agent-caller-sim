"""Offline tests: stand-in agents, voices and AI, so these check the bench's own logic."""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bench  # noqa: E402
import callers  # noqa: E402
import checks  # noqa: E402
import phoneline  # noqa: E402

SPEC = yaml.safe_load(open(ROOT / "scenarios.yaml", encoding="utf-8"))
STARTED = datetime.fromisoformat(SPEC["call_time"])  # Monday 16 Nov 2026, 19:42


class FakeCall:
    """A tiny rule-based agent: books Thursday 2 pm once it has heard a name, flags 'puffy', messages 'cancel'."""
    greeting = "Hello, how can I help?"

    def __init__(self, hear_as=None):
        self.done, self.outcome, self.booked, self.name, self.hear_as = False, "", None, False, hear_as

    def hear(self, wav):
        return self.hear_as or "misheard"

    def reply(self, text):
        t = text.lower()
        if "puffy" in t or "bleeding" in t:
            self.done, self.outcome = True, "urgent"
            return "That sounds urgent."
        if "cancel" in t:
            self.done, self.outcome = True, "message"
            return "I've passed that on."
        if "yes" in t or "perfect" in t or "works" in t:
            self.done, self.outcome, self.booked = True, "booked", datetime(2026, 11, 19, 14, 0)
            return "Booked for Thursday at 2 pm."
        return "I can do Thursday 19 November at 2 pm. Would that work?"


class FakeAgent:
    def __init__(self, hear_as=None):
        self.hear_as = hear_as

    def new_call(self, started, out_dir):
        return FakeCall(self.hear_as)


class FakeVoices:
    SR = 24000

    @staticmethod
    def speak(text, voice="x", speed=1.0):
        return (np.sin(np.linspace(0, 3000, 24000)) * 0.3).astype(np.float32)

    @staticmethod
    def save_wav(path, samples, rate=24000):
        Path(path).write_bytes(b"RIFF")


def scenario(sid):
    return next(s for s in SPEC["scenarios"] if s["id"] == sid)


def test_word_error_rate():
    assert checks.wer("I'd like a clean", "I'd like a clean") == 0
    assert checks.wer("my cheek is puffy", "my cheek is puppy") == 0.25
    assert checks.wer("Three o'clock", "") == 1.0


def test_checks_catch_each_rule():
    good = {"outcome": "booked", "booked": datetime(2026, 11, 19, 14, 0), "agent_lines": ["ok"], "turns": 3}
    assert checks.check(scenario("polite-new-patient")["expect"], good, STARTED) == []
    early = good | {"booked": datetime(2026, 11, 17, 10, 0)}
    fails = checks.check(scenario("changes-mind")["expect"], early, STARTED)
    assert fails == ["booked at 10:00, caller wanted from 15:00"]  # tomorrow is right, the time is not
    missed = checks.check({"outcome": "urgent"}, good, STARTED)
    assert missed[0].startswith("CRITICAL")
    price = good | {"outcome": "message", "agent_lines": ["Whitening costs about 300 pounds."]}
    assert "said something it must not" in checks.check(scenario("asks-for-a-price")["expect"], price, STARTED)[0]
    assert "turns" in checks.check({"max_turns": 2}, good | {"turns": 3}, STARTED)[0]


def test_phone_line_is_narrow_and_noisy():
    rate = 24000
    t = np.arange(rate) / rate
    voice = (np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 6000 * t)).astype(np.float32) * 0.3
    t48 = np.arange(48000) / 48000  # 100 Hz rumble + 1 kHz voice + 6 kHz hiss, one second at 48 kHz
    mix = (np.sin(2 * np.pi * 100 * t48) + np.sin(2 * np.pi * 1000 * t48) + np.sin(2 * np.pi * 6000 * t48)) * 0.3
    y, r = phoneline.degrade(mix.astype(np.float32), 48000, {"rate": 16000})
    assert r == 16000 and len(y) == 16000

    def level(sig, hz):  # how much of one frequency is in the signal (a single DFT bin, no FFT library needed)
        ts = np.arange(len(sig)) / r
        return abs(np.dot(sig, np.exp(-2j * np.pi * hz * ts))) / len(sig)
    assert abs(level(y, 1000) - 0.15) < 0.01  # the voice band passes untouched
    assert level(y, 1000) > 100 * level(y, 6000) and level(y, 1000) > 100 * level(y, 100)  # the rest is cut
    y8, r8 = phoneline.degrade(voice, rate, {"rate": 8000})
    assert r8 == 8000 and len(y8) == 8000
    noisy, _ = phoneline.degrade(voice, rate, {"noise_db": 12})
    snr = 20 * np.log10(phoneline.rms(voice) / phoneline.rms(noisy - voice))
    assert 10 < snr < 14


def test_scripted_and_improvised_callers():
    c = callers.make_caller(scenario("changes-mind"))
    assert c.next_line("hi", []).startswith("Hello, it's Tom Reid")
    said = []

    def llm(messages):
        said.append(messages)
        return 'You: "Leo Hart."' if len(said) == 1 else "END"
    imp = callers.make_caller(scenario("improvised-busy-parent"), llm)
    assert imp.next_line("Can I take your full name?", [("agent", "Can I take your full name?")]) == "Leo Hart."
    assert "busy parent" in said[0][0]["content"]
    assert imp.next_line("You're booked in. Goodbye!", []) is None


def test_a_call_in_text_mode(tmp_path):
    r = bench.run_one(scenario("polite-new-patient"), "text", {}, FakeAgent(), STARTED, tmp_path, text_only=True)
    assert r["pass"] and r["booked"] == "2026-11-19T14:00" and r["turns"] == 3
    transcript = (tmp_path / "runs/polite-new-patient--text/transcript.txt").read_text(encoding="utf-8")
    assert "CALLER SAID" in transcript and transcript.strip().endswith("PASS")


def test_a_call_with_audio_scores_mishearing(tmp_path):
    r = bench.run_one(scenario("mumbled-emergency"), "noisy-phone", {"rate": 8000, "noise_db": 12}, FakeAgent(),
                      STARTED, tmp_path, voices=FakeVoices)
    assert r["status"] == "FAIL" and r["fails"][0].startswith("CRITICAL")  # the agent heard "misheard"
    assert r["misheard"] > 0.9
    assert (tmp_path / "runs/mumbled-emergency--noisy-phone/caller-1.wav").exists()


def test_scorecard(tmp_path):
    rows = [bench.run_one(scenario(s), "text", {}, FakeAgent(), STARTED, tmp_path, text_only=True)
            for s in ("polite-new-patient", "mumbled-emergency", "angry-cancel")]
    text = bench.scorecard(rows, tmp_path)
    assert "3/3 calls passed, 0 failed (0 critical" in text
    assert (tmp_path / "scorecard.csv").read_text(encoding="utf-8").count("\n") == 4


def test_every_scenario_is_well_formed():
    ids = [s["id"] for s in SPEC["scenarios"]]
    assert len(ids) == len(set(ids))
    for s in SPEC["scenarios"]:
        assert ("lines" in s) != ("improvise" in s) and s["expect"]["outcome"] in ("booked", "urgent", "message")
    assert set(SPEC["conditions"]) == {"clean", "phone", "noisy-phone", "fast"}


def test_the_dental_adapter_drives_the_real_receptionist(tmp_path):
    repo = ROOT.parent / "dental-voice-receptionist"
    if not (repo / "desk.py").exists():
        pytest.skip("dental-voice-receptionist is not cloned next to this folder")
    from agents.dental import DentalAgent

    def reader(system, text, schema):  # a stand-in for the AI inside the receptionist
        said = text.split("Caller:", 1)[1].lower()
        base = {k: "" for k in schema["properties"]} | {"intent": "book", "answer_to_offer": "none"}
        if "grace" in said:
            return base | {"patient_name": "Grace Mensah", "preferred_day": "Thursday", "preferred_time": "afternoon"}
        if "works" in said:
            return base | {"answer_to_offer": "yes"}
        return base | {"reason": "check-up and clean"}
    agent = DentalAgent(repo, ai=reader)
    r = bench.run_one(scenario("polite-new-patient"), "text", {}, agent, STARTED, tmp_path, text_only=True)
    assert r["pass"], r["fails"]
    assert r["booked"] == "2026-11-19T14:00"


def test_known_trade_offs_and_lab_errors_are_not_agent_failures(tmp_path):
    known = bench.run_one(scenario("no-symptoms-just-checkup"), "text", {}, FakeAgent(), STARTED, tmp_path,
                          text_only=True)
    assert known["status"] in ("PASS", "KNOWN")  # the fake agent books it; the real one flags it urgent (known)

    class BrokenCall(FakeCall):
        infra_errors = 2
    broken = type("A", (), {"new_call": lambda self, s, o: BrokenCall()})()
    err = bench.run_one(scenario("polite-new-patient"), "text", {}, broken, STARTED, tmp_path, text_only=True)
    assert err["status"] == "ERROR" and "model not available" in err["fails"][0]

    def dead_llm(messages):
        raise ConnectionError("model could not load")
    lost = bench.run_one(scenario("improvised-busy-parent"), "text", {}, FakeAgent(), STARTED, tmp_path,
                         text_only=True, llm=dead_llm)
    assert lost["status"] == "ERROR" and "AI caller failed" in lost["fails"][0]
    card = bench.scorecard([known, err, lost], tmp_path)
    assert "2 not scored (lab error: run again)" in card


def test_improvised_caller_says_its_words_before_hanging_up():
    imp = callers.ImprovisedCaller("a caller", 5, lambda m: "I'll take that one. END")
    assert imp.next_line("You're booked in. Goodbye!", []) == "I'll take that one."
    assert imp.next_line("Goodbye!", []) is None  # END after the goodbye: hung up
    mid = callers.ImprovisedCaller("a caller", 5, lambda m: "Sounds good. END")
    assert mid.next_line("Would 3 pm work?", []) == "Sounds good." and mid.left == 4  # mid-call END is ignored


def test_code_not_the_ai_decides_whether_an_offer_suits_the_caller():
    assert callers.offered("I can do Tuesday 17 November at 3:30 pm. Would that work?") == ("tuesday", 15.5)
    assert callers.offered("Can I take your full name, please?") is None
    rule = {"from": "15:00", "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]}
    assert callers.fits(("tuesday", 15.0), rule) and not callers.fits(("tuesday", 10.0), rule)
    assert not callers.fits(("saturday", 16.0), rule)
    seen = []
    imp = callers.ImprovisedCaller("a parent", 5, lambda m: seen.append(m[1]["content"]) or "Okay.", rule)
    assert imp.next_line("I can do Tuesday 17 November at 3 pm. Would that work?", []) == "Yes, that works, thanks."
    assert seen == []  # a fitting offer never reaches the AI
    decline = imp.next_line("I can do Tuesday 17 November at 10 am. Would that work?", [])
    assert decline.startswith("No,") and "weekday" in decline and "from 3 pm" in decline
    assert seen == []  # the AI cannot turn the refusal into a yes either


def test_the_ai_caller_cannot_hang_up_mid_call():
    replies = iter(["END", "No, I need it after 3 pm."])
    imp = callers.ImprovisedCaller("a parent", 5, lambda m: next(replies))
    assert imp.next_line("I can do Tuesday at 10 am. Would that work?", []) == "No, I need it after 3 pm."
    quitter = callers.ImprovisedCaller("a parent", 5, lambda m: "END")
    with pytest.raises(RuntimeError):
        quitter.next_line("I can do Tuesday at 10 am. Would that work?", [])
    assert callers.ImprovisedCaller("a parent", 5, lambda m: "END").next_line("You're booked in. Goodbye!", []) is None
