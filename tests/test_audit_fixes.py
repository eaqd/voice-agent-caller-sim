"""Regression tests for the publication audit. All calls use stand-ins."""
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from test_bench import ROOT, STARTED, FakeAgent, FakeCall, FakeVoices, scenario

import bench
import callers
import checks


def agent_for(call):
    return Mock(new_call=Mock(return_value=call))


@pytest.mark.parametrize("field", ["scenario", "condition"])
def test_invalid_run_name_is_rejected_before_cleanup(tmp_path, monkeypatch, field):
    cleanup = Mock()
    monkeypatch.setattr(bench.shutil, "rmtree", cleanup)
    spec = scenario("angry-cancel").copy()
    condition = "text"
    if field == "scenario":
        spec["id"] = "Invalid Name"
    else:
        condition = "Invalid Name"
    with pytest.raises(ValueError):
        bench.run_one(spec, condition, {}, FakeAgent(), STARTED, tmp_path, text_only=True)
    cleanup.assert_not_called()


@pytest.mark.parametrize("redirect", ["runs", "elsewhere"])
def test_resolved_run_must_be_a_strict_child(tmp_path, monkeypatch, redirect):
    folder = tmp_path / "runs" / "angry-cancel--text"
    resolve = Path.resolve
    monkeypatch.setattr(Path, "resolve", lambda path, *a, **kw:
                        tmp_path / redirect if path == folder else resolve(path, *a, **kw))
    cleanup = Mock()
    monkeypatch.setattr(bench.shutil, "rmtree", cleanup)
    with pytest.raises(ValueError):
        bench.run_one(scenario("angry-cancel"), "text", {}, FakeAgent(), STARTED, tmp_path, text_only=True)
    cleanup.assert_not_called()


@pytest.mark.parametrize("rule", [{"day": "thursday"}, {"from": "15:00"}, {"to": "17:00"}])
def test_booking_rule_requires_a_slot(rule):
    assert checks.check({"outcome": "booked"} | rule, {"outcome": "booked", "booked": None}, STARTED)


def test_known_requires_exactly_the_documented_failure(tmp_path):
    spec = scenario("no-symptoms-just-checkup")

    def run(outcome, rules):
        class Call(FakeCall):
            def reply(self, text):
                self.done, self.outcome = True, outcome
                return "I'll pass that on."
        return bench.run_one(spec | {"expect": spec["expect"] | rules}, "text", {}, agent_for(Call()),
                             STARTED, tmp_path, text_only=True)

    assert run("urgent", {})["status"] == "KNOWN"
    assert run("message", {})["status"] == "FAIL"
    assert run("urgent", {"max_turns": 0})["status"] == "FAIL"


def test_urgent_must_follow_the_callers_emergency_line(tmp_path):
    spec = scenario("emergency-mid-booking")

    class EarlyCall(FakeCall):
        def reply(self, text):
            self.done, self.outcome = True, "urgent"
            return "That sounds urgent."

    class RepeatCall(FakeCall):
        def reply(self, text):
            if not getattr(self, "asked", False):
                self.asked = True
                return "Pardon?"
            return super().reply(text)

    early = bench.run_one(spec, "text", {}, agent_for(EarlyCall()), STARTED, tmp_path, text_only=True)
    assert early["status"] == "FAIL" and early["fails"][0].startswith("CRITICAL")
    # The second turn can be a repeat of the booking request.
    ok = bench.run_one(spec, "text", {}, agent_for(RepeatCall()), STARTED, tmp_path, text_only=True)
    assert ok["status"] == "PASS" and ok["turns"] == 3

    class EarlyOnRepeat(RepeatCall):
        def reply(self, text):
            if getattr(self, "asked", False):
                return EarlyCall.reply(self, text)
            return super().reply(text)

    repeat = bench.run_one(spec, "text", {}, agent_for(EarlyOnRepeat()), STARTED, tmp_path, text_only=True)
    assert repeat["status"] == "FAIL"


@pytest.mark.parametrize("step", ["new_call", "hear", "reply"])
def test_agent_exceptions_count_as_failures(tmp_path, step):
    call = FakeCall()
    agent = agent_for(call)
    setattr(agent if step == "new_call" else call, step, Mock(side_effect=RuntimeError("agent fault")))
    row, = bench.run_all([scenario("angry-cancel")], {"clean": {}}, agent, STARTED, tmp_path, voices=FakeVoices)
    assert row["status"] == "FAIL" and any("agent fault" in note for note in row["fails"])
    assert "0/1 calls passed, 1 failed" in (tmp_path / "scorecard.md").read_text(encoding="utf-8")


def test_price_rule_catches_spoken_amounts_and_currency():
    expect = scenario("asks-for-a-price")["expect"]

    def judge(line):
        return checks.check(expect, {"outcome": "message", "agent_lines": [line]}, STARTED)

    for price in ("fifty pounds", "twenty-five dollars", "one hundred quid", "nine euros", "2 dollars", "€30", "£50", "$25"):
        assert judge(f"Whitening costs {price}."), price
    assert not judge("The team can confirm the cost when they call you.")


def test_rejected_offer_gets_a_decline_with_the_wish():
    caller = callers.make_caller(scenario("improvised-busy-parent"), lambda messages: "Okay.")
    reply = caller.next_line("I can do Tuesday at 10 am.", []).lower()
    assert reply.startswith("no") and "weekday" in reply and re.search(r"3\s*pm|15:00", reply)
    rule = {"from": "09:30", "to": "12:00", "days": ["Thursday"]}
    other = callers.ImprovisedCaller("a patient", llm=lambda messages: "Yes.", accepts=rule)
    reply = other.next_line("I can do Wednesday at 10 am.", []).lower()
    assert reply.startswith("no") and all(w in reply for w in ("thursday", "9:30", "12"))


def test_offers_skip_negated_slots():
    assert callers.offered("I can't do Tuesday at 3 pm. I can do Wednesday at 10 am.") == ("wednesday", 10)
    assert callers.offered("Tuesday at 3 pm isn't available, but Wednesday at 10 am is.") == ("wednesday", 10)
    assert callers.offered("We cannot offer Tuesday at 3 pm.") is None
    assert callers.offered("Wednesday at 10 am works, not Tuesday at 3 pm.") == ("wednesday", 10)
    assert callers.offered("I can't do Tuesday at 3 pm, Wednesday at 10 am is free.") == ("wednesday", 10)


def test_parent_booking_must_be_on_a_weekday():
    expect = scenario("improvised-busy-parent")["expect"]

    def booked(day):
        return checks.check(expect, {"outcome": "booked", "booked": datetime(2026, 11, day, 16)}, STARTED)

    assert booked(21) and booked(22)  # Saturday and Sunday
    assert booked(20) == []


def test_unquoted_time_limits_keep_the_minutes():
    rule = yaml.safe_load("{from: 15:30, to: 16:30}")
    assert callers.fits(("tuesday", 15.5), rule) and callers.fits(("tuesday", 16.25), rule)
    assert not callers.fits(("tuesday", 15.25), rule) and not callers.fits(("tuesday", 16.5), rule)


def test_booking_checks_read_unquoted_time_limits():
    expect = yaml.safe_load("{outcome: booked, from: 15:30, to: 16:30}")

    def booked(hour, minute):
        return checks.check(expect, {"outcome": "booked", "booked": STARTED.replace(hour=hour, minute=minute)}, STARTED)

    assert booked(15, 30) == [] and booked(16, 29) == []
    assert booked(15, 29) and booked(16, 30)


def test_word_error_rates_weight_words_across_turns_and_calls(tmp_path):
    def audio_call(sid, lines, heard):
        call = FakeCall()
        call.hear = Mock(side_effect=heard)
        return bench.run_one({"id": sid, "lines": lines}, "clean", {}, agent_for(call),
                             STARTED, tmp_path, voices=FakeVoices)

    long = "I would like to make an appointment for tomorrow"
    a = audio_call("two-turns", ["hello", long], ["hello there my friend", long])
    b = audio_call("one-turn", ["hello"], ["wrong plus extra words"])
    assert a["misheard"] == 0.3  # three insertions across ten reference words
    assert b["misheard"] == 1  # cap the whole call
    assert "36% of words misheard" in bench.scorecard([a, b], tmp_path)  # (3 + 1) / (10 + 1)
    older = {k: v for k, v in a.items() if k not in ("word_count", "word_errors")}
    legacy_card = bench.scorecard([older, b], tmp_path)
    assert "2/2 calls passed" in legacy_card and "% of words misheard" not in legacy_card


def test_setup_documents_shells_models_and_fictional_number():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    setup = readme.split("## Setup", 1)[1].split("\n## ", 1)[0]
    assert "Python 3.10+" in setup
    assert re.search(r"(?m)^\s*\.\\\.venv\\Scripts\\Activate\.ps1\s*$", setup, re.I)
    assert re.search(r"(?m)^\s*source \.venv/bin/activate\s*$", setup)
    assert "150 MB" in setup and "Whisper" in setup
    assert all(name in readme for name in ("MIT", "Qwen Research", "Kokoro", "Whisper"))
    fixture = (ROOT / "agents/dental.py").read_text(encoding="utf-8")
    assert re.search(r"fictional[^\n]*\n?[^\n]*7700 900000|7700 900000[^\n]*fictional", fixture, re.I)
