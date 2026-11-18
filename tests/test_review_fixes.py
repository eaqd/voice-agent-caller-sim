"""One test per problem a pre-release review found (2026-11-18). Offline, like the rest."""
from datetime import datetime

import numpy as np
import pytest

from test_bench import STARTED, FakeAgent, FakeCall, scenario

import bench  # noqa: E402
import callers  # noqa: E402
import checks  # noqa: E402
import phoneline  # noqa: E402


def test_each_call_starts_from_a_clean_folder(tmp_path):
    stale = tmp_path / "runs/polite-new-patient--text/agent/calendar.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('["2026-11-19T14:00"]')
    bench.run_one(scenario("polite-new-patient"), "text", {}, FakeAgent(), STARTED, tmp_path, text_only=True)
    assert not stale.exists()  # a booking from an earlier run can't make a slot look taken


def test_one_crash_costs_one_call_not_the_run(tmp_path):
    class CrashingCall(FakeCall):
        def reply(self, text):
            raise RuntimeError("voice model missing")
    agents = iter([type("A", (), {"new_call": lambda self, s, o: CrashingCall()})(), FakeAgent()])

    class Switch:
        def new_call(self, s, o):
            return next(agents).new_call(s, o)
    rows = bench.run_all([scenario("polite-new-patient"), scenario("angry-cancel")], {"text": {}}, Switch(), STARTED,
                         tmp_path, text_only=True)
    assert [r["status"] for r in rows] == ["ERROR", "PASS"]
    assert "the bench crashed" in rows[0]["fails"][0]
    assert "angry-cancel" in (tmp_path / "scorecard.md").read_text(encoding="utf-8")


def test_a_missed_emergency_is_a_failure_even_with_lab_errors(tmp_path):
    class BookingCall(FakeCall):
        infra_errors = 1

        def reply(self, text):  # books everything, emergencies included
            self.done, self.outcome, self.booked = True, "booked", datetime(2026, 11, 19, 14, 0)
            return "Booked."
    agent = type("A", (), {"new_call": lambda self, s, o: BookingCall()})()
    r = bench.run_one(scenario("mumbled-emergency"), "text", {}, agent, STARTED, tmp_path, text_only=True)
    assert r["status"] == "FAIL" and r["fails"][0].startswith("CRITICAL")
    assert any("model not available" in f for f in r["fails"])


def test_a_typo_in_a_day_rule_is_an_error_not_a_pass():
    with pytest.raises(ValueError):
        checks.check({"outcome": "booked", "day": "thu"}, {"outcome": "booked", "booked": datetime(2026, 11, 19, 14)},
                     STARTED)


def test_no_aliasing_when_the_line_drops_to_8_khz():
    rate = 24000
    t = np.arange(rate) / rate
    hiss = (np.sin(2 * np.pi * 5000 * t) * 0.5).astype(np.float32)  # 5 kHz would fold to 3 kHz without a filter
    y, r = phoneline.degrade(hiss, rate, {"rate": 8000})
    ts = np.arange(len(y)) / r
    folded = abs(np.dot(y, np.exp(-2j * np.pi * 3000 * ts))) / len(y)
    assert folded < 0.005  # a 0.25 tone would show as 0.25 at 3 kHz


def test_numbers_said_as_words_are_not_mishearings():
    assert checks.wer("Three o'clock is great", "3 o'clock is great") == 0
    assert checks.wer("hi", "hi there my friend") == 1.0  # capped


def test_accept_rules_read_any_way_of_writing_a_time():
    assert callers.fits(("tuesday", 9.5), {"from": "9:00", "to": "10:00"})
    assert callers.fits(("tuesday", 15.0), {"from": 900})  # YAML 1.1 turns an unquoted 15:00 into 900 minutes
    assert callers.fits(("tuesday", 15.0), {"from": "15:00", "days": ["Tuesday"]})


def test_the_offer_is_the_day_said_with_the_time():
    assert callers.offered("I can't do Monday, but I can do Tuesday at 3 pm.") == ("tuesday", 15.0)
    assert callers.offered("How about Thursday 19 November at 2:30 p.m.?") == ("thursday", 14.5)


def test_a_scripted_caller_repeats_itself_once_when_asked():
    c = callers.ScriptedCaller(["Three is perfect."])
    assert c.next_line("Would 3 pm work?", []) == "Three is perfect."
    assert c.next_line("Sorry, I didn't catch that. Would 3 pm work?", []) == "Three is perfect."  # repeated once
    assert c.next_line("Sorry, I didn't catch that. Would 3 pm work?", []) is None  # then gives up, like people do
