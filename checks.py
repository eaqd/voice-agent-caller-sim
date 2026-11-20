"""Plain code scores every call against the scenario's "expect" block. No AI judges the AI."""
import re
from datetime import date, datetime, time, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MAX_TURNS = 8


NUMBERS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def words(text: str) -> list[str]:
    """Lower-case words, with number words as digits: the voice says "three", Whisper writes "3"; that's no error."""
    found = re.findall(r"[a-z0-9']+", (text or "").lower().replace("’", "'"))
    return [NUMBERS.get(w, w) for w in found]


def word_errors(said: str, heard: str) -> tuple[int, int]:
    """Count edits and reference words before combining turns into a call."""
    a, b = words(said), words(heard)
    row = list(range(len(b) + 1))  # edit distance over words, one row at a time
    for i, wa in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, wb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (wa != wb))
    return row[-1], len(a)


def wer(said: str, heard: str) -> float:
    """Word error rate: how much of what the caller said the agent misheard (0 = perfect, 1 = all of it)."""
    errors, count = word_errors(said, heard)
    return min(1.0, errors / count) if count else float(bool(errors))


def hours(value) -> float:
    """A clock time as hours. YAML 1.1 reads unquoted 15:30 as 930 minutes; plain 15 still means 3 pm."""
    if isinstance(value, (int, float)):
        return float(value) if value <= 24 else value / 60
    if not isinstance(value, time):
        h, _, rest = value.strip().partition(":")
        if h == "24" and rest in ("", "00"):
            return 24.0
        value = time.fromisoformat(f"{h.zfill(2)}:{rest.zfill(2)}")
    return value.hour + value.minute / 60 + (value.second + value.microsecond / 1e6) / 3600


def clock_text(value) -> str:
    if isinstance(value, str):
        return value
    h, m = divmod(round(hours(value) * 60), 60)
    return f"{h:02}:{m:02}"


def expected_day(name: str, started: datetime) -> date | None:
    """'' -> no day rule; 'tomorrow' or a weekday name -> that date. Anything else is a typo in scenarios.yaml."""
    name = (name or "").lower()
    if not name:
        return None
    if name == "tomorrow":
        return started.date() + timedelta(days=1)
    if name in WEEKDAYS:
        return started.date() + timedelta(days=(WEEKDAYS.index(name) - started.weekday()) % 7 or 7)
    raise ValueError(f"unknown day {name!r} in an expect block: use tomorrow or a full weekday name")


def check(expect: dict, result: dict, started: datetime) -> list[str]:
    """Every rule the call broke, worst first. An empty list is a pass."""
    fails = []
    outcome, booked = result.get("outcome") or "hung up", result.get("booked")
    if expect.get("outcome") == "urgent" and outcome == "booked":
        fails.append("CRITICAL: an emergency was booked as a routine appointment")
    elif expect.get("outcome") == "urgent" and outcome != "urgent":
        fails.append(f"CRITICAL: an emergency ended as '{outcome}', not flagged urgent")
    elif expect.get("outcome") and outcome != expect["outcome"]:
        fails.append(f"expected {expect['outcome']}, got {outcome}")
    if outcome == "urgent" and (cue := expect.get("urgent_after")):
        lines = result.get("caller_lines", [])[:result.get("urgent_turn") or 0]
        if not any(cue.casefold() in line.casefold() for line in lines):
            fails.insert(0, f"CRITICAL: flagged urgent before the caller mentioned {cue}")
    if outcome == expect.get("outcome") == "booked" and booked is None and any(
            key in expect for key in ("day", "days", "from", "to")):
        fails.append("reported a booking without the slot needed to check it")
    if booked and expect.get("outcome") == "booked":
        day = expected_day(expect.get("day", ""), started)
        if day and booked.date() != day:
            fails.append(f"booked {booked:%a %d %b}, expected {day:%a %d %b}")
        if "days" in expect and WEEKDAYS[booked.weekday()] not in [d.lower() for d in expect["days"]]:
            fails.append(f"booked on {WEEKDAYS[booked.weekday()]}, caller wanted {' or '.join(expect['days'])}")
        t = booked.time()
        if "from" in expect and hours(t) < hours(expect["from"]):
            fails.append(f"booked at {t:%H:%M}, caller wanted from {clock_text(expect['from'])}")
        if "to" in expect and hours(t) >= hours(expect["to"]):
            fails.append(f"booked at {t:%H:%M}, caller wanted before {clock_text(expect['to'])}")
    for pattern in expect.get("never_say", []):
        hit = next((line for line in result.get("agent_lines", []) if re.search(pattern, line, re.I)), None)
        if hit:
            fails.append(f"said something it must not: {hit[:80]!r}")
    if result.get("turns", 0) > expect.get("max_turns", MAX_TURNS):
        fails.append(f"took {result['turns']} turns (limit {expect.get('max_turns', MAX_TURNS)})")
    return fails
