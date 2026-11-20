"""The fake callers. A scripted caller says its lines in order; an improvised caller is played by a local AI that
gets a persona and a goal and answers whatever the agent just said (free, through Ollama)."""
import json
import os
import re
import urllib.request

from checks import WEEKDAYS, hours

ASKED_AGAIN = re.compile(r"didn't catch|say (it|that) again|repeat that|pardon", re.I)


class ScriptedCaller:
    """Says its lines in order, and repeats a line once when the agent didn't catch it, as people do (found on a
    noisy-line run: "Three is perfect" was heard as "The green is perfect" and the caller had nothing left to say)."""

    def __init__(self, lines: list[str]):
        self.lines, self.last, self.repeated = list(lines), None, False

    def next_line(self, agent_said: str, history: list) -> str | None:
        if self.last and not self.repeated and ASKED_AGAIN.search(agent_said or ""):
            self.repeated = True
            return self.last
        if not self.lines:
            return None
        self.last, self.repeated = self.lines.pop(0), False
        return self.last


def ollama_chat(messages: list[dict], model=None, host=None, timeout=120) -> str:
    body = {"model": model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b"), "stream": False,
            "options": {"temperature": 0}, "messages": messages}
    req = urllib.request.Request((host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")) + "/api/chat",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


CLOCK = re.compile(r"\bat\s+(\d{1,2})(?::(\d\d))?\s*(am|pm)\b", re.I)
NEGATED_SLOT = re.compile(r"\b(?:no|not|never|cannot|unavailable|taken|full|closed|booked|\w+n't)\b")
CLAUSE_BREAK = re.compile(r"[.?!;]|\b(?:but|however)\b|(?:,|\band\b)\s*(?=(?:i|we|no|not|"
                         + "|".join(WEEKDAYS) + r")\b)")


def offered(agent_said: str):
    """The (weekday, hour) the agent just offered, read by plain code, e.g. ("tuesday", 15.5); else None.
    A time the receptionist rules out is never an offer."""
    text = (agent_said or "").lower().replace("’", "'").replace("a.m.", "am").replace("p.m.", "pm")
    for clause in CLAUSE_BREAK.split(text):
        if NEGATED_SLOT.search(clause):
            continue
        for m in CLOCK.finditer(clause):
            days = [d for d in re.findall(r"[a-z]+", clause[:m.start()]) if d in WEEKDAYS]
            if days:
                hour = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
                return days[-1], hour + (int(m.group(2)) / 60 if m.group(2) else 0)
    return None


def fits(offer, accepts: dict) -> bool:
    day, hour = offer
    days = [d.lower() for d in accepts.get("days", WEEKDAYS[:5])]
    return day in days and hours(accepts.get("from", "0:00")) <= hour < hours(accepts.get("to", "24:00"))


def spoken_clock(value) -> str:
    """15:00 or 15 -> "3 pm": the caller's line is spoken aloud, and a person says "3 pm", not "fifteen hundred"."""
    h, m = divmod(round(hours(value) * 60) % (24 * 60), 60)
    return f"{(h - 1) % 12 + 1}{f':{m:02d}' if m else ''} {'am' if h < 12 else 'pm'}"


def decline(accepts: dict) -> str:
    days = [d.lower() for d in accepts.get("days", WEEKDAYS[:5])]
    when = "a weekday" if set(days) == set(WEEKDAYS[:5]) else " or ".join(d.title() for d in days)
    wish = f"No, I want a slot on {when}"
    if "from" in accepts and "to" in accepts:
        wish += f" between {spoken_clock(accepts['from'])} and {spoken_clock(accepts['to'])}"
    elif "from" in accepts:
        wish += f" from {spoken_clock(accepts['from'])}"
    elif "to" in accepts:
        wish += f" before {spoken_clock(accepts['to'])}"
    return wish + "."


class ImprovisedCaller:
    """An AI plays the caller and writes the words. Whether to accept an offered time is NOT left to it: a small
    model turned down 3 pm as "before 3 pm" on a real run. With an `accepts` rule, plain code answers the offer."""

    def __init__(self, persona: str, max_lines: int = 5, llm=ollama_chat, accepts: dict | None = None):
        self.persona, self.left, self.llm, self.accepts = persona, max_lines, llm, accepts

    def next_line(self, agent_said: str, history: list) -> str | None:
        if self.left <= 0:
            return None
        self.left -= 1
        if self.accepts and (offer := offered(agent_said)):
            if fits(offer, self.accepts):
                return "Yes, that works, thanks."  # a fitting offer gets a plain yes: no AI needed (it ignored "accept")
            return decline(self.accepts)
        convo = "\n".join(f"{'Receptionist' if who == 'agent' else 'You'}: {text}"
                          for who, text in history if who in ("agent", "caller said"))
        over = bool(re.search(r"\b(booked|goodbye)\b", agent_said or "", re.I))  # has the agent ended the call?
        for nudge in ("", "\nThe call is NOT over: answer the receptionist."):
            reply = self.llm([
                {"role": "system", "content": (
                    "You are role-playing a person phoning a dental practice after hours. Stay in character: "
                    f"{self.persona}. Answer only what the receptionist just asked, in one or two short spoken "
                    "sentences, with no stage directions. If the receptionist has booked you in or said goodbye, "
                    "reply with exactly END.")},
                {"role": "user", "content": f"The call so far:\n{convo}\n{nudge}\nWhat do you say next?"}])
            line = reply.strip().splitlines()[0] if reply.strip() else ""
            line = re.sub(r"^(you|caller|me)\s*:\s*", "", line, flags=re.I).strip().strip('"')
            if re.search(r"\bEND\W*$", line):  # "I'll take it. END": say the words; hang up only if the call is over
                line = re.sub(r"\s*\bEND\W*$", "", line).strip()
                if over:
                    self.left = 0
            if line or over:
                return line or None
        # the model kept hanging up mid-call: the caller failed, not the agent (the bench scores this as ERROR)
        raise RuntimeError("the AI caller hung up while the receptionist was still talking to it")


def make_caller(scenario: dict, llm=ollama_chat):
    if "improvise" in scenario:
        imp = scenario["improvise"]
        return ImprovisedCaller(imp["persona"], imp.get("max_lines", 5), llm, imp.get("accepts"))
    return ScriptedCaller(scenario["lines"])
