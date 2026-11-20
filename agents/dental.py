"""Adapter for the agent under test: the after-hours dental receptionist from the previous tutorial
(https://github.com/eaqd/dental-voice-receptionist). Clone it next to this folder, or set AGENT_DIR.

Any voice agent can be tested by writing an adapter like this one. The bench only needs:
  agent.new_call(started, out_dir) -> a call with .greeting, .hear(wav) -> text, .reply(text) -> text,
                                      .done, .outcome ("booked" | "urgent" | "message" | ""), .booked (datetime | None)
"""
import importlib
import os
import sys
from pathlib import Path


class DentalCall:
    def __init__(self, agent, started, out_dir: Path):
        self.agent, self.out = agent, Path(out_dir)
        self.call = agent.desk.Call("+44 7700 900000", started)  # fictional UK mobile number for the simulator
        self.greeting = self.call.said = agent.rules["greeting"]

    def hear(self, wav_path) -> str:
        return self.agent.ears.hear(wav_path)  # the agent's own speech-to-text, exactly as in production

    infra_errors = 0  # AI calls that failed (e.g. the model could not load): the lab's fault, not the agent's

    def reply(self, text: str) -> str:
        failed = []

        def counted(*args):
            try:
                return self.agent.ai(*args)
            except Exception:
                failed.append(1)
                raise  # the receptionist handles it as in production (its safety rules still run)
        reply = self.agent.receptionist.answer(self.call, text, self.agent.rules, counted, self.out)
        self.infra_errors += len(failed)
        return reply

    done = property(lambda self: self.call.done)
    outcome = property(lambda self: self.call.outcome)
    booked = property(lambda self: self.call.booked)


class DentalAgent:
    def __init__(self, repo=None, ai=None):
        repo = Path(repo or os.getenv("AGENT_DIR") or Path(__file__).resolve().parents[2] / "dental-voice-receptionist")
        if not (repo / "desk.py").exists():
            raise FileNotFoundError(f"dental receptionist not found at {repo}: clone "
                                    "https://github.com/eaqd/dental-voice-receptionist next to this folder")
        sys.path.append(str(repo))  # appended, so the agent's files can never shadow the bench's own
        self.desk, self.receptionist, self.brain = (importlib.import_module(m) for m in ("desk", "receptionist", "brain"))
        self.ears = importlib.import_module("ears_voice")
        self.rules = self.desk.load(repo / "clinic.yaml")
        self.ai = ai or self.brain.backend(os.getenv("AI_BACKEND", "ollama"))

    def new_call(self, started, out_dir) -> DentalCall:
        return DentalCall(self, started, out_dir)
