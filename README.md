# I built fake callers that phone my AI receptionist, and they found 4 bugs

> Idea: a post by gus (@igus_ai) on X about a product where thousands of fake customers phone your voice agent and
> every call gets scored (https://x.com/igus_ai/status/2118400769695092769). The product is closed. This is a full
> open build of the idea, written from scratch: free tools, every line explained, runs on a laptop. The agent it
> tests is my after-hours dental receptionist (https://github.com/eaqd/dental-voice-receptionist).

## What you get

Every voice-agent demo sounds great. Then a real caller mumbles, changes their mind, rings from a noisy street,
or slips an emergency into the middle of a booking.

This bench phones your agent with fake callers before real ones do:

- **Callers** in `scenarios.yaml`: scripted lines, or an AI that improvises from a persona ("a busy parent, short
  answers, wants a slot after school").
- **Phone lines**: every line is spoken by a free voice (Kokoro), then degraded like a real call: 8 kHz, only
  300-3400 Hz gets through, background noise, or a fast talker. The agent hears it with its own speech-to-text.
- **A judge that is plain code**: did it book the right day and time, flag the emergency, avoid quoting a price,
  finish in a sensible number of turns? An emergency that isn't flagged is CRITICAL.
- **A scorecard**: every caller on every line, with the reason for each failure.

The real scorecard from the run in the video (9 callers x 2 phone lines, on a laptop CPU):

```
| scenario | clean | noisy-phone |
|---|---|---|
| polite-new-patient | PASS | PASS |
| changes-mind | PASS | PASS |
| vague-caller | PASS | PASS |
| mumbled-emergency | PASS | PASS |
| emergency-mid-booking | PASS | PASS |
| angry-cancel | PASS | **FAIL** |
| asks-for-a-price | PASS | PASS |
| no-symptoms-just-checkup | known | known |
| improvised-busy-parent | PASS | PASS |

15/18 calls passed, 1 failed (0 critical: an emergency not flagged), 2 known trade-off.
- clean: 8/9 passed, 6% of words misheard, 18.5 s per turn
- noisy-phone: 7/9 passed, 24% of words misheard, 17.4 s per turn
```

This run came before the independent audit described under "What I tested". The audit's fixes make the judge
stricter and have plain code speak the caller's "no" to a time that doesn't suit. A full audio re-run with those
fixes is still to come (the PC it runs on was out of memory), and I'll add its scorecard here.

## What it found in my own agent

The receptionist from the last tutorial had 44 passing tests and a fresh code review. The fake callers still found:

1. **"After school" meant nothing to it.** An improvised caller asked for "the earliest slot after school" and was
   offered 10 am.
2. **It looped.** When its AI missed a time wish ("I want a slot after school on any weekday"), it repeated the
   same offer until the caller gave up. Now plain code reads day and time wishes from the caller's own words too,
   but only clear wishes ("I want", "how about"), never a "yes".
3. **One misheard word changed the outcome.** On the noisy line "tomorrow morning" became "tomorrow course", and
   the AI called "I've chipped a tooth" an emergency. On the clean line it booked. Its instructions now follow
   ordinary dental triage.
4. **Noise ate an angry caller's words, and the AI over-reacted.** On the noisy line only "This is ridiculous"
   was heard of "This is ridiculous, I've been trying to reach you all day. I need to cancel my appointment on
   Wednesday.", and the AI called it an
   emergency. I tried telling it that anger isn't an emergency; the small model ignored that. I didn't let code
   overrule it, because the rule is that the AI can raise the alarm but never lower it, and a false alarm is the
   safe direction. So it stays a FAIL on the scorecard: the next thing to fix.

And one deliberate trade-off: it treats "no pain or swelling" as urgent, because it errs towards urgent. The bench
marks that KNOWN, so it stays visible without failing the run.

Three of the four are fixed in the receptionist: the first two with unit tests, the third in the AI's instructions
(a prompt has no unit test, so the bench's own run is its check). In the run above, the bench passes all three. On the noisy line
there's a nice moment in that run: "Three is perfect" was heard as "The green is perfect", the receptionist
asked again, the caller repeated it, and the call was booked.

The evidence is in the repo: every call's transcript from the run above is in `evidence/video-run/`
(`changes-mind--noisy-phone.txt` has "tomorrow course" and the "green is perfect" moment,
`angry-cancel--noisy-phone.txt` the swallowed cancellation), and the chipped tooth called urgent, before the fix,
is in `first-audio-run-output.txt`.

## Two honest answers before we build

**"Why not let an AI judge the calls?"** Because then you're testing one AI with another. The callers can be AI,
but every verdict comes from plain code reading the outcome, the booking and the transcript.

**"Aren't fake callers too clean?"** Cleaner than people, yes: no interruptions, no talking over the agent. That's
why the phone line is degraded and the callers include mumbling, mind-changing and hidden emergencies. It finds
the bugs you can find before launch; it doesn't replace listening to real calls.

## Setup (about 10 minutes)

1. Clone this repo and the agent under test side by side:
   ```
   git clone https://github.com/eaqd/voice-agent-caller-sim
   git clone https://github.com/eaqd/dental-voice-receptionist
   ```
2. Use Python 3.10+ in `voice-agent-caller-sim`. In PowerShell, run

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   On macOS or Linux, run

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Free local AI: install Ollama (https://ollama.com), then `ollama pull qwen2.5:3b`. Voice: download
   `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the "model-files-v1.0" release of
   https://github.com/thewh1teagle/kokoro-onnx/releases and set `KOKORO_DIR` in `.env` (copy `.env.example`).
   The first audio run downloads the Whisper model (about 150 MB).
4. Run it:
   ```
   python bench.py --text          # the agent's decisions only, about 2 minutes
   python bench.py                 # every caller on a clean and a noisy phone line, about 20 minutes
   ```
   Then open `out/scorecard.md`, and `out/runs/<caller>--<line>/` for each call's transcript and audio.

The MIT licence covers this code. The separately downloaded models keep their own licences, including the Qwen Research licence for `qwen2.5:3b` and the respective Kokoro and Whisper licences.

## Step 1: the callers (`scenarios.yaml`)

```yaml
  - id: changes-mind
    voice: bm_lewis
    lines:
      - Hello, it's Tom Reid. I've chipped a tooth and I'd like it looked at tomorrow morning.
      - Actually, mornings are bad for me. Anything after three?
      - Three is perfect.
    expect: {outcome: booked, day: tomorrow, from: "15:00", to: "17:00"}
```

Nine callers in all: polite, mind-changing, vague, a mumbled emergency, an emergency slipped into a booking, an
angry cancellation, someone asking for a price, someone saying "no swelling", and an improvised busy parent.

## Step 2: the phone line (`phoneline.py`)

A phone line keeps 300-3400 Hz at 8 kHz, plus whatever noise is around the caller. Two plain filters do it:
a band-pass (a convolution, after an anti-alias filter so high sounds don't fold into the voice band) and pink
noise at a chosen level below the voice. No `numpy.fft`: on the PC this was built on, Windows Smart App Control
blocked numpy 2.5's FFT library, and a convolution runs anywhere.

## Step 3: callers that can't cheat (`callers.py`)

An improvised caller is a small local model with a persona. It was a poor actor: it turned down 3 pm as "before
3 pm" and hung up mid-call. So it only writes the words. Whether an offered time suits it is decided by plain
code from an `accepts` rule, and a caller that quits mid-call counts as a lab error, not as the agent's fault.
Scripted callers repeat a line once when the agent didn't catch it, like people do.

## Step 4: the judge (`checks.py`)

Plain code checks the outcome, the day ("tomorrow" or a weekday), the time window, phrases the agent must never
say (a price), and the number of turns. It also measures how much of each line the agent misheard (word error
rate, with "three" and "3" counted as the same).

## Step 5: four kinds of result (`bench.py`)

PASS, FAIL (the agent broke a rule), KNOWN (a failure the scenario marks as a deliberate trade-off) and ERROR (the
lab broke: the model couldn't load, the AI caller failed). That split matters: on a busy PC the local model once
failed to load for lack of memory, and without it those calls would have counted against the agent.

## Plug in your own agent (`agents/dental.py`)

The bench needs five things from an agent: a greeting, `hear(wav)`, `reply(text)`, and `done` / `outcome` /
`booked`. The dental adapter is 40 lines; write one like it for your own agent.

## What I tested, and what I didn't

- `pytest`: 42 offline tests with stand-in agents, voices and AI: the scoring rules, the phone-line filters
  (measured, without an FFT), the callers, ERROR vs FAIL vs KNOWN, crash safety, and the adapter driving the real
  receptionist. The receptionist's own tests went from 44 to 54 with the fixes the bench and the audit below found.
- The run above: real speech (Kokoro), real phone-line degradation, the receptionist's real speech-to-text
  (faster-whisper) and local model (Ollama `qwen2.5:3b`), on a CPU-only Windows PC.
- A fresh review before release found 13 problems in the bench and in my receptionist fix (the worst: the fix
  could turn down a time the caller had accepted). All fixed, with 9 tests in `tests/test_review_fixes.py`.
- Then a second AI model (OpenAI Codex) audited both repos before publishing and found 13 more: a bad scenario name
  could make the cleanup delete files outside `out/runs`; the judge passed a booking with no slot, a Saturday
  booking for the weekday-only parent, and an emergency flagged before the caller mentioned it; an agent crash
  was scored as a lab error; the caller misread "I can't do Tuesday at 3 pm" as an offer; spoken prices and
  unquoted `15:30` in YAML slipped through; and my receptionist fix read "I don't want mornings" as a wish for
  mornings. All fixed, each with a test that fails without its fix (`tests/test_audit_fixes.py` in both repos).
- Not tested: real people, interruptions (the agent is turn-based), recorded street or office noise (the noise is
  synthetic), and a live phone line.

## Extend it

- Add callers by writing YAML: an accent (Kokoro has British and American voices), a caller who interrupts, a
  caller who gives a wrong name and corrects it.
- Run the full grid (`--conditions all`) before every change to your agent, and keep the scorecards.
- Test a different agent: write an adapter.

Credit: the idea from a post by gus (@igus_ai) on X. Code: this repo, MIT licence. Agent under test:
https://github.com/eaqd/dental-voice-receptionist
