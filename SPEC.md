# Voice-agent caller simulator: spec

## Where the idea came from
A sponsored post by gus (@igus_ai) on X, 5 Nov 2026 (https://x.com/igus_ai/status/2118400769695092769; 45.9K views,
378 likes when read on 2026-11-16), promoting a closed product: thousands of fake customers phone your voice agent
(they interrupt, mumble, get angry, change their minds) and every call is scored. Replies agreed with the need:
"everyone's voice demo sounds great until someone interrupts halfway". The post shows the idea and a product; no code.
Its money claims are the product's own and are not repeated here.

## What ours does
1. Fake callers from `scenarios.yaml`: scripted lines, or an AI-improvised caller (local model) from a persona. When
   an improvised caller has an `accepts` rule, plain code decides whether an offered time suits it.
2. Each caller line is spoken (Kokoro, a different voice per caller), then degraded like a real phone line
   (`phoneline.py`: 8 kHz, 300-3400 Hz, background noise at a chosen level, or a fast talker).
3. The agent under test hears that audio with its own speech-to-text and answers, exactly as in production.
4. Plain code scores every call (`checks.py`): the outcome, the booked day and time, words it must never say, the
   number of turns, and how much it misheard. An emergency that is not flagged is CRITICAL.
5. A scorecard (scenario x condition) with every failure spelled out; transcripts and audio for each call.
6. Each call ends PASS, FAIL (the agent broke a rule), KNOWN (a failure the scenario marks as a deliberate trade-off)
   or ERROR (the lab broke: the model could not load, the AI caller failed). Lab errors never count against the agent.

## The agent under test
The after-hours dental receptionist from the previous tutorial (https://github.com/eaqd/dental-voice-receptionist),
through a small adapter (`agents/dental.py`). Any voice agent can be tested by writing an adapter with the same
five members (greeting, hear, reply, done/outcome/booked).

## What building it found
- The receptionist didn't understand "after school" (it offered 10 am). Fixed in the receptionist.
- When its AI missed a time wish ("a slot after school on any weekday"), the receptionist repeated the same offer
  until the caller gave up. Fixed: plain code now reads day and time wishes from the caller's words too.
- A small model is a poor actor: it turned down 3 pm as "before 3 pm" and hung up mid-call. So plain code decides
  the improvised caller's yes or no, and a caller that quits mid-call counts as a lab error.
- With memory short, the local model failed to load and calls ended "hung up": now reported as ERROR, not FAIL.
- On the noisy phone line one misheard word ("tomorrow morning" heard as "tomorrow course") made the receptionist's
  AI call "I've chipped a tooth" an emergency; on a clean line it booked. Fixed: the AI's instructions now follow
  ordinary dental triage (a chipped tooth without pain is a booking).
- A fresh review found 13 problems before release, including a high one in the receptionist fix (it could turn
  down a time the caller had accepted). All fixed, each with a test.

## Limits (said plainly in the tutorial)
- Everything is local and free; a full run of 9 callers x 2 line conditions takes about 15-20 minutes on a CPU.
- Synthetic callers are cleaner than people: no interruptions, no talking over the agent (the agent is turn-based).
- Improvised callers add variety, not truth: the judge is always plain code.
- The noise is synthetic pink noise, not recorded street or office sound.
