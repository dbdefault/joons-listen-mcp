# Limitations

This is a 0.1.0 preview. These are the weaknesses we measured or know about.

## Audio quality

- **Far-field recordings are hard.** On a 56-minute phone recording made from across a room, silero-vad marked about 21 minutes of ordinary conversation as "no speech" and did not improve when its threshold was lowered from 0.5 to 0.35. The service now fills VAD gaps longer than 30 seconds with fixed 25-second chunks (chunks quieter than -50 dBFS are skipped). This raised coverage of detected speech from 31% to 84% and the recall of a human-written minutes checklist from 38% to 69% (see [docs/EVALUATION.md](docs/EVALUATION.md)).
- **Numbers and codes suffer first.** In the same test, most missed items were numbers (rates, quantities, counts) and product codes. On noisy audio treat them as hints, never as facts. The service flags every extracted number, price, date and code as "verify".
- **Gap filling can add noise text.** Chunks that contain only room noise can produce short junk text. A filter removes the case where the recognizer echoes the dictionary hint list back as text, but other noise artifacts remain possible.

## Speakers

- Speaker separation works for clearly different voices. Similar voices and overlapping speech are weak. The speaker count of a long, noisy recording can be wrong (the 6-minute tail of the same meeting came out as 3 speakers in one run and 4 in another). Pass `num_speakers` when you know it.
- Speaker labels are `S1`, `S2`, ... per job. They are not identities.

## Recognition and translation

- Only two ASR engines are supported: Whisper large-v3 and Qwen3-ASR-1.7B. The default routing (Qwen for Chinese, Korean and mixed speech, Whisper for English) is a starting point, not a verified ranking: the evaluation sets are small.
- Qwen3-ASR gives no confidence scores through the library we use, so low-confidence flags come mostly from Whisper and from timing and repetition checks.
- Chinese dialects such as Cantonese, and heavy code-switching inside one sentence, were not tested.
- Translation and minutes are written by a small local model (default 8B). They are drafts: check them against the transcript. The model sometimes needs help with currencies (a post-processing rule turns "원" after a Chinese number into "위안").

## Platform

- Tested on Windows 11 with one NVIDIA GPU (8 GB). Linux and macOS are expected to work for the pure-Python parts and the test suite, but GPU use there is untested. CPU-only use is untested.
- One job runs at a time. The GPU holds one ASR engine at a time.
- Error messages and documentation are in English. Translation and minutes are produced in Korean by design, so the minutes Markdown layout, its headings and the speaker labels (`화자 1`) are Korean. Source code comments are mostly Korean.

## Not included

- No speaker identification by name, no real-time mode, no cloud engines.
- No authentication. The service is meant to be reachable from the local machine only (see [SECURITY.md](SECURITY.md)).
