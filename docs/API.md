# HTTP API (v0.1)

The service is started with `python -m joons_listen serve`. Default address `http://127.0.0.1:18796` (`JOONS_LISTEN_PORT`). It binds to loopback only. All responses are UTF-8 JSON. Errors look like `{"ok":false,"code":"...","message":"..."}` with a matching HTTP status. Error messages are in English.

The MCP server (`python -m joons_listen mcp`) is a thin wrapper around these endpoints; see the tool table in the README.

## Endpoints

- `GET /health` returns `{ok, version, gpu:{name, vram_total_mb, vram_free_mb}|null, engines:{whisper:{ready,model}, qwen:{ready,model}}, llm:{ready,model}|null, queue:{running, queued}, store:{mirror, pending}, uptime_sec}`.
- `GET /models` returns the two engines (license, languages, loaded state, VRAM estimate) and the language routing.
- `POST /jobs` with `{kind:"transcribe"|"diarize", audio_path, language?:"auto"|"zh"|"ko"|"en", engine?:"auto"|"whisper"|"qwen"|"ensemble", diarize?:true, num_speakers?:1..12, translate_ko?:bool, minutes?:bool, title?:string, dictionary?:[{from,to}], customer?:string}` returns `202 {ok:true, job_id}`.
  - `kind:"diarize"` returns speaker turns only (no transcript).
  - `audio_path` must be inside an allowed folder (`PATH_NOT_ALLOWED` otherwise, `FILE_NOT_FOUND` if missing). A bare file name is searched for in the allowed folders.
- `POST /jobs/<id>/summarize` with `{style?:"minutes"|"brief", force?:bool}` returns `202 {ok:true, job_id:<new job id>}`: creates minutes or a summary from a finished transcription job.
- `POST /jobs/<id>/crosscheck` with `{notes?:string, references?:[{name, text}], arbitrate?:bool, customer?:string}` returns `202 {ok:true, job_id}`: a new job with the same segments after the checks, plus `crosscheck:{customer, references, reference_vote, llm, notes}`. Read the report with `format=crosscheck`. At least one of notes, references or arbitrate is required; up to 5 references of 100000 characters each; the request body stays under 1 MB.
- `GET /jobs` returns the 50 most recent jobs.
- `GET /jobs/<id>` returns `{ok, job_id, kind, state:"queued"|"running"|"done"|"error"|"cancelled", stage, progress:0..100, error?:{code,message}, engine?, language?, duration_sec?, speakers?, title, created_at, updated_at, outputs:{json?,txt?,md?,srt?,minutes?}, mirror, warnings:[..]}`.
- `GET /jobs/<id>/result?format=json|txt|md|srt|minutes&offset=0&max_chars=20000` returns `{ok, job_id, format, offset, next_offset|null, total_chars, text}`. `409 NOT_READY` if the job is not finished. `max_chars` is clamped to 1000..200000.
- `POST /jobs/<id>/cancel` returns `{ok, state}`.

## Stages

`ingest` then `vad`, `diarize`, `asr`, `verify`, `translate`, `minutes`, `store`. Progress is weighted by stage. Every stage writes a checkpoint in the job folder, so a restarted service resumes unfinished jobs.

## Error codes

`BAD_REQUEST`, `PATH_NOT_ALLOWED`, `FILE_NOT_FOUND`, `NOT_FOUND`, `NOT_READY`, `ENGINE_UNAVAILABLE`, `LLM_UNAVAILABLE`, `GPU_OOM`, `AUDIO_DECODE_FAILED`, `CANCELLED`, `INTERNAL`. The MCP server adds `SERVICE_DOWN` when it cannot reach the service.

## Result JSON (`format=json`)

```text
{ "job_id", "title", "source": {"path","sha256","duration_sec"}, "engine", "language", "created_at",
  "speakers": [{"id":"S1","label":"화자 1","total_sec":123.4}],
  "segments": [{"i":0,"start":0.0,"end":4.2,"speaker":"S1","lang":"zh","text":"original","text_ko":"Korean translation|null",
                "confidence":0.0..1.0|null,"flags":["uncertain","repeat","hint_echo","price","quantity","date","code","check_price"],
                "audio":{"start":0.0,"end":4.2}}],
  "facts": [{"type":"price|quantity|date|code","text":"...","segment":3,"verify":true}],
  "uncertain": [3,7],
  "minutes": {"title","summary","decisions":[..],"actions":[{"who","what","due"}],"open_questions":[..],"llm":"model|null"} | null }
```

`flags`: `uncertain` (low confidence, odd speech rate or repetition), `repeat` (repeated text, probably hallucination), `hint_echo` (the recognizer echoed the dictionary hints; the text was removed), `engines_disagree` (ensemble only: Qwen and Whisper agree on less than 60% of the tokens), `price`, `quantity`, `date`, `code` (a fact was extracted from this segment), `check_*` (a fact in a low-confidence segment).

## Ensemble mode (`engine: "ensemble"`)

Qwen3-ASR and Whisper transcribe the same speaker turns one after the other (one model on the GPU at a time, about twice the time of a single engine). Qwen is the primary text. Where the two disagree, the Whisper span is taken only when it contains a dictionary term or a number and the Qwen span does not; text that only one engine produced follows Qwen. Digit-by-digit Chinese readings of three or more characters (六五零六) are written as Arabic digits (6506) before comparison. Each segment then carries `alts:{qwen,whisper}`, `agree` (0..1) and `conflicts:[{primary,secondary,pick}]`, and the result has `ensemble:{engines, primary, mean_agree, segments_disagree, picked_secondary}`.

## Customer dictionaries (`customer`)

`customer: "acme"` loads `<data>/dictionaries/acme.json` (same format as the global dictionary) in front of the global one. Names are plain words (letters, digits, `_`, `-`). Keep these files private: they hold customer names and product codes.

## Cross-check (`POST /jobs/<id>/crosscheck`)

Runs on a finished transcription (ensemble results get the most out of it). Nothing is confirmed automatically; every change records `resolved_by`.

1. `references`: other transcripts of the same recording, for example from a cloud assistant. For each ensemble conflict, a candidate that appears in a reference while the other does not is chosen (`resolved_by: "reference:<name>"`). Numbers are compared after normalisation (一千四百五十 = 1450, 零点九六五 = 0.965).
2. `arbitrate: true`: the local LLM picks A/B for up to 60 conflicts the references could not decide (`resolved_by: "llm"`); unsure answers keep the current text.
3. `notes`: every line of the meeting notes that contains a number, a product code or a dictionary term is marked `ok`, `partial` or `not_found` with the times where each item was heard, and numbers and codes heard in the meeting but missing from the notes are listed (`missing_from_notes`).
