# HTTP API (v0.1)

The service is started with `python -m joons_listen serve`. Default address `http://127.0.0.1:18796` (`JOONS_LISTEN_PORT`). It binds to loopback only. All responses are UTF-8 JSON. Errors look like `{"ok":false,"code":"...","message":"..."}` with a matching HTTP status. Messages are written in Korean.

The MCP server (`python -m joons_listen mcp`) is a thin wrapper around these endpoints; see the tool table in the README.

## Endpoints

- `GET /health` returns `{ok, version, gpu:{name, vram_total_mb, vram_free_mb}|null, engines:{whisper:{ready,model}, qwen:{ready,model}}, llm:{ready,model}|null, queue:{running, queued}, store:{mirror, pending}, uptime_sec}`.
- `GET /models` returns the two engines (license, languages, loaded state, VRAM estimate) and the language routing.
- `POST /jobs` with `{kind:"transcribe"|"diarize", audio_path, language?:"auto"|"zh"|"ko"|"en", engine?:"auto"|"whisper"|"qwen", diarize?:true, num_speakers?:1..10, translate_ko?:bool, minutes?:bool, title?:string, dictionary?:[{from,to}]}` returns `202 {ok:true, job_id}`.
  - `kind:"diarize"` returns speaker turns only (no transcript).
  - `audio_path` must be inside an allowed folder (`PATH_NOT_ALLOWED` otherwise, `FILE_NOT_FOUND` if missing). A bare file name is searched for in the allowed folders.
- `POST /jobs/<id>/summarize` with `{style?:"minutes"|"brief", force?:bool}` returns `202 {ok:true, job_id:<new job id>}`: creates minutes or a summary from a finished transcription job.
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

`flags`: `uncertain` (low confidence, odd speech rate or repetition), `repeat` (repeated text, probably hallucination), `hint_echo` (the recognizer echoed the dictionary hints; the text was removed), `price`, `quantity`, `date`, `code` (a fact was extracted from this segment), `check_*` (a fact in a low-confidence segment).
