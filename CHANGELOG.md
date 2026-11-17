# Changelog

## Unreleased

- `POST /jobs/<id>/crosscheck` and the `listen_crosscheck` MCP tool: reference transcripts vote on ensemble conflicts, optional local-LLM arbitration, and a line-by-line meeting-notes check with a `crosscheck.md` report.
- Ensemble results record the `customer` used, so a later cross-check reuses that dictionary.
- `engine: "ensemble"`: run Qwen3-ASR and Whisper on the same turns and merge them with dictionary and number evidence; segments report `alts`, `agree` and `conflicts`.
- `customer`: per-customer dictionaries in `<data>/dictionaries/<name>.json`, loaded before the global dictionary.
- `JOONS_LISTEN_MIRROR_CMD`: send finished results with any command (JSON array, no shell) instead of a folder copy.
- A job database from before the `mirror` rename (column `brain_sync`) is migrated in place on start.
- Error and warning messages of the service, the CLI and the MCP server are now in English. Minutes output stays Korean by design.
- Documentation of `num_speakers` now matches the code (1 to 12).

## 0.1.0 (preview)

First public release, extracted from a private deployment where it ran for real meeting recordings.

- Local HTTP service (loopback only) with a persistent single-worker job queue and per-stage checkpoints (ingest, VAD, diarize, ASR, verify, translate, minutes).
- Two ASR engines: Whisper large-v3 (faster-whisper) and Qwen3-ASR-1.7B, routed by language.
- Speaker turns from ECAPA embeddings with agglomerative clustering, cut at 3-second windows.
- Fact extraction for prices, quantities, dates and product codes, including Chinese numerals (for example 两万一吨), with a `verify` flag on everything.
- Optional Korean translation and minutes through a local Ollama model, with a currency fix for Chinese sources.
- Private dictionary: recognition hints and replacement rules.
- Far-field handling: VAD gaps longer than 30 seconds are filled with 25-second chunks; hint-list echo filter.
- Standalone stdio MCP server with six tools (`listen_transcribe`, `listen_diarize`, `listen_summarize`, `listen_status`, `listen_result`, `listen_models`).
- Optional result mirroring to a folder as checksummed zip files.
- Evaluation tools for FLEURS and for a checklist-based recall test on real recordings.
