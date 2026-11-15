# Changelog

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
