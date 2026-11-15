# JOONS LISTEN MCP

Local meeting transcription for MCP clients: Whisper large-v3 and Qwen3-ASR-1.7B on your own GPU, with speaker labels, fact flags and optional Korean translation and minutes.

> Part of the [JOONS ecosystem](https://github.com/dbdefault/joons-ecosystem): MCP-Based AI Infrastructure Ecosystem.

Korean: [README.ko.md](README.ko.md)

Status: 0.1.0 Preview. Read [LIMITATIONS.md](LIMITATIONS.md) before relying on it.

## At a glance

- **What**: a small local service plus a stdio MCP server. You give it a meeting or call recording; it returns a transcript with speaker turns, timestamps, "check this" flags on numbers, prices, dates and product codes, and optionally a Korean translation and minutes written by a local LLM.
- **For**: people who record meetings or calls (Chinese, Korean, English, mixed) and want the audio to stay on their own machine, and want an MCP client such as Claude Desktop or Claude Code (anything that can start a local stdio server) or a script to drive it. ChatGPT cannot start local stdio servers; it needs a remote MCP endpoint, so put the service behind an MCP gateway you control.
- **Not for**: real-time captions, cloud processing, legal-grade transcripts, or machines without a CUDA GPU (it is tested only on an 8 GB NVIDIA card; CPU-only use is untested).
- **How**: install the dependencies, start the service, point your MCP client at `python -m joons_listen mcp`, then ask the client to transcribe a file.
- **Trust**: the service binds to loopback only, reads audio only from folders you allow, and sends meeting text only to a local LLM. Every number it extracts is marked "verify": nothing is auto-confirmed. See [SECURITY.md](SECURITY.md).

## What it does

```text
audio file
  -> ffmpeg decode (16 kHz mono)
  -> silero-vad speech segments (gaps the VAD misses are filled, see LIMITATIONS)
  -> speaker turns: ECAPA embeddings + agglomerative clustering
  -> ASR: Qwen3-ASR-1.7B (zh, ko, mixed) or Whisper large-v3 (en), chosen per language
  -> fact extraction (price, quantity, date, code) + uncertainty flags
  -> optional Korean translation and minutes via a local Ollama model
  -> txt / md / srt / json / minutes files
```

- Two ASR engines only, loaded one at a time so an 8 GB GPU is enough.
- Jobs run one by one in a queue and survive restarts: each stage writes a checkpoint, so a stopped job resumes instead of starting over.
- A private dictionary (names, replacement rules) biases recognition and fixes recurring mistakes.
- Hallucination guards: repeated-text detection, a filter for dictionary hints echoed back as text on noisy audio, low-confidence flags.

## Requirements

- Python 3.10 or newer, ffmpeg on PATH (or `JOONS_LISTEN_FFMPEG`).
- An NVIDIA GPU with roughly 8 GB of memory and a CUDA build of PyTorch.
- Python packages: `numpy`, `soundfile`, `scikit-learn`, `torch`, `faster-whisper`, `qwen-asr`, `silero-vad`, `speechbrain`, `huggingface_hub`.
- Optional, for translation and minutes: [Ollama](https://ollama.com) with a local model (default `qwen3:8b`).
- Model weights (about 10 GB) are downloaded from Hugging Face on first use into the data folder. They are not part of this repository.

## Install and run

```sh
git clone https://github.com/dbdefault/joons-listen-mcp
cd joons-listen-mcp
pip install -e .
pip install torch faster-whisper qwen-asr silero-vad speechbrain huggingface_hub   # pick the torch build for your CUDA

python -m joons_listen check      # GPU, engines, local LLM
python -m joons_listen serve      # the HTTP service on 127.0.0.1:18796
```

The package is installed from source; it is not published to PyPI.

Without an MCP client you can process one file directly (stop the service first, both would share one job database):

```sh
python -m joons_listen run meeting.m4a --translate --minutes
```

## Connect an MCP client

The MCP server speaks JSON-RPC over stdio and only calls the running service on loopback. Example client configuration (also in `examples/mcp-client-config.json`):

```json
{
  "mcpServers": {
    "joons-listen": {
      "command": "python",
      "args": ["-m", "joons_listen", "mcp"],
      "env": { "JOONS_LISTEN_URL": "http://127.0.0.1:18796" }
    }
  }
}
```

Tools:

| Tool | Purpose | Read-only |
|---|---|---|
| `listen_transcribe` | queue a transcription job, returns `job_id` immediately | no |
| `listen_diarize` | queue a speaker-turn job (no transcript) | no |
| `listen_summarize` | minutes or brief summary from a finished job | no |
| `listen_status` | one job's progress, or service health and recent jobs | yes |
| `listen_result` | read a result in chunks (`json`, `txt`, `md`, `srt`, `minutes`) | yes |
| `listen_models` | engines, state and language routing | yes |

Long recordings take time (a 56-minute recording took about 20 minutes on an RTX 4060 Ti), so the tools never wait: call `listen_status` until the state is `done`, then `listen_result`.

## Allowed folders

The service reads only audio files inside allowed folders. The default is the data folder's `inbox`, `~/Desktop` and `~/Downloads`. Set `JOONS_LISTEN_ROOTS` (a path-separator list) to change it. A bare file name is looked up inside the allowed folders. Files whose names look like secrets are refused.

## Configuration

All settings are environment variables. The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `JOONS_LISTEN_DATA` | `~/.joons-listen` | jobs, database, models, dictionary |
| `JOONS_LISTEN_PORT` | `18796` | service port (loopback only) |
| `JOONS_LISTEN_ROOTS` | inbox, Desktop, Downloads | allowed audio folders |
| `JOONS_LISTEN_URL` | `http://127.0.0.1:18796` | where the MCP server finds the service (loopback only) |
| `JOONS_LISTEN_LLM_URL` / `JOONS_LISTEN_LLM_MODEL` | Ollama on loopback / `qwen3:8b` | translation and minutes |
| `JOONS_LISTEN_MIRROR_DIR` | empty (off) | copy finished results (no audio) as zip files to this folder |
| `JOONS_LISTEN_DICT` | `<data>/dictionary.json` | private dictionary, see `examples/dictionary.example.json` |

## Evaluation

[docs/EVALUATION.md](docs/EVALUATION.md) has error rates and speed on a public speech set, and a field test on a hard far-field phone recording. `tools/eval_fleurs.py` and `tools/eval_recording.py` reproduce them. The measurements are small and meant to find weaknesses, not to rank the two engines.

## Documentation

- [docs/API.md](docs/API.md): HTTP API of the service
- [docs/EVALUATION.md](docs/EVALUATION.md): measurements and what they mean
- [LIMITATIONS.md](LIMITATIONS.md): known weaknesses
- [SECURITY.md](SECURITY.md): trust model and reporting
- [CHANGELOG.md](CHANGELOG.md)

## Third-party models

| Component | License |
|---|---|
| Whisper large-v3 (CTranslate2 build `Systran/faster-whisper-large-v3`) | MIT |
| Qwen3-ASR-1.7B | Apache-2.0 |
| SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) | Apache-2.0 |
| silero-vad | MIT |
| FLEURS (evaluation data, not included) | CC-BY-4.0 |

Check each upstream license yourself before redistributing weights. This repository contains none.

## License

Apache License 2.0, see [LICENSE](LICENSE).
