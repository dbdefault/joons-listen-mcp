# Evaluation

Measured on one machine: Windows 11, NVIDIA RTX 4060 Ti 8 GB, torch 2.9.1 (CUDA 12.6). The goal is to find weaknesses of this system, not to rank Whisper against Qwen. The samples are small: differences of about 2 percentage points can be noise.

## 1. Public speech set (FLEURS)

Data: google/fleurs dev, 40 sentences per language (Mandarin, Korean, English), fixed seed 7, tool `tools/eval_fleurs.py`. Metric: character error rate (CER) for Chinese and Korean, word error rate (WER) for English, after removing punctuation and normalizing to lowercase NFKC. Chinese numerals are mapped to digits first, so "一万九千五百" and "19500" count as equal. Speed is real-time factor (RTF, processing time divided by audio length; lower is faster). Language was fixed for both engines and each sentence was transcribed on its own (no language detection, no diarization).

| Engine | Language | Metric | Error rate | RTF | Audio |
|---|---|---|---|---|---|
| Whisper large-v3 | Chinese | CER | 10.5% | 0.097 | 462 s |
| Whisper large-v3 | Korean | CER | 6.8% | 0.090 | 523 s |
| Whisper large-v3 | English | WER | 5.8% | 0.075 | 391 s |
| Qwen3-ASR-1.7B | Chinese | CER | 8.1% | 0.170 | 462 s |
| Qwen3-ASR-1.7B | Korean | CER | 6.8% | 0.259 | 523 s |
| Qwen3-ASR-1.7B | English | WER | 4.3% | 0.209 | 391 s |

Model load: Whisper about 8 s, Qwen about 15 s (disk cache warm). GPU memory: Whisper about 3.2 GB, Qwen about 4 to 5 GB (bf16). The two models are loaded one at a time. Before numeral normalization the Chinese CER was 11.2% (Whisper) and 10.8% (Qwen): the same transcript can score differently by notation alone.

## 2. Whole-system checks

| Check | Input | Result |
|---|---|---|
| Korean (Whisper) | synthetic speech, 16 s | digits and dates correct, 18.7 s including model load |
| Korean (Qwen) | same file | all correct, 25.9 s |
| English with language detection | synthetic speech, 16 s | all correct (`$1,100`, `AB1234`), 37.6 s |
| Two voices, two languages | 47 s | 2 speakers and 4 turns separated correctly, routed to Qwen, 25.6 s |
| Translation and minutes (local Qwen3-8B) | same 47 s | summary, decisions and actions produced, 84 s in total |

Synthetic speech came from built-in operating-system voices, so it says little about real meetings.

## 3. Field test: a hard far-field recording

Input: a business meeting in Mandarin recorded with a phone from across the room (62 minutes in two files, poor sound quality, laughter, background noise). Reference: the human-written minutes of the same meeting, turned into a checklist of 16 key facts (`tools/eval_recording.py`; see `examples/checklist.example.json` for the format). A fact counts as found when its regular expressions appear in the transcript. This measures information recall, not character error rate.

| Version | Recall | Speech found in the 56-minute file |
|---|---|---|
| First run | 6 of 16 (38%) | about 1054 s of 3370 s (31%) |
| With VAD gap filling | 11 of 16 (69%) | about 2840 s (84%) |

What happened: silero-vad labelled about 21 minutes of normal conversation as non-speech. Loudness was fine (mean about -22 dB) and running Whisper directly on 30-second windows of that part produced Chinese conversation. A lower VAD threshold (0.35) did not help. Filling VAD gaps longer than 30 seconds with 25-second chunks fixed most of it. Side effect: on noisy chunks the recognizer once echoed the dictionary hint list as text; a filter now removes such segments.

What was still missed: numbers (rates, quantities, counts) and some topics that were mentioned briefly. Topic-level information (companies, products, test plans, who supplies what) came through well. The practical advice for this kind of recording: record closer to the speakers when numbers matter, and verify every flagged number.

Processing time for the 56-minute file with gap filling: about 20 minutes for 457 speech segments, without translation or minutes. Before gap filling (140 segments, with translation and minutes) it took about 6.5 minutes, so covering the missing speech costs real time. The 6-minute file took about 50 seconds.

## 4. Known gaps

1. Only one real recording was evaluated against human minutes.
2. Speaker separation was checked on clearly different voices; similar voices and overlap are weak.
3. Translation and minutes quality is that of an 8B local model.
4. Cantonese and heavy code-switching were not tested.
