# JOONS LISTEN MCP

MCP 클라이언트용 로컬 회의 녹음 전사 도구입니다. 내 GPU에서 Whisper large-v3와 Qwen3-ASR-1.7B로 받아쓰고, 화자 표시, 숫자·날짜 확인 표시, 선택적으로 한국어 번역과 회의록까지 만듭니다.

> [JOONS 생태계](https://github.com/dbdefault/joons-ecosystem)의 일부입니다.

English: [README.md](README.md)

상태: 0.1.0 Preview. 쓰기 전에 [LIMITATIONS.md](LIMITATIONS.md)를 읽어 주세요.

## 한눈에 보기

- **무엇**: 작은 로컬 서비스와 표준 입출력(stdio) MCP 서버입니다. 회의나 통화 녹음을 주면 화자 구간·시각이 붙은 전사문, 숫자·가격·날짜·제품 코드의 "확인 필요" 표시, 그리고 로컬 LLM이 쓴 한국어 번역과 회의록을 돌려줍니다.
- **대상**: 중국어·한국어·영어(혼용 포함) 회의를 녹음하고, 음성이 내 컴퓨터 밖으로 나가지 않기를 바라며, Claude 같은 MCP 클라이언트로 쓰고 싶은 사람입니다.
- **아닌 경우**: 실시간 자막, 클라우드 처리, 법적 증거용 속기, CUDA GPU가 없는 환경(8GB NVIDIA 카드에서만 시험했고 CPU만으로는 시험하지 않았습니다).
- **방법**: 의존 패키지를 설치하고, 서비스를 켠 뒤, MCP 클라이언트가 `python -m joons_listen mcp`를 부르게 설정하고, 파일을 전사해 달라고 시키면 됩니다.
- **믿어도 되는 범위**: 서비스는 루프백에만 묶이고, 허용한 폴더의 오디오만 읽으며, 회의 글은 로컬 LLM에만 보냅니다. 뽑은 숫자는 모두 "확인 필요"로 표시하고 자동으로 확정하지 않습니다. [SECURITY.md](SECURITY.md) 참고.

## 처리 흐름

```text
오디오 파일
  -> ffmpeg 변환(16kHz 모노)
  -> silero-vad 음성 구간(VAD가 놓친 빈 구간은 메움, LIMITATIONS 참고)
  -> 화자 구간: ECAPA 임베딩 + 군집
  -> 받아쓰기: Qwen3-ASR-1.7B(중·한·혼용) 또는 Whisper large-v3(영어), 언어별로 선택
  -> 사실 후보(가격·수량·날짜·코드) 추출과 불확실 표시
  -> 선택: 로컬 Ollama 모델로 한국어 번역·회의록
  -> txt / md / srt / json / minutes 파일
```

## 설치와 실행

필요한 것: Python 3.10 이상, PATH의 ffmpeg, 약 8GB GPU 메모리의 NVIDIA GPU와 CUDA용 PyTorch, 그리고 `numpy soundfile scikit-learn torch faster-whisper qwen-asr silero-vad speechbrain huggingface_hub`. 번역·회의록에는 [Ollama](https://ollama.com)와 로컬 모델(기본 `qwen3:8b`)이 필요합니다. 모델 가중치(약 10GB)는 처음 쓸 때 Hugging Face에서 데이터 폴더로 내려받으며 이 저장소에는 없습니다.

```sh
git clone https://github.com/dbdefault/joons-listen-mcp
cd joons-listen-mcp
pip install -e .
pip install torch faster-whisper qwen-asr silero-vad speechbrain huggingface_hub

python -m joons_listen check
python -m joons_listen serve
```

PyPI에는 올리지 않고 소스에서 설치합니다. MCP 설정 예시는 `examples/mcp-client-config.json`에 있고, 도구 6개(`listen_transcribe`, `listen_diarize`, `listen_summarize`, `listen_status`, `listen_result`, `listen_models`)의 설명은 영어 README에 있습니다.

긴 녹음은 시간이 걸립니다(56분 녹음이 RTX 4060 Ti에서 약 20분). 그래서 도구는 기다리지 않고 `job_id`를 바로 돌려주며, `listen_status`로 `done`이 될 때까지 확인한 뒤 `listen_result`로 읽습니다.

## 허용 폴더

서비스는 허용한 폴더 안의 오디오만 읽습니다. 기본은 데이터 폴더의 `inbox`, `~/Desktop`, `~/Downloads`이고 `JOONS_LISTEN_ROOTS`로 바꿉니다. 파일 이름만 주면 허용 폴더 안에서 찾습니다. 비밀 파일처럼 보이는 이름은 거부합니다.

## 설정

모든 설정은 환경 변수입니다. 주요 항목은 영어 README의 표를 보세요(`JOONS_LISTEN_DATA`, `JOONS_LISTEN_PORT`, `JOONS_LISTEN_ROOTS`, `JOONS_LISTEN_URL`, `JOONS_LISTEN_LLM_URL`, `JOONS_LISTEN_LLM_MODEL`, `JOONS_LISTEN_MIRROR_DIR`, `JOONS_LISTEN_DICT`).

## 평가와 문서

[docs/EVALUATION.md](docs/EVALUATION.md)에 공개 음성 세트의 오류율·속도와, 먼 곳에서 녹음한 어려운 녹음의 현장 시험 결과가 있습니다(영어). 표본이 작아 엔진 순위가 아니라 약점을 찾기 위한 측정입니다. HTTP API는 [docs/API.md](docs/API.md), 한계는 [LIMITATIONS.md](LIMITATIONS.md), 보안은 [SECURITY.md](SECURITY.md)입니다.

## 라이선스

Apache License 2.0. 사용한 모델의 라이선스(Whisper MIT, Qwen3-ASR Apache-2.0, SpeechBrain ECAPA Apache-2.0, silero-vad MIT)는 각자 확인해 주세요. 이 저장소에는 모델 가중치가 없습니다.
