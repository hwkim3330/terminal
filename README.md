# LFM AGI System

**Liquid Foundation Model 기반 AGI 시스템**

LFM 2.5 모델을 사용한 자율 시스템 운영 및 컴퓨터 제어 에이전트.

## 구성 요소

### 1. TerminaI - 시스템 운영 에이전트
- **LFM 2.5 Instruct (1.2B)** 기반
- OODA Loop 추론
- Native PTY 터미널 제어
- MCP/A2A 프로토콜 지원
- Fleet Commander 멀티 에이전트

### 2. Computer Use VLA - 컴퓨터 제어 에이전트
- **LFM 2.5 VL (1.6B)** 기반
- 비전-언어-액션 모델
- UI 요소 인식 및 제어
- OCR 및 스크린 분석

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                       LFM AGI SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │      TerminaI        │    │     Computer Use VLA         │  │
│  │  LFM 2.5 Instruct    │    │      LFM 2.5 VL             │  │
│  │      (1.2B)          │    │        (1.6B)               │  │
│  └──────────┬───────────┘    └─────────────┬────────────────┘  │
│             │                              │                    │
│  ┌──────────▼───────────┐    ┌─────────────▼────────────────┐  │
│  │    OODA Loop         │    │   Vision-Language-Action     │  │
│  │ Observe-Orient-      │    │  Screen → VL Model → Action  │  │
│  │ Decide-Act-Verify    │    │                              │  │
│  └──────────┬───────────┘    └─────────────┬────────────────┘  │
│             │                              │                    │
│  ┌──────────▼───────────┐    ┌─────────────▼────────────────┐  │
│  │     Native PTY       │    │    Computer Controller       │  │
│  │  sudo/ssh/vim/etc    │    │  Mouse/Keyboard/Screen      │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                Support Systems                            │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │  │
│  │  │   MCP   │  │   A2A   │  │  Fleet  │  │   Ollama    │  │  │
│  │  │ Tools   │  │Protocol │  │Commander│  │  (Local)    │  │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 설치

```bash
# 저장소 클론
git clone https://github.com/hwkim3330/terminal.git
cd terminal

# 의존성 설치
pip install -r requirements.txt

# LFM 2.5 VL을 위한 transformers (선택)
pip install git+https://github.com/huggingface/transformers.git
```

## 사용법

### 통합 시스템

```bash
# 자동 모드 (터미널/GUI 자동 선택)
python lfm_agi_system.py "ls -la 실행하고 결과 분석"

# 터미널 모드
python lfm_agi_system.py -m terminal "시스템 리소스 확인"

# GUI 모드
python lfm_agi_system.py -m gui "Chrome 열고 Google 검색"

# 인터랙티브 모드
python lfm_agi_system.py -i
```

### TerminaI 단독

```bash
cd terminai
python cli.py "로그 파일 분석"
python cli.py --interactive
```

### Python API

```python
import asyncio
from lfm_agi_system import LFMAGISystem

async def main():
    system = LFMAGISystem()

    # 터미널 작업
    result = await system.run_terminal("디스크 사용량 확인")

    # GUI 작업
    result = await system.run_gui("브라우저에서 검색")

    # 자동 선택
    result = await system.run_auto("파일 정리하고 결과 보여줘")

asyncio.run(main())
```

## 모델 정보

| 모델 | 파라미터 | 용도 | 컨텍스트 |
|-----|---------|-----|---------|
| LFM 2.5 Instruct | 1.2B | 텍스트 생성, 에이전트 | 32K |
| LFM 2.5 VL | 1.6B | 비전-언어 이해 | 32K |

### 특징

- **28T 토큰** 학습
- **다국어 지원**: 영어, 한국어, 일본어, 중국어 등
- **에이전트 최적화**: 태스크 수행, 데이터 추출, RAG
- **경량 모델**: 엣지 디바이스에서도 실행 가능

## 프로젝트 구조

```
lfm_agi/
├── lfm_agi_system.py          # 통합 시스템
├── requirements.txt           # 의존성
├── README.md                  # 문서
├── terminai/                  # TerminaI 에이전트
│   ├── cli.py                 # CLI
│   ├── pyproject.toml
│   └── src/
│       ├── core/              # OODA Loop 에이전트
│       │   ├── agent.py
│       │   └── lfm_agent.py   # LFM 기반 에이전트
│       ├── pty/               # Native PTY
│       │   └── native_pty.py
│       ├── llm/               # LLM 프로바이더
│       │   ├── providers.py   # Gemini/Ollama/OpenAI
│       │   └── lfm_provider.py # LFM 2.5
│       ├── mcp/               # MCP 프로토콜
│       ├── a2a/               # A2A 프로토콜
│       └── fleet/             # Fleet Commander
└── computer_use_vla/          # Computer Use 에이전트
    ├── pyproject.toml
    └── src/
        ├── action_space/      # 액션 정의
        ├── models/            # VLA 모델
        │   └── lfm_vl_model.py # LFM 2.5 VL
        ├── controller/        # 컴퓨터 제어
        └── diffusion/         # Flow Matching
```

## 참고

- [Liquid AI](https://www.liquid.ai/)
- [LFM 2.5](https://huggingface.co/LiquidAI)
- [NVIDIA Alpamayo](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)

## 라이선스

Apache-2.0
