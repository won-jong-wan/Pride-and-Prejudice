# 오만과 편견 (Pride and Prejudice)

> **제목 의미**: 면접 상황을 AI 분석 결과를 통해 표정이나 자세를 정량적으로 표현하여, 면접이 얼마나 딱딱하고 형식적인지를 보여줍니다. 동시에, 형식적인 것도 중요하다는 메시지를 담고 있습니다.

---

## 📌 프로젝트 개요

이 프로젝트는 **라즈베리파이5 + Hailo Edge Device**를 활용하여 AI 모델을 구동하고,  
면접자의 **표정 인식**과 **자세 인식**을 통해 면접 상황을 정량적으로 평가하는 **AI 면접 평가 시스템**입니다.  

- 면접자의 답변(목소리의 떨림, 불안정함, 크기, 자세 등)을 분석  
- 질문/답변 데이터를 LLM(ChatGPT 등)에 전달  
- LLM이 종합적으로 평가하여 점수와 기준을 산출  
- 최종적으로 PDF 등 문서 형태로 결과를 제공  

**배경**: 혼자서도 손쉽게 AI 면접을 체험하고, 객관적인 평가를 받아보고자 하는 아이디어에서 출발했습니다.

---

## 🔄 시스템 흐름도

아래 다이어그램은 면접 평가 시스템의 전체 흐름을 보여줍니다.

<img width="1024" height="1536" alt="Copilot_20251016_145148" src="https://github.com/user-attachments/assets/79427c4a-0b88-42f4-954a-c1873c27c874" />

**흐름**  
1. **카메라 입력** → 면접자의 영상/음성 수집  
2. **온 디바이스 AI (표정 인식 + 자세 인식)** → 실시간 분석  
3. **LLM 평가** → 답변 내용 + 비언어적 신호 종합 평가  
4. **PDF 리포트** → 점수와 평가 기준 문서화  

---

## 🎯 주요 기능 및 목표

- **표정 인식 + 자세 인식 모델**을 동시에 활용  
- **SadTalker 기능**을 이용해 면접자의 얼굴과 움직임을 합성 → 실제 면접처럼 생동감 있는 시뮬레이션  
- 질문 단위로 **“면접 시작 / 면접 끝” 버튼**을 눌러 진행  
- 모든 질문이 끝나면 면접관이 “면접이 끝났습니다.”라고 알리고,  
  LLM이 종합적으로 평가하여 **점수와 평가 기준**을 출력  
- **목표**: 이러한 과정을 통해 **적절하고 신뢰할 수 있는 평가 점수**를 산출하는 것  

---

### 요구 사항
- Raspberry Pi 5
- Hailo Edge Device
- Python 3.10+
- OpenCV, PyTorch, SadTalker, Hailo SDK 등

---

### 설치
```bash
git clone https://github.com/won-jong-wan/Pride-and-Prejudice.git
cd Pride-and-Prejudice
pip install -r requirements.txt
```

---

### 실행 (예정)
```bash
python run_interview.py --camera /dev/video0
```

---

### 📂 프로젝트 구조
Pride-and-Prejudice/
├─ data/          # 텍스트 및 데이터
├─ scripts/       # 분석 및 처리 스크립트
├─ notebooks/     # Jupyter 노트북
├─ reports/       # 결과 리포트
├─ requirements.txt
└─ README.md

---

### 🛠️ 사용 예시
아직 구체적인 사용 예시는 준비되지 않았습니다. 향후 실제 면접 시뮬레이션 영상과 평가 리포트 샘플을 추가할 예정입니다.

---

### 🗺️ 로드맵

    [x] 표정 인식 모델 연동

    [x] 자세 인식 모델 연동

    [ ] SadTalker 기반 시뮬레이션

    [ ] LLM 평가 시스템 연결

    [ ] PDF 리포트 자동 생성

---

### 🤝 기여하기
Issue와 Pull Request 환영합니다.
기능 제안, 버그 리포트, 코드 개선 모두 환영합니다.
브랜치 전략: 기능 단위로 브랜치를 생성하고, 작은 단위의 PR을 권장합니다.

---

### 📜 라이선스

MIT License (추후 변경 가능)

---

### 🙏 감사의 말

Jane Austen의 원작 Pride and Prejudice (퍼블릭 도메인)에서 제목 영감을 얻었습니다.
Hailo, SadTalker, OpenAI LLM 등 오픈소스 및 AI 기술에 감사드립니다.
