import os
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "너는 전문 면접관이다. 아래는 면접자의 행동 분석 결과이다. 음성, 자세, 제스처 등의 데이터를 참고하여 면접자의 발성과 태도를 평가하고 피드백을 제공하라."),
    ("human", """
면접 답변:
- Transcript: {transcript}

[음성 분석 요약]
- 안정성 점수: {stability_score:.2f}/10
- 전반적 평가 등급: {label}

[자세 및 제스처]
- {posture}

요구사항:
1. 위 데이터를 참고해 면접자의 **발성 안정성, 자신감, 전달력, 태도**를 종합적으로 평가하라.  
2. Jitter, Shimmer, HNR 같은 기술적 용어는 언급하지 말고,  
   **청각적으로 느껴지는 인상**으로 표현하라.  
3. 피드백은 **3문장 이내** (강점→개선→긍정 마무리).  
4. 말투는 **전문적이고 따뜻하게**, 실제 면접관처럼 작성.

출력 형식:
- 분석 요약: ...
- 강점: ...
- 피드백: ...
""")
])

def build_feedback_chain(*, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.6):
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)
    return LLMChain(llm=llm, prompt=PROMPT)
