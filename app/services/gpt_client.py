import os
import openai
from ..setting.config import OPENAI_API_KEY
import time
from openai import AsyncOpenAI
import asyncio

# 1) OpenAI API 키 설정 (config.py에서 가져온 상수 사용)
openai.api_key = OPENAI_API_KEY


def generate_comparison_comment(focus: str, top_items: list) -> str:
    """
    ChatGPT API를 호출하여 상위 4개 개선 활동(딕셔너리)의 숫자 데이터를 비교한 설명을 생성.
    - focus: "balanced" | "ghg" | "saving" | "roi"
    - top_items: recommend_by_focus 반환 결과(딕셔너리 리스트)에서 상위 4개 아이템
    반환값: 1위 활동이 2~4위에 비해 수치적으로 어떤 점에서 우수한지 설명하는 문장 (한두 문장)
    """
    if not top_items:
        return "비교 설명을 생성하기 위한 데이터가 부족합니다."

    top1 = top_items[0]
    others = top_items[1:4]  # 2~4위만 추출

    # 프롬프트 구성
    prompt_lines = [
        "당신은 중소기업을 대상으로 한 탄소 배출 저감 컨설턴트이며, "
        "AI 기반 맞춤형 솔루션 추천 시스템을 운영하고 있습니다."
        "아래의 입력값과 추천 솔루션을 바탕으로 다음의 두 가지 항목을 작성해주세요."
        f"아래는 '{focus}' 관점에서 상위 4개 개선 활동 정보입니다:",
        "",
    ]
    # 1위 정보
    prompt_lines += [
        "1위 활동:",
        f"  - 개선활동명: {top1.get('개선활동명_요약')}",
        f"  - 투자비: {top1.get('투자비')}",
        f"  - 절감액: {top1.get('절감액')}",
        f"  - 투자비회수기간: {top1.get('투자비회수기간')}",
        f"  - 온실가스감축량: {top1.get('온실가스감축량')}",
        "",
    ]

    # 2~4위 정보
    if others:
        prompt_lines.append("2~4위 활동:")
        for idx, item in enumerate(others, start=2):
            if item:
                prompt_lines += [
                    f"{idx}위 활동:",
                    f"  - 개선활동명: {item.get('개선활동명_요약')}",
                    f"  - 투자비: {item.get('투자비')}",
                    f"  - 절감액: {item.get('절감액')}",
                    f"  - 투자비회수기간: {item.get('투자비회수기간')}",
                    f"  - 온실가스감축량: {item.get('온실가스감축량')}",
                    "",
                ]

    prompt_lines.append(
        """

    ※ 추천 솔루션 배열에서 각 항목의 `rank` 값을 기준으로 Top1~Top4를 정의해주세요. 예를 들어 `rank: 1`은 Top 1, `rank: 2`는 Top 2로 간주합니다.

    요청 항목:

    1. "Top1 솔루션 설명": 기업 담당자가 이해하기 쉽게, 긍정적인 어조로 해설해주세요. 
        마크다운 문법을 활용하여 **굵은 글씨 강조**와 **문단 구분**이 잘 되도록 작성해주세요.

        다음 키워드를 설명에 포함하고, 각 요소는 **굵은 텍스트(`**`)**로 표시해주세요:
        - **설비 특성과 중소기업의 현실적 상황**
        - **투자 부담, 실행 가능성, 기대 효과**
        - **조건 불충족 시에도 전략적 가치**

    2. "Top1과 Top2~4와의 비교 분석": 단순한 우열 비교가 아닌 전략적 해석 방식으로 설명해주세요.
        - Rank별로 어떤 상황에서 적합한지, 어떻게 조합할 수 있는지 조언
        - 마치 컨설팅 전문가가 맞춤형 제안서를 작성하듯 구성
        - 다음과 같은 마크다운 리스트 형식으로 작성해주세요:

        - **Top 1**: 
        - **Top 2**: 
        - **Top 3**: 
        - **Top 4**: 

        - 마지막에는 한 문단으로 전체 해석 요약을 추가해주세요.

    응답은 아래 형식으로 출력해주세요 (markdown 렌더링/LaTeX 대응 가능하도록):

    {{
    "top1": "...",
    "comparison": "..."
    }}
    """
    )

    prompt = "\n".join(prompt_lines)

    try:
        start_time = time.time()
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": "사용자에게 친절하고 간결하게 설명합니다.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.3,
        )
        # API 호출 후 경과 시간 계산
        elapsed = time.time() - start_time
        print(f"{focus} LLM 응답 시간: {elapsed:.2f}초")
        comment = response.choices[0].message.content.strip()
    except Exception as e:
        # API 호출 실패 시 기본 문구 반환
        comment = "비교 설명 생성 중 오류가 발생했습니다."
        print(f"{focus} ERROR: {type(e).__name__}: {e}")
    return comment
