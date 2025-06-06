# service/gpt_client.py

import os
from ..setting.config import OPENAI_API_KEY
import time
import json
from typing import List
from openai import AsyncOpenAI

# 1) AsyncOpenAI 클라이언트 생성 (config.py에서 OPENAI_API_KEY 사용)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def generate_comparison_comment_async(focus: str, top_items: List[dict]) -> dict:
    """
    비동기로 ChatGPT API를 호출하여 상위 4개 개선 활동의 숫자 데이터를 비교한 JSON 응답을 반환합니다.
    - focus: "balanced" | "ghg" | "saving" | "roi" 등의 문자열
    - top_items: 개선 활동 딕셔너리 리스트 (최대 4개)
    반환값: {"top1": "...", "comparison": "..."} 형태의 파싱된 JSON
    """
    if not top_items:
        return {"top1": "", "comparison": ""}

    # 1) 프롬프트 구성
    prompt_lines = [
        "당신은 중소기업을 대상으로 한 탄소 배출 저감 컨설턴트이며, "
        "AI 기반 맞춤형 솔루션 추천 시스템을 운영하고 있습니다."
        f"아래는 '{focus}' 관점에서 상위 4개 개선 활동 정보입니다:",
        "",
    ]

    # 1위 부터 4위 정보 추가
    for idx, item in enumerate(top_items, start=1):
        prompt_lines += [
            f"  - rank: {idx}:",
            f"  - 업종: {item.get('industry', 'N/A')}",
            f"  - 대상설비: {item.get('facility', 'N/A')}",
            f"  - 개선구분: {item.get('improvementType', 'N/A')}",
            f"  - 개선활동명: {item.get('activity', 'N/A')}",
            f"  - 투자비: {item.get('investmentCost', 'N/A')}",
            f"  - 절감액: {item.get('costSaving', 'N/A')}",
            f"  - 투자비회수기간: {item.get('roiPeriod', 'N/A')}",
            f"  - 온실가스감축량: {item.get('emissionReduction', 'N/A')}",
            "",
        ]

    prompt_lines.append(
        """
        ※ 추천 솔루션 배열에서 `rank` 값이 1~4위 순서를 나타냅니다.

        요청 항목:

        1. "Top1 솔루션 설명": 기업 담당자가 이해하기 쉽게, 긍정적인 어조로 해설해주세요. 
            마크다운 문법을 활용하여 **굵은 글씨 강조**와 **문단 구분**이 잘 되도록 작성해주세요.

            다음 키워드를 반드시 포함하여, 다음 세가지 관점에서 응답을 만들어주세요.
            각 요소는 **굵은 텍스트(`**`)**로 표시해주세요:
            다른 설명을 앞에 붙이지 말고 바로 시작해주세요.

            - **설비 특성과 중소기업의 현실적 상황**
            - 설명

            - **투자 부담, 실행 가능성, 기대 효과**
            - 설명
            
            - **조건 불충족 시에도 전략적 가치**
            - 설명

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

    full_prompt = "\n".join(prompt_lines)

    try:
        start_ts = time.time()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # 원하시는 모델로 교체
            messages=[
                {
                    "role": "system",
                    "content": "친절하고 간결하게 설명해주세요. 응답 형식을 지켜 순수한 json으로만 응답해주세요",
                },
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=1000,
            temperature=0.1,
        )
        elapsed = time.time() - start_ts
        print(f"[LLM] focus={focus} 응답 시간: {elapsed:.2f}초")

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        # JSON에 "top1", "comparison" 키가 있는지 확인
        if "top1" in parsed and "comparison" in parsed:
            return {"top1": parsed["top1"], "comparison": parsed["comparison"]}
        else:
            # 키 누락 시 빈 문자열로 반환
            return {
                "top1": parsed.get("top1", ""),
                "comparison": parsed.get("comparison", ""),
            }
    except Exception as e:
        print(f"[LLM ERROR] focus={focus} 예외: {e}")
        return {"top1": "LLM 호출 중 오류가 발생했습니다.", "comparison": ""}
