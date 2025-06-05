import os
import openai
from ..setting.config import OPENAI_API_KEY

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
    prompt_lines = [f"아래는 '{focus}' 관점에서 상위 4개 개선 활동 정보입니다:", ""]
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
        "위 정보를 바탕으로, 1위 활동이 수치적으로 2~4위 활동에 비해 어떠한 점에서 더 우수한지 "
        "한문장으로 길지 않게 한국어로 설명해 주세요."
    )

    prompt = "\n".join(prompt_lines)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "사용자에게 친절하고 간결하게 설명합니다.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        comment = response.choices[0].message.content.strip()
    except Exception as e:
        # API 호출 실패 시 기본 문구 반환
        comment = "비교 설명 생성 중 오류가 발생했습니다."
        print(f"[generate_comparison_comment] ERROR: {type(e).__name__}: {e}")
    return comment
