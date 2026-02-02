# backend/infra/llm_client.py

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from together import Together

from backend.domain.aggregation import (
    build_negative_ratio,
    extract_top_keywords,
    build_scores,
)
from backend.exceptions import LLMError
from backend.infra.prompts import load_system_prompt, load_user_prompt
from backend.infra.paths import KEY_PATH

logger = logging.getLogger(__name__)

# 전역 클라이언트 캐시
_TOGETHER_CLIENT: Optional[Together] = None


def _load_together_api_key() -> str:
    """
    Together API 키를 로딩한다.
    1순위: 환경변수 TOGETHER_API_KEY
    2순위: conf/key-togetherai.txt
    둘 다 없으면 LLMError 발생.
    """
    key = os.getenv("TOGETHER_API_KEY")
    if key:
        return key.strip()

    if KEY_PATH.exists():
        content = KEY_PATH.read_text(encoding="utf-8").strip()
        if content:
            return content

    raise LLMError(
        "Together API 키를 찾을 수 없습니다. "
        "환경변수 TOGETHER_API_KEY 또는 conf/key-togetherai.txt를 설정해 주세요."
    )


def _get_together_client() -> Together:
    """
    Together Python 클라이언트를 전역으로 한 번만 생성해서 재사용.
    """
    global _TOGETHER_CLIENT
    if _TOGETHER_CLIENT is None:
        api_key = _load_together_api_key()
        _TOGETHER_CLIENT = Together(api_key=api_key)
        logger.info("Together 클라이언트가 초기화되었습니다.")
    return _TOGETHER_CLIENT


def _build_analysis_context(
    student: Dict[str, Any],
    base_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    규칙 기반 분석 결과를 LLM에 넘길 수 있는 JSON 컨텍스트로 정리.
    (LLM 프롬프트 안에서 {{analysis_context}} 자리로 들어가는 내용)
    """
    # 이미 usecase에서도 쓰지만, LLM 컨텍스트에 넣기 위해 여기서 한 번 더 계산
    negative_ratio = build_negative_ratio(base_result)
    top_keywords = extract_top_keywords(base_result)
    scores = build_scores(base_result)

    context: Dict[str, Any] = {
        "student": {
            "name": student.get("name"),
            "age": student.get("age"),
            "gender": student.get("gender"),
            "precious_thing": student.get("precious_thing"),
        },
        "negative_ratio": negative_ratio,
        "top_keywords": top_keywords,
        "scores": scores,
        "base_result": {
            "text_length": base_result.get("text_length"),
            "total_negative_words_found": base_result.get("total_negative_words_found"),
            "total_negative_word_frequency": base_result.get("total_negative_word_frequency"),
            "emotion_distribution": base_result.get("emotion_distribution"),
            "emotion_percentages": base_result.get("emotion_percentages"),
            "word_frequency": base_result.get("word_frequency"),
        },
    }

    # 필요하면 detailed_word_analysis 같은 추가 필드도 여기서 더 붙일 수 있음
    detailed = base_result.get("detailed_word_analysis")
    if detailed:
        context["detailed_word_analysis"] = detailed

    return context


def _build_prompt(
    student: Dict[str, Any],
    base_result: Dict[str, Any],
) -> str:
    """
    유저 프롬프트 템플릿(NA-user-prompt-for-LM.txt)에
    {{analysis_context}}를 실제 JSON 문자열로 치환해서 최종 user 메시지 문자열을 만든다.
    """
    user_template = load_user_prompt()
    analysis_context = _build_analysis_context(student, base_result)

    # 보기 좋게 JSON 예쁘게 출력 (ensure_ascii=False로 한글 깨짐 방지)
    context_json = json.dumps(analysis_context, ensure_ascii=False, indent=2)

    placeholder = "{{analysis_context}}"

    if placeholder in user_template:
        # 템플릿 안에 placeholder가 있으면 그 자리에 끼워넣기
        return user_template.replace(placeholder, context_json)

    # 혹시 템플릿에 placeholder가 빠져있어도 동작하게 폴백 처리
    return f"{user_template.rstrip()}\n\n[분석 컨텍스트]\n{context_json}"

def summarize_diary(
    student: Dict[str, Any],
    base_result: Dict[str, Any],
) -> str:
    """
    규칙 기반 분석 결과를 바탕으로 LLM 요약을 생성한다.

    - system 프롬프트: NA-system-prompt-for-LM.txt
    - user 프롬프트:   NA-user-prompt-for-LM.txt (+ {{analysis_context}} 치환)

    예외 발생 시 LLMError를 던지며,
    상위 usecase(run_single_diary_usecase)에서 폴백 요약으로 전환할 수 있도록 한다.
    """
    system_prompt = load_system_prompt()
    user_content = _build_prompt(student, base_result)

    client = _get_together_client()

    model = os.getenv(
        "TOGETHER_CHAT_MODEL",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    )
    temperature_str = os.getenv("TOGETHER_TEMPERATURE", "0.2")
    max_tokens_str = os.getenv("TOGETHER_MAX_TOKENS", "1600")

    try:
        temperature = float(temperature_str)
    except ValueError:
        temperature = 0.2

    try:
        max_tokens = int(max_tokens_str)
    except ValueError:
        max_tokens = 1600

    # ====== LLM 호출 ======
    try:
        logger.info(
            "Together LLM 호출 시작: model=%s, temperature=%s, max_tokens=%s",
            model,
            temperature,
            max_tokens,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    except Exception as e:
        # 네트워크, 인증 실패 등 모든 예외를 LLMError로 래핑
        logger.exception("LLM 호출 중 예외 발생")
        raise LLMError(f"LLM 호출 중 오류가 발생했습니다: {e}") from e

    # ====== 응답 파싱 (⚠️ JSON 파싱은 여기서 하지 않음!) ======
    try:
        # Together Python SDK 공식 문서 기준: response.choices[0].message.content
        message = response.choices[0].message
        content = getattr(message, "content", None)

        if not content:
            raise ValueError("LLM 응답 content가 비어 있습니다.")

        # 일부 버전에서 content가 list일 수 있으므로 방어적으로 처리
        if isinstance(content, list):
            # 텍스트 조각들만 합치기
            merged: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    # {"type": "text", "text": "..."} 형태일 수 있음
                    text = part.get("text")
                    if text:
                        merged.append(text)
                elif isinstance(part, str):
                    merged.append(part)
            content_text = "".join(merged)
        else:
            content_text = str(content)

        result_text = content_text.strip()

        if not result_text:
            raise ValueError("LLM 응답이 비어 있습니다.")

        # ✅ 여기서는 LLM이 준 문자열 그대로 반환한다.
        #    (```json ... ``` 을 포함하더라도 그대로 보냄.
        #     프론트 main.js에서 알아서 코드블럭 제거 + JSON.parse 수행)
        return result_text

    except Exception as e:
        logger.exception("LLM 응답 파싱 중 예외 발생")
        raise LLMError(f"LLM 응답 파싱에 실패했습니다: {e}") from e