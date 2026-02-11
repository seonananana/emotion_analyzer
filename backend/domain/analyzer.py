import re
import yaml
import os
from typing import Dict, List, Tuple, Any
from backend.infra.paths import NEGATIVE_LEXICON_PATH
from collections import defaultdict
from datetime import datetime
from backend.domain.analyzer import NegativeEmotionAnalyzer

class NegativeEmotionAnalyzer:
    def __init__(self):
        """
        부정 감성 분석기 초기화

        Args:
            NEGATIVE_LEXICON_PATH: 부정 감성 사전 파일 경로
        """
        self.negative_lexicon = self._load_negative_lexicon()
        self.word_variations = self._create_word_variations()

    def _load_negative_lexicon(self) -> Dict[str, Dict]:
        """부정 감성 사전 로드"""
        with open(NEGATIVE_LEXICON_PATH, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        lexicon = {}
        for entry in data["negative_emotion_lexicon"]["entries"]:
            word = entry["word"]
            lexicon[word] = {
                "emotions": entry["emotions"],
                "total": entry["total"],
                "id": entry["id"],
            }
        return lexicon

    def _create_word_variations(self) -> Dict[str, str]:
        """
        어휘의 다양한 변형 형태를 생성
        한국어 동사/형용사의 활용 형태를 고려
        """
        variations = {}

        for word in self.negative_lexicon.keys():
            # 원형 등록
            variations[word] = word

            # 동사 활용 형태
            if word.endswith("다"):
                stem = word[:-1]  # '다' 제거

                # 기본 활용 형태들
                forms = [
                    stem + "어",  # 해요체 어간
                    stem + "았",  # 과거형
                    stem + "었",  # 과거형
                    stem + "고",  # 연결어미
                    stem + "지",  # 부정문
                    stem + "아",  # 해라체
                    stem + "으니",  # 연결어미
                    stem + "면",  # 조건문
                    stem + "는",  # 현재 관형사형
                    stem + "은",  # 과거 관형사형
                    stem + "을",  # 미래 관형사형
                    stem + "게",  # 부사형
                    stem + "도록",  # 목적 연결어미
                ]

                # 불규칙 활용 고려

                # ㅂ/ㅍ 불규칙 직접 매핑 (확실한 처리)
                irregular_map = {
                    # ㅂ 불규칙
                    "무섭": "무서워",
                    "두렵": "두려워",
                    "어렵": "어려워",
                    "가까워": "가까워",  # 이미 워 형태
                    "아까워": "아까워",  # 이미 워 형태
                    # ㅍ 불규칙
                    "아프": "아파",
                    "쉬프": "쉬파",  # 드물지만 있을 수 있음
                }

                if stem in irregular_map:
                    new_stem = irregular_map[stem]

                    # 다양한 활용형 추가
                    forms.extend(
                        [
                            new_stem,  # 무서워, 두려워, 아파
                            new_stem + "서",  # 무서워서, 두려워서, 아파서
                            new_stem + "요",  # 무서워요, 두려워요, 아파요
                            new_stem + "서요",  # 무서워서요, 두려워서요, 아파서요
                            new_stem + "했다",  # 무서워했다, 두려워했다, 아파했다
                            new_stem
                            + "했어요",  # 무서워했어요, 두려워했어요, 아파했어요
                            new_stem + "하는",  # 무서워하는, 두려워하는, 아파하는
                            new_stem + "하고",  # 무서워하고, 두려워하고, 아파하고
                            new_stem + "한",  # 무서워한, 두려워한, 아파한
                        ]
                    )

                    # 과거형과 관형사형 (개별 처리)
                    if stem == "무섭":
                        past_stem = "무서웠"  # 무서웠
                        adj_stem = "무서운"  # 무서운
                    elif stem == "두렵":
                        past_stem = "두려웠"  # 두려웠
                        adj_stem = "두려운"  # 두려운
                    elif stem == "어렵":
                        past_stem = "어려웠"  # 어려웠
                        adj_stem = "어려운"  # 어려운
                    elif stem == "아프":
                        past_stem = "아팠"  # 아팠
                        adj_stem = "아픈"  # 아픈
                    else:
                        past_stem = new_stem + "웠"  # 기본
                        adj_stem = new_stem[:-1] + "운"  # 기본

                    forms.extend(
                        [
                            past_stem + "다",  # 무서웠다, 두려웠다, 아팠다
                            past_stem + "어요",  # 무서웠어요, 두려웠어요, 아팠어요
                            past_stem
                            + "습니다",  # 무서웠습니다, 두려웠습니다, 아팠습니다
                            adj_stem,  # 무서운, 두려운, 아픈
                            adj_stem + "데",  # 무서운데, 두려운데, 아픈데
                        ]
                    )

                # 르 불규칙 (예: 모르다 -> 몰라)
                if stem.endswith("르"):
                    new_stem = stem[:-2] + "ll"
                    forms.extend([new_stem + "아", new_stem + "았"])

                # ㄷ 불규칙 (예: 걷다 -> 걸어)
                if stem.endswith("ㄷ"):
                    new_stem = stem[:-1] + "ㄹ"
                    forms.extend([new_stem + "어", new_stem + "었"])

                # 명사형 전성어미 추가
                nominalization_endings = [
                    "기",  # -기 (울다 -> 울기)
                    "음",  # -음 (울다 -> 욺)
                    "ㅁ",  # -ㅁ (울다 -> 욺)
                ]

                for ending in nominalization_endings:
                    forms.append(stem + ending)

                # 한국어 음성 변화 규칙 적용 (ㅏ + ㅓ → ㅐ)
                if stem.endswith("하"):  # "하다" 동사의 특별 처리
                    ha_stem = stem[:-1]  # "하" 제거
                    forms.extend(
                        [
                            ha_stem + "해",  # 하 + 어 → 해
                            ha_stem + "해서",  # 하 + 어서 → 해서
                            ha_stem + "했",  # 하 + 었 → 했
                            ha_stem + "했어요",  # 하 + 었어요 → 했어요
                            ha_stem + "했습니다",  # 하 + 었습니다 → 했습니다
                            ha_stem + "한",  # 하 + ㄴ → 한 (관형사형)
                            ha_stem + "할",  # 하 + ㄹ → 할 (관형사형)
                        ]
                    )

                # 어간 + 일반적인 어미들
                common_endings = [
                    "어요",
                    "었어요",
                    "아요",
                    "았어요",
                    "습니다",
                    "었습니다",
                    "어서",
                    "었어서",
                    "아서",
                    "았어서",
                    "더니",
                    "던",
                    "겠",
                    "려고",
                    "려면",
                    "면서",
                    "지만",
                    "지만서도",
                    "거나",
                ]

                for ending in common_endings:
                    forms.append(stem + ending)

                # 모든 변형 등록
                for form in forms:
                    if form and form not in variations:
                        variations[form] = word

        return variations

    def _find_word_positions(
        self, text: str, word_or_pattern: str
    ) -> List[Tuple[int, int, str]]:
        """
        텍스트에서 단어의 모든 위치를 찾음

        Returns:
            List of (start_pos, end_pos, matched_text)
        """
        positions = []

        # 한국어의 경우 단어 경계(\b, 단어 비단어 간의 경계)가 제대로 작동하지 않으므로 단순 문자열 검색 사용
        start = 0
        while True:
            pos = text.find(word_or_pattern, start) #단순 문자열 검색
            if pos == -1: #못 찾으면 종료
                break
            positions.append((pos, pos + len(word_or_pattern), word_or_pattern))
            start = pos + 1

        return positions

    def analyze_text(self, text: str, student_name: str = "") -> Dict[str, Any]:
        """
        텍스트에서 부정 감성 어휘 분석

        Args:
            text: 분석할 텍스트
            student_name: 학생 이름

        Returns:
            분석 결과 딕셔너리
        """
        word_frequency = defaultdict(list)  # {original_word(원형, 키): [(form, start, end), ...](값, list구성)}
        emotion_scores = defaultdict(int) # {emotion: score}(감정 별 누적점수 합)
        total_negative_words = 0
        total_frequency = 0

        # 중복 위치 체크를 위한 집합, 중복X
        used_positions = set()

        # 단어 길이순으로 정렬하여 긴 단어를 먼저 매칭 (더 정확한 매칭을 위해)
        sorted_variations = sorted(
            self.word_variations.items(), key=lambda x: len(x[0]), reverse=True
        )

        # 모든 변형 단어에 대해 검색 (길이순)
        for variation, original_word in sorted_variations:
            positions = self._find_word_positions(text, variation)

            for start, end, matched_text in positions:
                # 위치 중복 체크
                position_range = set(range(start, end))
                if not position_range.intersection(used_positions):
                    # 새로운 위치인 경우만 추가
                    word_frequency[original_word].append(
                        {
                            "form": matched_text,
                            "start": start,
                            "end": end,
                            "context": text[
                                max(0, start - 10) : end + 10
                            ],  # 앞뒤 10글자 맥락
                        }
                    )

                    # 감정 점수 추가
                    emotions = self.negative_lexicon[original_word]["emotions"] # 감정 점수 dict
                    for emotion, score in emotions.items():
                        emotion_scores[emotion] += score # 감정 점수 누적(원형 단어 기준)

                    total_frequency += 1 # 매치 수 증가 ex)총 N회 사용
                    used_positions.update(position_range)

        # 발견된 단어 수
        total_negative_words = len(
            [w for w in word_frequency.keys() if word_frequency[w]]
        )

        # 감정 비율 계산
        total_emotion_score = sum(emotion_scores.values()) # 전체 감정 점수 합
        emotion_percentages = {}
        if total_emotion_score > 0:
            for emotion, score in emotion_scores.items():
                emotion_percentages[emotion] = round(
                    (score / total_emotion_score) * 100, 1 # 소수점 1자리까지 백분율로 계산
                )

        # 단어별 빈도 정리 (이전 결과와 호환)
        # 먼저 빈도순, 같은 빈도일 때는 가나다순으로 정렬
        sorted_words = sorted(
            [
                (word, occurrences)
                for word, occurrences in word_frequency.items()
                if occurrences
            ],
            key=lambda x: (-len(x[1]), x[0]),  # 빈도 내림차순, 단어명 오름차순
        )

        simple_word_frequency = {} # {original_word: frequency}
        detailed_word_analysis = {} # {original_word: {frequency, forms, emotions}}

        for original_word, occurrences in sorted_words:
            simple_word_frequency[original_word] = len(occurrences) # 빈도 수 기록
            detailed_word_analysis[original_word] = {
                "frequency": len(occurrences), # 빈도 수
                "forms": occurrences, # 형태 및 위치 정보
                "emotions": self.negative_lexicon[original_word]["emotions"], # 감정 점수(가중치 포함)
            }

        return {
            "student_name": student_name, # 이름
            "analysis_date": datetime.now().strftime("%Y-%m-%d"), # 분석 날짜
            "text_length": len(text), # 글 길이(문자 수)
            "total_negative_words_found": total_negative_words, # 발견된 부정 어휘 수(고유 단어 수)
            "total_negative_word_frequency": total_frequency, # 부정 어휘 총 사용 빈도(중복 포함)
            "word_frequency": simple_word_frequency, # 단어별 빈도 (간단)
            "detailed_word_analysis": detailed_word_analysis, # 단어별 상세 분석
            "emotion_distribution": dict(emotion_scores), # 감정별 누적 점수
            "total_weighted_emotion_score": total_emotion_score, # 전체 감정 점수 합
            "emotion_percentages": emotion_percentages, # 감정별 비율(백분율, 소수1자리)
        }

    def generate_text_analysis_comment(
        self, analysis_result: Dict[str, Any], text: str
    ) -> str:
        """
        텍스트 특성에 대한 코멘트 생성 (전반적 특성과 감정 분포 특성만)
        """
        student_name = analysis_result["student_name"]
        total_words = analysis_result["total_negative_words_found"]
        total_freq = analysis_result["total_negative_word_frequency"]
        text_length = analysis_result["text_length"]
        emotions = analysis_result["emotion_percentages"]

        # 부정 어휘 밀도 계산
        density = (total_freq / text_length) * 1000 if text_length > 0 else 0

        # 코멘트 작성
        comment = f"""
## {student_name} 학생의 글 분석 코멘트

### 전반적 특성
- **글 길이**: {text_length:,}자
- **부정 어휘 수**: {total_words}개 (중복 포함 {total_freq}회)
- **부정 어휘 밀도**: 1000자당 {density:.1f}개

### 감정 분포 특성
"""

        if emotions:
            # 주요 감정 분석
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True) # 비율 내림차순 정렬
            primary_emotion = sorted_emotions[0][0] if sorted_emotions else None # 주요 감정

            comment += f"- **주요 감정**: {primary_emotion} ({emotions.get(primary_emotion, 0):.1f}%)\n"
 
            for emotion, percentage in sorted_emotions: 
                if percentage > 10:  # 10% 이상인 감정만 언급
                    comment += f"- **{emotion}**: {percentage}% - "

                    if emotion == "슬픔":
                        comment += "깊은 상실감과 애도의 감정이 드러남\n"
                    elif emotion == "우울":
                        comment += "지속적인 우울감과 무력감이 나타남\n"
                    elif emotion == "불안":
                        comment += "미래에 대한 걱정과 불확실성이 표현됨\n"
                    elif emotion == "외로움":
                        comment += "고립감과 소외감이 느껴짐\n"
                    elif emotion == "좌절":
                        comment += "목표 달성의 어려움과 실망감이 드러남\n"
                    elif emotion == "분노":
                        comment += "강한 분노와 적대감이 표현됨\n"

        return comment

    def generate_negative_word_analysis_summary(
        self, analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM 프롬프트에 사용할 부정 어휘 분석 요약 생성

        Args:
            analysis_result: analyze_text의 결과

        Returns:
            LLM 프롬프트용 요약 딕셔너리
        """
        # 고빈도 단어 추출 (빈도 2 이상 또는 상위 3개)
        word_frequency = analysis_result["word_frequency"] 
        high_frequency_words = [] 

        # 빈도순으로 정렬된 단어들에서 상위 항목 추출
        sorted_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True) 

        for word, freq in sorted_words:
            if freq >= 2 or len(high_frequency_words) < 3:  # 빈도 2 이상이거나 상위 3개
                high_frequency_words.append(f"{word}({freq}회)") # (단어, 빈도) list
            if len(high_frequency_words) >= 5:  # 최대 5개까지
                break

        return {
            "total_negative_words_found": analysis_result["total_negative_words_found"], # 발견된 부정 어휘 수
            "total_negative_word_frequency": analysis_result[
                "total_negative_word_frequency" # 부정 어휘 총 사용 빈도
            ],
            "emotion_percentages": analysis_result["emotion_percentages"], # 감정별 비율
            "high_frequency_words": high_frequency_words, # 고빈도 단어 리스트
            "negative_word_density": round( 
                (
                    analysis_result["total_negative_word_frequency"] # 부정 어휘 밀도
                    / analysis_result["text_length"] # 글 길이
                )
                * 1000, 
                1, 
            ),
        }