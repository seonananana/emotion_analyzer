# backend/exceptions.py
"""
프로젝트 전역에서 공통으로 사용하는 예외 정의 모듈.

- ConfigError      : 설정/환경(.env, 키 파일 등) 문제
- LexiconLoadError : 사전/데이터 YAML 로딩 문제
- StudentDataError : 웹/입력 데이터 검증 실패
- AnalysisError    : 규칙 기반 분석 전체 실패
- LLMError         : LLM 호출 및 응답 파싱 실패
"""

class ConfigError(RuntimeError):
    """환경 설정(.env, 키 파일 등) 문제."""
    pass


class LexiconLoadError(IOError):
    """부정 사전 / 학생 데이터 등 설정 파일 로딩 실패."""
    pass


class StudentDataError(ValueError):
    """학생 이름/나이/텍스트 등 입력 데이터 검증 실패."""
    pass


class AnalysisError(RuntimeError):
    """규칙 기반 분석 전체 실패."""
    pass


class LLMError(RuntimeError):
    """LLM 호출, 응답 포맷, 파싱 과정에서 발생하는 오류."""
    pass
