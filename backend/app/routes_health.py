# backend/app/routes_health.py

from __future__ import annotations
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health():
    """
    단순 헬스 체크 엔드포인트.

    - 서버가 살아있는지만 확인할 때 사용
    - 추후 DB 연결, 사전 로딩 상태 등을 같이 점검하도록 확장 가능
    """
    return {
        "status": "ok",
        "service": "emotion-diary-analyzer",
    }
