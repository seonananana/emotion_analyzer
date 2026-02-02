# backend/app/routes_pages.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from backend.core.config import TEMPLATES_DIR

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@router.get("/", response_class=HTMLResponse)
async def show_index(request: Request):
    
    return templates.TemplateResponse("input.html", {"request": request})

@router.get("/result", response_class=HTMLResponse)
async def show_result_page_empty(request: Request):
    """
    분석 결과 페이지.

    실제 데이터(학생 정보, 인지사항 등)는
    브라우저에서 main.js가 sessionStorage에 저장된
    분석 결과를 읽어와 DOM을 채우는 방식으로 렌더링한다.
    여기서는 템플릿 틀만 내려준다.
    """
    return templates.TemplateResponse("result.html", {"request": request})
