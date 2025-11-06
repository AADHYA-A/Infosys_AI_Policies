# policy.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from config import templates

router = APIRouter()

@router.get("/policy", response_class=HTMLResponse)
def show_policy_page(request: Request):
    return templates.TemplateResponse("policy.html", {"request": request})
