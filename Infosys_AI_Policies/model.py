from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Directory setup
app.mount("/static", StaticFiles(directory="static"), name="static")  # optional if you use CSS/images
templates = Jinja2Templates(directory="templates")

# ---------- Home Page ----------

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- Category Pages ----------
@app.get("/education", response_class=HTMLResponse)
def education_page(request: Request):
    return templates.TemplateResponse("education.html", {"request": request})


@app.get("/healthcare", response_class=HTMLResponse)
def healthcare_page(request: Request):
    return templates.TemplateResponse("healthcare.html", {"request": request})


@app.get("/agriculture", response_class=HTMLResponse)
def agriculture_page(request: Request):
    return templates.TemplateResponse("agriculture.html", {"request": request})


@app.get("/women", response_class=HTMLResponse)
def women_page(request: Request):
    return templates.TemplateResponse("women.html", {"request": request})


@app.get("/employment", response_class=HTMLResponse)
def employment_page(request: Request):
    return templates.TemplateResponse("employment.html", {"request": request})
