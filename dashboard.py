"""
PolyMarket Trader Dashboard
============================
FastAPI web dashboard for monitoring bot performance.
Served by Railway as the `web` process.

Access at your Railway URL (e.g. https://polymarkettraderbot.up.railway.app)
"""

import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

from src import database as db

app = FastAPI(title="PolyMarket Trader Dashboard", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "changeme")


def verify_password(credentials: HTTPBasicCredentials = Depends(security)):
    correct = secrets.compare_digest(credentials.password.encode(), DASHBOARD_PASSWORD.encode())
    if not correct:
        raise HTTPException(status_code=401, detail="Incorrect password",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials


@app.on_event("startup")
async def startup():
    await db.init_db()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, _=Depends(verify_password)):
    stats = await db.get_dashboard_stats()
    return templates.TemplateResponse("index.html", {"request": request, **stats})


@app.get("/api/stats")
async def api_stats(_=Depends(verify_password)):
    return await db.get_dashboard_stats()
