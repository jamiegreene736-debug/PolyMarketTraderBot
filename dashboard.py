"""
PolyMarket Trader Dashboard
============================
FastAPI web dashboard for monitoring bot performance.
Served by Railway as the `web` process.

Access at your Railway URL (e.g. https://polymarkettraderbot.up.railway.app)
"""

import os
import json
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader
import secrets

from src import database as db

app = FastAPI(title="PolyMarket Trader Dashboard", docs_url=None, redoc_url=None)

jinja_env = Environment(loader=FileSystemLoader("templates"))
jinja_env.filters["tojson"] = json.dumps

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
async def dashboard(_=Depends(verify_password)):
    stats = await db.get_dashboard_stats()
    template = jinja_env.get_template("index.html")
    html = template.render(**stats)
    return HTMLResponse(content=html)


@app.get("/api/stats")
async def api_stats(_=Depends(verify_password)):
    return await db.get_dashboard_stats()
