import asyncio
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone

import anthropic
from loguru import logger

from src import database as db


DEFAULT_CONFIG = {
    "enabled": False,
    "provider": "auto",
    "model": "claude-haiku-4-5-20251001",
    "openai_model": "gpt-4o-mini",
    "run_interval_seconds": 300,
    "max_logs": 60,
    "max_reports": 2,
    "issues_only": True,
    "write_to_logs": True,
    "repeated_error_threshold": 3,
    "recent_loss_streak_threshold": 3,
    "recent_pnl_window": 5,
    "critical_recent_loss_usdc": 0.25,
}


class AIObserver:
    """
    Advisory sidecar that watches logs and realized P&L.
    It never places orders or edits config; it only emits findings.
    """

    def __init__(self, config: dict | None = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.enabled = bool(cfg.get("enabled", False))
        self.provider = str(cfg.get("provider", "auto")).strip().lower()
        self.model = str(cfg.get("model", DEFAULT_CONFIG["model"])).strip()
        self.openai_model = str(cfg.get("openai_model", DEFAULT_CONFIG["openai_model"])).strip()
        self.run_interval_seconds = max(30, int(cfg.get("run_interval_seconds", 300)))
        self.max_logs = max(10, int(cfg.get("max_logs", 60)))
        self.max_reports = max(1, int(cfg.get("max_reports", 2)))
        self.issues_only = bool(cfg.get("issues_only", True))
        self.write_to_logs = bool(cfg.get("write_to_logs", True))
        self.repeated_error_threshold = max(2, int(cfg.get("repeated_error_threshold", 3)))
        self.recent_loss_streak_threshold = max(2, int(cfg.get("recent_loss_streak_threshold", 3)))
        self.recent_pnl_window = max(3, int(cfg.get("recent_pnl_window", 5)))
        self.critical_recent_loss_usdc = max(0.01, float(cfg.get("critical_recent_loss_usdc", 0.25)))

        self._anthropic_client = None
        self._openai_client = None
        self._last_run = 0.0
        self._analysis_task: asyncio.Task | None = None

    def maybe_schedule(
        self,
        *,
        balance: float,
        open_orders: int,
        bot_status: str,
        last_error: str | None,
        closed_positions: list[dict] | None,
        session_start_ts: float | None = None,
    ):
        if not self.enabled:
            return

        if self._analysis_task and not self._analysis_task.done():
            return

        now = asyncio.get_event_loop().time()
        if (now - self._last_run) < self.run_interval_seconds:
            return

        self._last_run = now
        session_closed = [
            pos
            for pos in list(closed_positions or [])
            if session_start_ts is None or self._closed_position_ts(pos) >= session_start_ts
        ]
        snapshot = {
            "balance": round(float(balance or 0.0), 2),
            "open_orders": int(open_orders or 0),
            "bot_status": str(bot_status or "unknown"),
            "last_error": str(last_error or "").strip(),
            "closed_positions": session_closed,
            "session_start_ts": session_start_ts,
        }
        self._analysis_task = asyncio.create_task(self._run_snapshot(snapshot))

    async def _run_snapshot(self, snapshot: dict):
        try:
            since_timestamp = None
            if snapshot.get("session_start_ts") is not None:
                since_timestamp = datetime.utcfromtimestamp(
                    float(snapshot["session_start_ts"])
                ).isoformat()
            logs = await db.get_recent_logs(
                limit=self.max_logs,
                exclude_prefix="[ai_observer]",
                since_timestamp=since_timestamp,
            )
            heuristic_reports = self._build_heuristic_reports(logs, snapshot)
            reports = heuristic_reports

            if not reports and self.issues_only:
                return

            model_reports = await self._generate_model_reports(logs, snapshot, heuristic_reports)
            if model_reports:
                reports = model_reports

            for report in reports[: self.max_reports]:
                await self._store_report(report)
        except Exception as e:
            logger.warning(f"[ai_observer] analysis failed: {e}")

    def _build_heuristic_reports(self, logs: list[dict], snapshot: dict) -> list[dict]:
        reports: list[dict] = []
        levels = Counter(str(row.get("level") or "").upper() for row in logs)
        noisy_messages = Counter(
            self._normalize_message(str(row.get("message") or ""))
            for row in logs
            if str(row.get("level") or "").upper() in {"ERROR", "WARNING"}
        )
        noisy_messages = Counter({msg: count for msg, count in noisy_messages.items() if msg})

        repeated_message, repeated_count = ("", 0)
        if noisy_messages:
            repeated_message, repeated_count = noisy_messages.most_common(1)[0]

        if snapshot.get("bot_status") == "error" or snapshot.get("last_error"):
            reports.append({
                "category": "glitch",
                "severity": "critical",
                "title": "Bot entered an error state",
                "summary": snapshot.get("last_error") or "The supervisor marked the bot as errored.",
                "recommendation": "Check the newest ERROR log lines before restarting so the same failure does not loop.",
            })
        elif levels.get("ERROR", 0) > 0 or repeated_count >= self.repeated_error_threshold:
            summary = (
                f"Saw {levels.get('ERROR', 0)} error log(s) and {levels.get('WARNING', 0)} warning log(s)"
                f" in the last {len(logs)} entries."
            )
            if repeated_count >= self.repeated_error_threshold:
                summary += f" Most repeated issue: '{repeated_message[:120]}' ({repeated_count}x)."
            reports.append({
                "category": "glitch",
                "severity": "warning" if levels.get("ERROR", 0) <= 2 else "critical",
                "title": "Repeated operational noise detected",
                "summary": summary,
                "recommendation": "Treat repeated failures as a root-cause issue, not as normal noise. Fix the highest-frequency warning/error first.",
            })

        pnl_report = self._build_pnl_report(snapshot.get("closed_positions") or [])
        if pnl_report:
            reports.append(pnl_report)

        return reports

    def _build_pnl_report(self, closed_positions: list[dict]) -> dict | None:
        if not closed_positions:
            return None

        ordered = sorted(
            closed_positions,
            key=lambda row: self._closed_position_ts(row),
            reverse=True,
        )
        pnls = [self._safe_float(row.get("realizedPnl")) for row in ordered]
        if not pnls:
            return None

        recent = pnls[: self.recent_pnl_window]
        loss_streak = 0
        for pnl in recent:
            if pnl < 0:
                loss_streak += 1
            else:
                break

        recent_total = round(sum(recent), 2)
        recent_wins = sum(1 for pnl in recent if pnl > 0)

        if loss_streak < self.recent_loss_streak_threshold and recent_total >= 0:
            return None

        is_critical = (
            loss_streak >= self.recent_pnl_window
            or recent_total <= -abs(self.critical_recent_loss_usdc)
        )

        return {
            "category": "pnl",
            "severity": "critical" if is_critical else "warning",
            "title": "Recent realized P&L is slipping",
            "summary": (
                f"Last {len(recent)} closed trades: {recent_wins} win(s), "
                f"{len(recent) - recent_wins} loss(es), net "
                f"{recent_total:+.2f} USDC. Current loss streak: {loss_streak}."
            ),
            "recommendation": (
                "Reduce fresh risk until the losing cluster is understood. Review the strategy mix and exit timing before scaling back up."
            ),
            "recommended_action": "pause" if is_critical else "review",
        }

    async def _generate_model_reports(
        self,
        logs: list[dict],
        snapshot: dict,
        heuristic_reports: list[dict],
    ) -> list[dict]:
        provider = self._resolved_provider()
        if provider == "heuristic":
            return []

        payload = {
            "bot_status": snapshot.get("bot_status"),
            "last_error": snapshot.get("last_error"),
            "balance_usdc": snapshot.get("balance"),
            "open_orders": snapshot.get("open_orders"),
            "recent_closed_positions": [
                {
                    "title": row.get("title") or row.get("slug") or "",
                    "outcome": row.get("outcome") or "",
                    "realizedPnl": self._safe_float(row.get("realizedPnl")),
                    "timestamp": row.get("timestamp"),
                }
                for row in sorted(
                    snapshot.get("closed_positions") or [],
                    key=lambda row: self._closed_position_ts(row),
                    reverse=True,
                )[:8]
            ],
            "heuristic_findings": heuristic_reports,
            "recent_logs": [
                {
                    "timestamp": row.get("timestamp"),
                    "level": row.get("level"),
                    "message": str(row.get("message") or "")[:220],
                }
                for row in logs[-20:]
            ],
        }

        prompt = (
            "You are an advisory reliability and P&L analyst for a live prediction-market bot.\n"
            "Return STRICT JSON only: an array with up to 2 findings.\n"
            "Each finding must have keys: category, severity, title, summary, recommendation, recommended_action.\n"
            "Allowed category values: glitch, pnl.\n"
            "Allowed severity values: info, warning, critical.\n"
            "Allowed recommended_action values: review, pause, restart.\n"
            "Use recommended_action=pause for critical P&L/loss-streak findings.\n"
            "Do not flag scheduled countdown or idle logs as stalls when they clearly describe an intentional interval.\n"
            "Focus on concrete anomalies visible in the logs or realized P&L.\n"
            "Do not suggest autonomous code edits, config rewrites, or placing new trades.\n"
            "If nothing looks notable, return [].\n\n"
            f"Context:\n{json.dumps(payload, ensure_ascii=True)}"
        )

        try:
            if provider == "anthropic":
                raw = await self._ask_anthropic(prompt)
            else:
                raw = await self._ask_openai(prompt)
            parsed = json.loads(self._extract_json(raw))
            if isinstance(parsed, list):
                return [self._normalize_report(item) for item in parsed if isinstance(item, dict)]
        except Exception as e:
            logger.warning(f"[ai_observer] model analysis failed, using heuristics only: {e}")
        return []

    async def _store_report(self, report: dict):
        normalized = self._normalize_report(report)
        fingerprint = hashlib.sha1(
            json.dumps(
                {
                    **normalized,
                    "time_bucket": datetime.utcnow().strftime("%Y-%m-%dT%H"),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        inserted = await db.insert_ai_observer_report(
            category=normalized["category"],
            severity=normalized["severity"],
            title=normalized["title"],
            summary=normalized["summary"],
            recommendation=normalized["recommendation"],
            recommended_action=normalized["recommended_action"],
            fingerprint=fingerprint,
        )
        if not inserted:
            return

        if not self.write_to_logs:
            return

        level = self._level_from_severity(normalized["severity"])
        message = (
            f"[ai_observer][{normalized['category']}][{normalized['severity']}] "
            f"{normalized['title']} | {normalized['summary']} | "
            f"Action={normalized['recommended_action']} | Next step: {normalized['recommendation']}"
        )
        await db.log_to_db(level, message)
        getattr(logger, level.lower())(message)

    def _resolved_provider(self) -> str:
        if self.provider == "anthropic":
            return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "heuristic"
        if self.provider == "openai":
            return "openai" if os.getenv("OPENAI_API_KEY") else "heuristic"
        if self.provider == "heuristic":
            return "heuristic"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "heuristic"

    def _get_anthropic(self) -> anthropic.AsyncAnthropic:
        if self._anthropic_client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._anthropic_client

    async def _ask_anthropic(self, prompt: str) -> str:
        client = self._get_anthropic()
        message = await client.messages.create(
            model=self.model,
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
            timeout=20.0,
        )
        return message.content[0].text.strip()

    def _get_openai(self):
        if self._openai_client is None:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._openai_client = openai.AsyncOpenAI(api_key=api_key)
        return self._openai_client

    async def _ask_openai(self, prompt: str) -> str:
        client = self._get_openai()
        response = await client.chat.completions.create(
            model=self.openai_model,
            max_tokens=350,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            timeout=20.0,
        )
        return (response.choices[0].message.content or "").strip()

    def _normalize_report(self, report: dict) -> dict:
        category = str(report.get("category") or "glitch").strip().lower()
        if category not in {"glitch", "pnl"}:
            category = "glitch"

        severity = str(report.get("severity") or "info").strip().lower()
        if severity not in {"info", "warning", "critical"}:
            severity = "info"

        recommended_action = str(report.get("recommended_action") or "").strip().lower()
        if recommended_action not in {"review", "pause", "restart"}:
            if severity == "critical":
                recommended_action = "pause"
            elif category == "glitch" and severity == "warning":
                recommended_action = "review"
            else:
                recommended_action = "review"
        elif category == "pnl" and severity == "critical":
            recommended_action = "pause"

        title = str(report.get("title") or "AI observer note").strip()[:100]
        summary = str(report.get("summary") or "").strip()[:280]
        recommendation = str(report.get("recommendation") or "").strip()[:220]

        if not summary:
            summary = "No summary provided."
        if not recommendation:
            recommendation = "Review recent logs and P&L context before changing risk."

        return {
            "category": category,
            "severity": severity,
            "title": title or "AI observer note",
            "summary": summary,
            "recommendation": recommendation,
            "recommended_action": recommended_action,
        }

    @staticmethod
    def _extract_json(raw: str) -> str:
        text = raw.strip()
        if text.startswith("[") or text.startswith("{"):
            return text
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") or part.startswith("{"):
                    return part
        start = min([idx for idx in (text.find("["), text.find("{")) if idx != -1], default=-1)
        if start == -1:
            raise ValueError("No JSON found in model response")
        return text[start:]

    @staticmethod
    def _normalize_message(message: str) -> str:
        message = message.strip()
        if "|" in message:
            message = message.split("|", 1)[0].strip()
        return message[:180]

    @staticmethod
    def _level_from_severity(severity: str) -> str:
        return {
            "critical": "ERROR",
            "warning": "WARNING",
        }.get(severity, "INFO")

    @staticmethod
    def _closed_position_ts(row: dict) -> float:
        value = row.get("timestamp") or row.get("resolved_at")
        raw = AIObserver._safe_float(value)
        if raw > 1_000_000_000_000:
            raw /= 1000.0
        if raw > 0:
            return raw

        text = str(value or "").strip()
        if not text:
            return 0.0

        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return 0.0

    @staticmethod
    def _safe_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
