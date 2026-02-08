import time
from typing import List, Dict, Optional, Tuple

from app.services.confusion_trend import compute_confusion_trend
from app.services.db import db_conn, alerts
from app.services.question_log import get_questions


def _compute_slope(points: List[Dict]) -> float:
    if len(points) < 2:
        return 0.0
    y = [p["avg_confusion"] for p in points]
    n = len(y)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x) or 1.0
    return num / den


def detect_confusion_spike(
    course_id: str,
    lecture_id: Optional[str] = None,
    window: int = 6,
    min_points: int = 4,
    avg_threshold: float = 0.45,
    slope_threshold: float = 0.08,
    bucket_minutes: float = 0.25,
    min_questions: int = 6,
) -> Optional[Tuple[Dict, float, float]]:
    qs = get_questions(course_id, lecture_id=lecture_id)
    trend = compute_confusion_trend(qs, bucket_minutes=bucket_minutes)
    if len(trend) < min_points:
        # Fallback: if enough recent questions are confusing, trigger a medium alert.
        if len(qs) >= min_questions:
            recent = qs[-min_questions:]
            avg_conf = sum(q.get("confusion", 0.0) for q in recent) / max(len(recent), 1)
            if avg_conf >= avg_threshold:
                alert = {
                    "type": "confusion_spike",
                    "message": (
                        f"High confusion detected (avg={avg_conf:.2f}) in recent questions."
                    ),
                    "severity": "medium",
                }
                return alert, avg_conf, 0.0
        return None

    recent = trend[-window:]
    avg_conf = sum(p["avg_confusion"] for p in recent) / max(len(recent), 1)
    slope = _compute_slope(recent)

    if avg_conf >= avg_threshold and slope >= slope_threshold:
        alert = {
            "type": "confusion_spike",
            "message": (
                f"Confusion spike detected (avg={avg_conf:.2f}, slope={slope:.2f})."
            ),
            "severity": "high",
        }
        return alert, avg_conf, slope
    if avg_conf >= avg_threshold and slope < slope_threshold:
        alert = {
            "type": "confusion_spike",
            "message": (
                f"High sustained confusion detected (avg={avg_conf:.2f})."
            ),
            "severity": "medium",
        }
        return alert, avg_conf, slope
    return None


def debug_alert_metrics(
    course_id: str,
    lecture_id: Optional[str] = None,
    bucket_minutes: float = 0.25,
    min_questions: int = 6,
) -> Dict:
    qs = get_questions(course_id, lecture_id=lecture_id)
    trend = compute_confusion_trend(qs, bucket_minutes=bucket_minutes)
    recent_qs = qs[-min_questions:] if qs else []
    avg_recent = (
        sum(q.get("confusion", 0.0) for q in recent_qs) / max(len(recent_qs), 1)
        if recent_qs
        else 0.0
    )
    slope = _compute_slope(trend[-6:]) if len(trend) >= 2 else 0.0
    return {
        "total_questions": len(qs),
        "trend_points": len(trend),
        "recent_count": len(recent_qs),
        "avg_recent_confusion": round(avg_recent, 3),
        "slope_recent": round(slope, 3),
        "bucket_minutes": bucket_minutes,
    }


def create_alert(
    course_id: str,
    lecture_id: Optional[str],
    alert: Dict,
) -> Dict:
    with db_conn() as conn:
        conn.execute(
            alerts.insert().values(
                course_id=course_id,
                lecture_id=lecture_id,
                type=alert["type"],
                message=alert["message"],
                severity=alert["severity"],
                created_at=time.time(),
            )
        )
    return alert


def recent_alert_exists(
    course_id: str,
    lecture_id: Optional[str],
    alert_type: str,
    window_seconds: int = 900,
) -> bool:
    cutoff = time.time() - window_seconds
    with db_conn() as conn:
        stmt = (
            alerts.select()
            .where(alerts.c.course_id == course_id)
            .where(alerts.c.type == alert_type)
            .where(alerts.c.created_at >= cutoff)
        )
        if lecture_id:
            stmt = stmt.where(alerts.c.lecture_id == lecture_id)
        row = conn.execute(stmt.limit(1)).first()
    return row is not None


def list_alerts(
    course_id: str,
    lecture_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    with db_conn() as conn:
        stmt = (
            alerts.select()
            .where(alerts.c.course_id == course_id)
            .order_by(alerts.c.created_at.desc())
            .limit(limit)
        )
        if lecture_id:
            stmt = stmt.where(alerts.c.lecture_id == lecture_id)
        rows = conn.execute(stmt).fetchall()

    out = []
    for r in rows:
        out.append(
            {
                "type": r[3],
                "message": r[4],
                "severity": r[5],
                "created_at": r[6],
                "lecture_id": r[2],
            }
        )
    return out
