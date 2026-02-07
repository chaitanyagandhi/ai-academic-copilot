from typing import List, Dict
from datetime import datetime

def compute_confusion_trend(questions: List[dict], bucket_minutes: int = 1) -> List[Dict]:
    """
    Groups confusion scores into time buckets.
    Returns a list of points sorted by time.
    """
    if not questions:
        return []

    buckets = {}

    for q in questions:
        ts = q.get("timestamp")
        confusion = q.get("confusion", 0.0)

        # bucket timestamp
        bucket = int(ts // (bucket_minutes * 60))
        buckets.setdefault(bucket, []).append(confusion)

    trend = []
    for bucket in sorted(buckets.keys()):
        avg_confusion = sum(buckets[bucket]) / len(buckets[bucket])
        bucket_time = datetime.fromtimestamp(bucket * bucket_minutes * 60)

        trend.append({
            "time": bucket_time.isoformat(),
            "avg_confusion": round(avg_confusion, 3),
            "count": len(buckets[bucket])
        })

    return trend
