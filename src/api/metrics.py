from prometheus_client import Counter

KTAS_PREDICTIONS_COUNTER = Counter(
    "ktas_predictions_total",
    "Total count of KTAS triage predictions by score level",
    labelnames=["ktas_level"]
)