from dataclasses import dataclass
from typing import List, Dict, Any, Callable


@dataclass
class AlertRule:
    name: str
    metric: str
    operator: str
    threshold: float

    def check(self, value: float) -> bool:
        ops: Dict[str, Callable[[float, float], bool]] = {
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }
        fn = ops.get(self.operator, lambda *_: False)
        return fn(value, self.threshold)


class AlertManager:
    def __init__(self):
        self.rules: List[AlertRule] = []

    def set_rules(self, rules: List[AlertRule]) -> None:
        self.rules = rules

    def evaluate(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        alerts = []
        for rule in self.rules:
            val = metrics.get(rule.metric)
            if val is None:
                continue
            if rule.check(val):
                alerts.append({"name": rule.name, "metric": rule.metric, "value": val})
        return alerts

