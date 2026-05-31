from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Result of a single transaction prediction."""
    is_fraud: bool
    fraud_probability: float
    confidence: str  # 'high' | 'medium' | 'low'
    label: str       # 'FRAUD' | 'LEGITIMATE'

    def __repr__(self):
        return (
            f"PredictionResult("
            f"label='{self.label}', "
            f"fraud_probability={self.fraud_probability:.4f}, "
            f"confidence='{self.confidence}')"
        )

    def to_dict(self) -> dict:
        return {
            "is_fraud": self.is_fraud,
            "fraud_probability": self.fraud_probability,
            "confidence": self.confidence,
            "label": self.label,
        }
