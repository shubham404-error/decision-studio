"""Shared, UI-independent contracts for all calculation engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from hashlib import sha256
import json
from typing import Any


@dataclass(frozen=True)
class CalculationRequest:
    calculator_id: str
    inputs: dict[str, Any]
    as_of_date: date | None = None
    tax_year: str | None = None
    market_data_context: dict[str, Any] = field(default_factory=dict)
    scenario_name: str = "Base case"

    @property
    def input_fingerprint(self) -> str:
        payload = asdict(self)
        payload["as_of_date"] = self.as_of_date.isoformat() if self.as_of_date else None
        raw = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class CalculationResult:
    calculator_id: str
    outputs: dict[str, Any]
    tables_and_chart_series: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data_quality: list[str] = field(default_factory=list)
    formula_version: str = "2026.1"
    input_fingerprint: str = ""

    def as_context(self) -> dict[str, Any]:
        """Safe, compact context for an AI explainer after user confirmation."""
        return {
            "calculator_id": self.calculator_id,
            "outputs": self.outputs,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "data_quality": self.data_quality,
            "formula_version": self.formula_version,
        }
