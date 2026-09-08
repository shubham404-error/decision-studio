"""Session-only CSV/image handling. These helpers never write user content to disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import re
from typing import Any

import pandas as pd
from PIL import Image, ImageOps


CSV_MAX_BYTES = 5 * 1024 * 1024
IMAGE_MAX_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 5
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@dataclass
class ExtractionResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    requires_confirmation: bool = True


def parse_holdings_csv(raw: bytes) -> ExtractionResult:
    if len(raw) > CSV_MAX_BYTES:
        return ExtractionResult(warnings=["CSV exceeds the 5 MB session limit."])
    try:
        frame = pd.read_csv(BytesIO(raw))
    except Exception as exc:
        return ExtractionResult(warnings=[f"The CSV could not be read: {exc}"])
    normalized = {str(column).strip().lower().replace(" ", "_"): column for column in frame.columns}
    aliases = {
        "symbol": ["symbol", "ticker", "security", "instrument", "stock"],
        "quantity": ["quantity", "qty", "units", "shares"],
        "average_cost": ["average_cost", "avg_cost", "cost_price", "buy_price"],
        "current_price": ["current_price", "ltp", "price", "market_price"],
        "value": ["value", "market_value", "current_value"],
    }
    matched: dict[str, str] = {}
    for target, candidates in aliases.items():
        source = next((normalized[name] for name in candidates if name in normalized), None)
        if source:
            matched[target] = source
    if "symbol" not in matched:
        return ExtractionResult(warnings=["CSV needs a symbol/ticker/security column."], confidence=0.0)
    records = []
    for _, row in frame.head(200).iterrows():
        symbol = str(row[matched["symbol"]]).strip().upper()
        if not symbol or symbol == "NAN":
            continue
        record = {"symbol": symbol}
        for target in ("quantity", "average_cost", "current_price", "value"):
            if target in matched:
                record[target] = pd.to_numeric(row[matched[target]], errors="coerce")
                record[target] = 0.0 if pd.isna(record[target]) else float(record[target])
        if not record.get("value") and record.get("quantity") and record.get("current_price"):
            record["value"] = record["quantity"] * record["current_price"]
        records.append(record)
    warnings = []
    if len(frame) > 200:
        warnings.append("Only the first 200 rows were loaded for this public-app session.")
    if not any(record.get("value", 0) for record in records):
        warnings.append("Add market value or quantity and current price to calculate concentration.")
    return ExtractionResult(records=records, confidence=0.92 if records else 0.0, warnings=warnings)


def prepare_image(raw: bytes, mime_type: str) -> tuple[Image.Image | None, str | None]:
    """Decode, remove metadata through re-rendering, and cap image dimensions in memory."""
    if mime_type not in ALLOWED_IMAGE_TYPES:
        return None, "Only JPG, PNG, and WebP images are accepted."
    if len(raw) > IMAGE_MAX_BYTES:
        return None, "Image exceeds the 10 MB session limit."
    try:
        with Image.open(BytesIO(raw)) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((1600, 1600))
            return image.copy(), None
    except Exception:
        return None, "This image could not be processed. Please upload a clear JPG, PNG, or WebP image."


def redact_for_display(text: str) -> str:
    """Mask obvious account/card-like sequences in user-visible extraction previews."""
    return re.sub(r"\b(?:\d[ -]?){8,18}\b", "••••••••", text)
