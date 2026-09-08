"""Optional Gemini helper. Personal analysis is intentionally never globally cached."""

from __future__ import annotations

import json
from typing import Any


PERSONAL_CALL_LIMIT = 4


def gemini_available(secrets: Any) -> bool:
    try:
        return bool(secrets.get("GEMINI_API_KEY"))
    except Exception:
        return False


def explain_result(api_key: str, model: str, question: str, result_context: dict[str, Any], confirmed_context: dict[str, Any] | None = None) -> str:
    """Ask Gemini to explain supplied deterministic data without manufacturing numbers."""
    from google import genai

    prompt = f"""You are the CapitalSense Decision Studio explainer for self-directed Indian investors.
Use ONLY the deterministic result and confirmed user context below. Do not invent values, prices, tax rules, or market facts.
Do not give a buy/sell instruction or guarantee an outcome. Explain uncertainty, assumptions, and what the user can check next.

User question: {question}
Deterministic result: {json.dumps(result_context, default=str)}
Confirmed context: {json.dumps(confirmed_context or {}, default=str)}

Respond with exactly these concise sections:
1. What it means
2. What is driving it
3. What could change it
4. A safe next check
"""
    client = genai.Client(api_key=api_key)
    import time
    delay = 2
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return str(response.text or "No explanation was returned.")
        except Exception as e:
            error_str = str(e).lower()
            if "503" in error_str or "unavailable" in error_str or "429" in error_str or "quota" in error_str:
                if attempt == 2:
                    return "?? **AI Insights Temporarily Unavailable**\n\nThe Gemini AI model is currently experiencing high demand. Please wait a few minutes and try again."
                time.sleep(delay)
                delay *= 2
            else:
                if attempt == 2:
                    return f"?? **Error Fetching Insights:**\n\n{str(e)}"
                time.sleep(delay)
                delay *= 2
    return "?? **Error:** Max retries exceeded."


def extract_image_records(api_key: str, model: str, image: Any) -> ExtractionResult:
    """Extract a conservative holdings/transaction draft from a consented image."""
    from google import genai
    from .documents import ExtractionResult

    prompt = """Extract only visible investment holdings or transactions from this financial document image.
Return strict JSON with keys: records (array), confidence (0 to 1), warnings (array).
Each record may contain symbol, instrument_name, quantity, average_cost, current_price, value, transaction_date, transaction_amount.
Do not infer missing values. Do not return account numbers, PAN, addresses, or other personal identifiers."""
    client = genai.Client(api_key=api_key)
    import time
    delay = 2
    payload = {}
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, image],
                config={"response_mime_type": "application/json"},
            )
            payload = json.loads(response.text or "{}")
            break
        except Exception as e:
            if attempt == 2:
                payload = {"warnings": [f"AI Error: {str(e)}"]}
            time.sleep(delay)
            delay *= 2
    records = payload.get("records", [])
    if not isinstance(records, list):
        records = []
    return ExtractionResult(
        records=records,
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        warnings=[str(item) for item in payload.get("warnings", [])],
        requires_confirmation=True,
    )
