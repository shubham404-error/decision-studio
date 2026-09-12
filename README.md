# Paisaan by CapitalSense

A public, Streamlit Community Cloud-ready decision-support MVP for self-directed Indian investors.

**Pehchaan your money first. Then plan the next move.** The optional in-app “Paisaan mode” uses light, meme-inspired cues while preserving a sober, explain-then-guide decision experience.

## What it includes

- Portfolio Health Check: CSV/manual holdings, concentration, returns, drawdown, VaR, and risk ratios.
- Goals & Retirement Planner: SIP, step-up, inflation, FIRE, and planning-only NPS allocation scenarios.
- Value a Stock: DCF/WACC base, bear, and bull scenarios.
- Sell / Tax-Aware Return: editable capital-gains assumptions and estimated net proceeds.
- Trade Plan: risk-based position sizing, charges, illustrative margin, option payoff, and Black-Scholes value.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Deploy `app.py` as the entry point. Add this optional secret in the app settings to enable Gemini explanations and consented image extraction:

```toml
GEMINI_API_KEY = "..."
GEMINI_MODEL = "gemini-3.5-flash"
```

The application remains usable without Gemini. Uploaded CSVs and images are processed in session only and are not written to disk, cached, or stored as user history.

## Important limits

- CSV uploads: 5 MB.
- Image uploads: JPG, PNG, or WebP; 10 MB each; five images per session.
- Personal Gemini calls: four per browser session.
- Market data: public Yahoo Finance data only; it can be delayed, incomplete, or unavailable.

This is educational decision support, not investment, tax, legal, or trading advice.
