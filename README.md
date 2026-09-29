# 🎯 AI Lead Scoring & Classification Dashboard

An industrial-grade B2B lead qualification system combining semantic query intent gating, economic feasibility checks, and a calibrated machine learning stacking ensemble.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🚀 Key Features

- **Master Query Gating**: Semantic vector analysis with `all-MiniLM-L6-v2` isolating buyer readiness (`[HOT]`), scope/custom inquiries (`[WARM]`), and competitor/disinterest rejections (`[COLD]`).
- **Economic Feasibility & Prank Detection**: Evaluates service count against budget and duration using the ratio:
  $$\text{Economic Ratio} = \frac{\ln(\text{Price}_{\text{INR}})}{\max(\text{Duration}_{\text{months}}, 1.0)}$$
  Automatically flags and suppresses impossible micro-budgets (e.g. ₹5,000 for 4 enterprise services) to `[COLD]`.
- **Calibrated Stacking Ensemble**: Random Forest + HistGradientBoosting + Logistic Regression meta-learner with Isotonic Probability Calibration.
- **Interactive Streamlit Web Dashboard**:
  - Live single-lead scoring with pre-filled test scenarios.
  - Probability distribution gauge charts.
  - Batch CSV / JSON qualification with exportable leaderboards.

---

## 🛠️ Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Local Web App
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2. Click **"New app"**.
3. Select your repository: `sayanroy2710-byte/Lead-Classification`.
4. Set Main file path: `app.py`.
5. Click **"Deploy!"**.
