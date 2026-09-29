# 🚀 Online Deployment Guide: AI Lead Scoring Dashboard

This guide explains how to deploy your **AI Lead Scoring & Classification Dashboard** online so anyone on your team or clients can access it via a public URL.

---

## 🌟 Option 1: Streamlit Community Cloud (Recommended — 100% Free)

Streamlit Community Cloud hosts your app directly from GitHub with zero server configuration.

### Steps:
1. **Create a GitHub Repository**:
   - Push this project folder (`Lead Classification/`) to a GitHub repository (e.g. `github.com/<your-username>/lead-classifier`).
   - Ensure the following files are in the repository:
     - `app.py`
     - `model_engine.py`
     - `dataset.json`
     - `requirements.txt`
     - `.streamlit/config.toml`
     - `artifacts/lead_model.joblib` *(optional, if omitted the app will auto-train on startup)*

2. **Deploy on Streamlit**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
   - Click **"New app"**.
   - Select your repository, branch (`main`), and set the main file path to:
     ```text
     app.py
     ```
   - Click **"Deploy!"**.
   - Your public URL will be live at: `https://<your-app-name>.streamlit.app`.

---

## 🤗 Option 2: Hugging Face Spaces (100% Free — 16 GB RAM CPU)

Hugging Face Spaces is great for machine learning apps because it provides a free 16 GB RAM CPU container.

### Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create a free account.
2. Click **"Create new Space"**.
3. Choose:
   - **Space SDK**: `Streamlit`
   - **Hardware**: `CPU basic • 2 vCPU • 16GB RAM` (Free)
4. Upload or git push the project files (`app.py`, `model_engine.py`, `requirements.txt`, `dataset.json`, etc.).
5. Hugging Face will automatically install dependencies and launch your public dashboard!

---

## ⚡ Option 3: Render (Free Web Service)

1. Push your code to GitHub.
2. Go to [render.com](https://render.com) and create a **New Web Service**.
3. Connect your repository.
4. Set the build and start commands:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Create Web Service"**.

---

## 💻 Local Testing & Usage

To test the dashboard on your machine at any time:
```powershell
& "C:\PROJECTS\Python Environments\S.Roy_Pytorch_env\Scripts\streamlit.exe" run app.py
```
Then open: **`http://localhost:8501`**
