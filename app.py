import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from model_engine import LeadScoringEngine

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Lead Scoring & Classification",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Modern Styling ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .tier-badge-hot {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .tier-badge-warm {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FBBF24;
    }
    .tier-badge-cold {
        background-color: #E2E8F0;
        color: #334155;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #94A3B8;
    }
    .action-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
    }
    .prank-box {
        background: #FFF1F2;
        border-left: 4px solid #E11D48;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model Engine (Cached) ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing ML Ensemble & Semantic Embeddings...")
def get_engine():
    return LeadScoringEngine()

engine = get_engine()

# ── Templates for Quick Testing ───────────────────────────────────────────────
TEMPLATES = {
    "Select a pre-filled template...": None,
    "🔥 Hot Lead: Approved Budget & Immediate Onboarding": {
        "name": "Sarah Jenkins",
        "company_name": "Apex Fintech Solutions",
        "company_size": "51-200",
        "industry": "Finance & Banking",
        "designation": "VP of Engineering",
        "query": "What is the pricing for your enterprise plan? We are ready to get started.",
        "message": "Board has approved the budget. We want to onboard your development team this month.",
        "project_description": "Cross-platform mobile wallet and high-throughput transaction ledger dashboard.",
        "service_required": ["Mobile App Development", "Web Development", "Backend Development", "Cloud Infrastructure"],
        "estimated_budget": "₹12,00,000 - ₹18,00,000",
        "project_duration": "4 months",
        "expected_start_date": "Immediate",
        "urgency": "High",
        "decision_maker": True,
        "lead_source": "Website Contact Form"
    },
    "🤡 Prank / Mocking Lead: ₹5K Budget for 4 Enterprise Services": {
        "name": "Amit Verma",
        "company_name": "SkyRocket Ventures",
        "company_size": "51-200",
        "industry": "Real Estate",
        "designation": "CEO",
        "query": "We have finalized our budget and would like to move forward.",
        "message": "Investors have approved the budget. We want to onboard immediately. Have detailed BRD ready.",
        "project_description": "Society management: visitor gate, Razorpay maintenance, complaint ticketing, analytics dashboard.",
        "service_required": ["Mobile App Development", "Web Development", "Backend Development", "Cloud Infrastructure"],
        "estimated_budget": "₹5,000 - ₹10,000",
        "project_duration": "3 months",
        "expected_start_date": "Within 1 month",
        "urgency": "High",
        "decision_maker": True,
        "lead_source": "Website Contact Form"
    },
    "💼 Warm Lead: Inquiring Extra Services & Scope": {
        "name": "Priya Sharma",
        "company_name": "Zenith Healthcare",
        "company_size": "11-50",
        "industry": "Healthcare",
        "designation": "Product Manager",
        "query": "we do not want website codes and files. Can you provide extra services like UI/UX design?",
        "message": "Looking for UI/UX audit and custom interactive prototype for our hospital management portal.",
        "project_description": "Patient scheduling and medical records interface redesign.",
        "service_required": ["Web Development", "UI/UX Design"],
        "estimated_budget": "₹3,00,000 - ₹6,00,000",
        "project_duration": "2 months",
        "expected_start_date": "Within 1 month",
        "urgency": "Medium",
        "decision_maker": True,
        "lead_source": "LinkedIn Outreach"
    },
    "🚫 Cold Lead: Chose Another Competitor": {
        "name": "Robert Vance",
        "company_name": "Vance Refrigeration",
        "company_size": "51-200",
        "industry": "Logistics",
        "designation": "Director of Operations",
        "query": "We have decided to go with another provider.",
        "message": "We signed an agreement with a local agency last Friday. Please remove our email from your follow-ups.",
        "project_description": "Fleet tracking and driver route optimization system.",
        "service_required": ["Mobile App Development", "Backend Development"],
        "estimated_budget": "₹8,00,000 - ₹12,00,000",
        "project_duration": "5 months",
        "expected_start_date": "Within 2 months",
        "urgency": "Low",
        "decision_maker": False,
        "lead_source": "Cold Outbound"
    },
    "❄️ Cold Lead: Student / Free Browsing": {
        "name": "Alex Carter",
        "company_name": "Independent",
        "company_size": "1-10",
        "industry": "Education",
        "designation": "Student",
        "query": "Just browsing and curious how much an app would cost. No money right now.",
        "message": "College final year project. Looking for free advice or open-source templates.",
        "project_description": "A student attendance tracking app.",
        "service_required": ["Mobile App Development"],
        "estimated_budget": "₹0 - ₹5,000",
        "project_duration": "1 month",
        "expected_start_date": "Sometime next year",
        "urgency": "Low",
        "decision_maker": False,
        "lead_source": "Search Engine"
    }
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/target.png", width=64)
    st.title("Model Insights")
    st.markdown("""
    **Industrial ML Stack**:
    - **Language Model**: `all-MiniLM-L6-v2` (384-dim semantic embeddings)
    - **Dimensionality Reduction**: `TruncatedSVD` (105 components)
    - **Classifier**: Calibrated Stacking Ensemble
      - *Base Models*: Random Forest, HistGradientBoosting, Logistic Regression
      - *Meta Model*: Logistic Regression (Isotonic calibration)
    - **Master Query Gating**: Contrastive & Rejection Intent detection
    - **Economic Feasibility**: Ratio of $\\frac{\\log(\\text{Price})}{\\text{Duration}}$
    """)
    st.divider()
    st.subheader("🧪 Load Test Scenario")
    selected_template_name = st.selectbox(
        "Choose a pre-built lead scenario:",
        list(TEMPLATES.keys())
    )
    st.caption("Selecting a scenario automatically populates the form inputs.")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎯 AI Lead Scoring & Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-modal B2B qualification combining semantic query intent, economic ratio checks, and stacking ensemble models.</div>', unsafe_allow_html=True)

tab_single, tab_batch, tab_docs = st.tabs(["⚡ Single Lead Qualifier", "📂 Batch Lead Evaluation (CSV / JSON)", "🧠 Architecture & Model Logic"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Single Lead Qualifier
# ──────────────────────────────────────────────────────────────────────────────
with tab_single:
    template = TEMPLATES.get(selected_template_name) or {}

    with st.form("lead_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 👤 Prospect & Company")
            lead_name = st.text_input("Contact Name", value=template.get("name", "Rahul Mehta"))
            company_name = st.text_input("Company Name", value=template.get("company_name", "Novus Tech Pvt Ltd"))
            col_a, col_b = st.columns(2)
            with col_a:
                company_size = st.selectbox("Company Size", ["1-10", "11-50", "51-200", "201-500", "500+"], index=2 if not template else ["1-10", "11-50", "51-200", "201-500", "500+"].index(template.get("company_size", "51-200")))
            with col_b:
                industry = st.selectbox("Industry", ["Fintech", "Healthcare", "E-Commerce", "Real Estate", "SaaS / Tech", "Education", "Logistics", "Other"], index=0)
            designation = st.text_input("Job Title / Designation", value=template.get("designation", "Chief Technology Officer"))
            decision_maker = st.checkbox("Contact is Confirmed Decision Maker", value=template.get("decision_maker", True))

            st.markdown("##### 💬 Inquiry Texts")
            query = st.text_area("Client Query (Primary Intent Signal)", value=template.get("query", "We want full deployment as soon as possible."), height=70, help="The primary direct message or form question from the client.")
            message = st.text_area("Message / Additional Context", value=template.get("message", "Investors have approved our Q3 budget. Ready for onboarding."), height=70)

        with col2:
            st.markdown("##### 💼 Project Scope & Budget")
            all_services = ["Web Development", "Mobile App Development", "Backend Development", "Cloud Infrastructure", "UI/UX Design", "AI / ML Integration", "QA & Testing"]
            default_services = template.get("service_required", ["Web Development", "Mobile App Development", "Backend Development"])
            # Ensure valid defaults
            valid_defaults = [s for s in default_services if s in all_services] or ["Web Development"]
            service_required = st.multiselect("Services Required", all_services, default=valid_defaults)

            project_description = st.text_area("Project Description", value=template.get("project_description", "End-to-end multi-tenant customer onboarding portal with microservices architecture."), height=70)

            col_c, col_d = st.columns(2)
            with col_c:
                budget_options = [
                    "₹5,000 - ₹10,000",
                    "₹25,000 - ₹50,000",
                    "₹1,00,000 - ₹3,00,000",
                    "₹3,00,000 - ₹6,00,000",
                    "₹6,00,000 - ₹10,00,000",
                    "₹10,00,000 - ₹15,00,000",
                    "₹15,00,000 - ₹25,00,000",
                    "₹25,00,000+"
                ]
                cur_b = template.get("estimated_budget", "₹10,00,000 - ₹15,00,000")
                b_idx = budget_options.index(cur_b) if cur_b in budget_options else 5
                estimated_budget = st.selectbox("Estimated Budget", budget_options, index=b_idx)
            with col_d:
                duration_options = ["2-3 weeks", "1 month", "2-3 months", "4-6 months", "6-12 months"]
                project_duration = st.selectbox("Estimated Project Duration", duration_options, index=2)

            col_e, col_f = st.columns(2)
            with col_e:
                urgency = st.selectbox("Urgency", ["High", "Medium", "Low"], index=0 if not template else ["High", "Medium", "Low"].index(template.get("urgency", "High")))
            with col_f:
                lead_source = st.selectbox("Lead Source", ["Website Contact Form", "Referral", "Inbound Call", "LinkedIn Outreach", "Cold Outbound"], index=0)

        submitted = st.form_submit_button("🚀 Classify & Score Lead", use_container_width=True, type="primary")

    if submitted:
        lead_dict = {
            "name": lead_name,
            "company_name": company_name,
            "company_size": company_size,
            "industry": industry,
            "designation": designation,
            "decision_maker": decision_maker,
            "query": query,
            "message": message,
            "project_description": project_description,
            "service_required": service_required,
            "estimated_budget": estimated_budget,
            "project_duration": project_duration,
            "urgency": urgency,
            "lead_source": lead_source
        }

        with st.spinner("Analyzing semantic vectors and economic feasibility..."):
            result = engine.classify_lead(lead_dict, use_llm=True)

        st.divider()

        # Display Hero Result
        tier = result["tier"]
        score = result["score"]
        conf = result["confidence"] * 100

        badge_class = "tier-badge-hot" if tier == "Hot" else ("tier-badge-warm" if tier == "Warm" else "tier-badge-cold")
        icon = "🔥" if tier == "Hot" else ("⚡" if tier == "Warm" else "❄️")

        res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])
        with res_col1:
            st.markdown(f'<div class="{badge_class}">{icon} {result["label"]} Lead</div>', unsafe_allow_html=True)
            st.markdown(f"### Final Qualification Score: **{score:.1f} / 100**")
        with res_col2:
            st.metric("Model Confidence", f"{conf:.1f}%")
        with res_col3:
            st.metric("Intent Gate Score", f"{result['intent_gate']:.3f}")

        # Action banner
        if result.get("prank_detected"):
            st.markdown(f"""
            <div class="prank-box">
                <strong style="color: #9F1239; font-size: 1.1rem;">🚨 Prank / Mocking Alert:</strong><br>
                <span>{result['action']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="action-box">
                <strong style="color: #1E40AF; font-size: 1.1rem;">📋 Recommended Sales Action:</strong><br>
                <span>{result['action']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # Charts and Breakdown
        col_chart, col_rationale = st.columns([1, 1])

        with col_chart:
            st.markdown("##### 📊 Class Probabilities")
            proba = result["proba"]
            fig = go.Figure(go.Bar(
                x=[proba["Cold"] * 100, proba["Warm"] * 100, proba["Hot"] * 100],
                y=["Cold", "Warm", "Hot"],
                orientation='h',
                marker=dict(
                    color=['#94A3B8', '#F59E0B', '#EF4444'],
                    line=dict(color='#334155', width=1)
                ),
                text=[f"{proba['Cold']*100:.1f}%", f"{proba['Warm']*100:.1f}%", f"{proba['Hot']*100:.1f}%"],
                textposition='outside'
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 115], title="Probability (%)"),
                yaxis=dict(autorange="reversed"),
                height=220,
                margin=dict(l=20, r=20, t=10, b=30),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"**Economic Ratio:** $\\frac{{\\log(\\text{{Price}})}}{{\\text{{Duration}}}} = {result.get('log_price_dur', 0.0):.2f}$")

        with col_rationale:
            st.markdown("##### 🔍 Model Rationale & Signals")
            for r in result["rationale"]:
                if "[PRANK" in r:
                    st.error(r)
                elif "[REJECTION" in r:
                    st.warning(r)
                elif "[Intent Engine]" in r or "[Semantic Engine]" in r:
                    st.info(r)
                else:
                    st.success(r)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch CSV / JSON Evaluation
# ──────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### 📂 Batch Lead Scoring")
    st.markdown("Upload a CSV or JSON file containing leads to score and qualify them in bulk.")

    col_up, col_dl = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload Leads File", type=["csv", "json"])
    with col_dl:
        st.write("")
        st.write("")
        sample_batch = [
            {"lead_id": "LD-101", "name": "Vikram Singh", "query": "Ready to get started immediately with deployment.", "estimated_budget": "₹15,00,000", "service_required": "Web Development|Backend Development", "project_duration": "4 months", "urgency": "High", "decision_maker": True},
            {"lead_id": "LD-102", "name": "Arun Patel", "query": "We have finalized budget and want to move forward.", "estimated_budget": "₹5,000 - ₹10,000", "service_required": "Web Development|Mobile App Development|Cloud Infrastructure", "project_duration": "3 months", "urgency": "High", "decision_maker": True},
            {"lead_id": "LD-103", "name": "Meera Sen", "query": "We chose another company and signed with them.", "estimated_budget": "₹10,00,000", "service_required": "Mobile App Development", "project_duration": "3 months", "urgency": "Low", "decision_maker": False},
            {"lead_id": "LD-104", "name": "Karan Johar", "query": "Can you provide extra services like UI/UX design?", "estimated_budget": "₹4,00,000", "service_required": "Web Development", "project_duration": "2 months", "urgency": "Medium", "decision_maker": True}
        ]
        sample_csv = pd.DataFrame(sample_batch).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Sample CSV Template", sample_csv, "sample_leads_template.csv", "text/csv")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_leads = pd.read_csv(uploaded_file)
            else:
                raw_json = json.load(uploaded_file)
                df_leads = pd.DataFrame(raw_json)

            st.success(f"Loaded {len(df_leads)} leads successfully!")
            
            if st.button("⚡ Score All Leads", type="primary"):
                progress_bar = st.progress(0)
                scored_records = []

                leads_list = df_leads.to_dict(orient="records")
                for i, ld in enumerate(leads_list):
                    res = engine.classify_lead(ld, use_llm=False)
                    scored_records.append({
                        "Lead ID": ld.get("lead_id", f"LD-{i+1}"),
                        "Name": ld.get("name", "N/A"),
                        "Tier": res["tier"],
                        "Score": res["score"],
                        "Confidence (%)": round(res["confidence"] * 100, 1),
                        "Prank": "🚨 Yes" if res["prank_detected"] else "No",
                        "Recommended Action": res["action"],
                        "Query": ld.get("query", ""),
                        "Budget": ld.get("estimated_budget", "")
                    })
                    progress_bar.progress((i + 1) / len(leads_list))

                res_df = pd.DataFrame(scored_records).sort_values(by="Score", ascending=False)

                # Tier counts
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                tier_counts = res_df["Tier"].value_counts()
                col_m1.metric("Total Scored", len(res_df))
                col_m2.metric("Hot Leads 🔥", tier_counts.get("Hot", 0))
                col_m3.metric("Warm Leads ⚡", tier_counts.get("Warm", 0))
                col_m4.metric("Cold Leads ❄️", tier_counts.get("Cold", 0))

                st.markdown("##### 🏆 Qualification Leaderboard")
                st.dataframe(res_df, use_container_width=True)

                out_csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Scored Results (CSV)", out_csv, "scored_leads_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error parsing file: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Model Architecture & Logic
# ──────────────────────────────────────────────────────────────────────────────
with tab_docs:
    st.markdown("### 🧠 System Architecture & Methodology")
    st.markdown("""
    This lead classification system solves the failure modes of traditional naive rule engines and uncalibrated models:

    #### 1. Master Query Gating
    - Direct queries like *"We have decided to go with another provider"* or *"We do not think your services are promising"* trigger **instant Rejection dampening** even if company size and budget are high.
    - Contrastive statements like *"we do not want website codes and files. We want full deployment"* correctly isolate commercial demand instead of falsely flagging negation keywords.

    #### 2. Economic Sanity & Prank Detection
    - A common adversarial case is when a lead submits enterprise demands (e.g. Mobile App + Web + Backend + Cloud) with a micro-budget (e.g. ₹5,000–₹10,000) while writing *"Budget approved, ready to start"*.
    - The engine computes the economic ratio:
    """)
    st.latex(r"\text{Economic Ratio} = \frac{\ln(\text{Price}_{\text{INR}})}{\max(\text{Duration}_{\text{months}}, 1.0)}")
    st.markdown("""
    - If budget is below economic viability for the requested service count ($< ₹50,000$ for multi-service), the lead is **automatically overridden to `[COLD]`** with an alert flag.

    #### 3. Stacking Ensemble Pipeline
    - **Feature Space (31 dimensions)**: Structured budget, size, urgency, source, start date, query anchor similarities, message similarities, and economic ratios.
    - **SVD Embedding Space**: 105 components extracted from 384-dimensional MiniLM embeddings.
    - **Models**:
      - Random Forest (300 estimators, balanced class weights)
      - HistGradientBoosting Classifier (learning rate 0.05, max depth 6)
      - L2 Regularized Logistic Regression
      - Meta Logistic Regression with **Isotonic Calibration** for well-calibrated probabilities.
    """)
