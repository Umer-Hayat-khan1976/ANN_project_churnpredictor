"""Customer Churn Predictor: Streamlit front end for the trained ANN."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import streamlit as st
from app_utils import load_artifacts, predict_proba

st.set_page_config(page_title="Customer Churn Predictor", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
html, body, [class*="css"] { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.hero { background: linear-gradient(120deg, #0f2447 0%, #1d4e89 100%);
        padding: 28px 34px; border-radius: 12px; margin-bottom: 24px; }
.hero h1 { color: #ffffff; font-size: 1.85rem; font-weight: 600; margin: 0 0 6px 0; letter-spacing: .2px; }
.hero p  { color: #c7d6ec; font-size: .98rem; margin: 0; }

.section-title { font-size: .78rem; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase;
                 color: #6b7a90; margin: 6px 0 10px 0; border-bottom: 1px solid #e3e8ef; padding-bottom: 6px; }

.card { border: 1px solid #e3e8ef; border-radius: 12px; padding: 22px 24px; background: #ffffff; }
.verdict { border-radius: 12px; padding: 22px 24px; margin-bottom: 16px; }
.verdict .label { font-size: .75rem; letter-spacing: 1.2px; text-transform: uppercase; font-weight: 700; opacity: .85; }
.verdict .text  { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
.verdict .sub   { font-size: .92rem; margin-top: 6px; opacity: .9; }
.leave { background: #fdecec; color: #8a1c1c; border: 1px solid #f3c3c3; }
.stay  { background: #e8f5ee; color: #14532d; border: 1px solid #b9e0c9; }
.idle  { background: #f3f6fa; color: #44546a; border: 1px dashed #c5d0de; text-align: center; padding: 48px 24px; }

.meter { height: 12px; background: #e6ebf2; border-radius: 6px; position: relative; margin: 10px 0 6px 0; }
.meter .fill { height: 100%; border-radius: 6px; }
.meter .mark { position: absolute; top: -4px; width: 2px; height: 20px; background: #1f2937; }
.meter-legend { display: flex; justify-content: space-between; font-size: .75rem; color: #6b7a90; }

div[data-testid="stMetric"] { background: #f7f9fc; border: 1px solid #e3e8ef; border-radius: 10px; padding: 12px 16px; }
div.stButton > button { background: #1d4e89; color: #fff; border: 0; border-radius: 8px; height: 3rem;
                        font-weight: 600; letter-spacing: .4px; width: 100%; }
div.stButton > button:hover { background: #0f2447; color: #fff; }
section[data-testid="stSidebar"] { background: #f7f9fc; border-right: 1px solid #e3e8ef; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


try:
    get_artifacts()
except Exception as exc:
    st.error("Model files not found. Run train.ipynb first to create models/model.keras and models/preprocessor.pkl.")
    st.exception(exc)
    st.stop()

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.markdown("### Model Information")
    st.markdown(
        "**Architecture**  \nInput (13) > Dense 16 (ReLU) > Dense 8 (ReLU) > Dense 1 (Sigmoid)\n\n"
        "**Preprocessing**  \nOne-Hot Encoding and StandardScaler\n\n"
        "**Training**  \nAdam, binary cross-entropy, EarlyStopping (patience 5)\n\n"
        "**Reported test accuracy**  \n86.2 percent")
    st.markdown("---")
    st.markdown("### Decision Threshold")
    threshold = st.slider("Classify as LEAVE at or above", 0.30, 0.70, 0.50, 0.01,
                          help="Lower values flag more customers as at risk and increase recall.")
    st.markdown("---")
    st.markdown("### Quick Scenarios")
    scenario = st.radio("Load a preset profile", ["Custom", "Loyal customer", "High-risk customer"],
                        label_visibility="collapsed")

PRESETS = {
    "Custom": dict(CreditScore=650, Geography="France", Gender="Male", Age=40, Tenure=5,
                   Balance=75000.0, NumOfProducts=2, HasCrCard="Yes", IsActiveMember="Yes", EstimatedSalary=100000.0),
    "Loyal customer": dict(CreditScore=650, Geography="France", Gender="Male", Age=38, Tenure=5,
                           Balance=76000.0, NumOfProducts=2, HasCrCard="Yes", IsActiveMember="Yes", EstimatedSalary=100000.0),
    "High-risk customer": dict(CreditScore=650, Geography="Germany", Gender="Female", Age=60, Tenure=5,
                               Balance=120000.0, NumOfProducts=1, HasCrCard="Yes", IsActiveMember="No", EstimatedSalary=100000.0),
}
P = PRESETS[scenario]
k = scenario  # widget keys change with the preset so defaults refresh

# ------------------------------ Header ------------------------------
st.markdown(
    '<div class="hero"><h1>Customer Churn Predictor</h1>'
    '<p>Artificial Neural Network that estimates the probability that a bank customer will leave.</p></div>',
    unsafe_allow_html=True)

left, right = st.columns([1.25, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    geography = c1.selectbox("Geography", ["France", "Germany", "Spain"],
                             index=["France", "Germany", "Spain"].index(P["Geography"]), key=f"geo{k}")
    gender = c2.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(P["Gender"]), key=f"gen{k}")
    age = c3.number_input("Age", 18, 100, P["Age"], key=f"age{k}")

    st.markdown('<div class="section-title">Financial Details</div>', unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    credit = c4.slider("Credit score", 300, 900, P["CreditScore"], key=f"cs{k}")
    tenure = c5.slider("Tenure (years)", 0, 10, P["Tenure"], key=f"ten{k}")
    c6, c7 = st.columns(2)
    balance = c6.number_input("Account balance", 0.0, 300000.0, P["Balance"], step=1000.0, key=f"bal{k}")
    salary = c7.number_input("Estimated annual salary", 0.0, 300000.0, P["EstimatedSalary"], step=1000.0, key=f"sal{k}")

    st.markdown('<div class="section-title">Banking Relationship</div>', unsafe_allow_html=True)
    c8, c9, c10 = st.columns(3)
    products = c8.selectbox("Number of products", [1, 2, 3, 4], index=P["NumOfProducts"] - 1, key=f"np{k}")
    has_card = c9.selectbox("Has credit card", ["Yes", "No"], index=["Yes", "No"].index(P["HasCrCard"]), key=f"cc{k}")
    active = c10.selectbox("Active member", ["Yes", "No"], index=["Yes", "No"].index(P["IsActiveMember"]), key=f"am{k}")

    st.write("")
    run = st.button("Predict churn risk")

with right:
    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
    if run:
        customer = dict(CreditScore=credit, Geography=geography, Gender=gender, Age=age, Tenure=tenure,
                        Balance=balance, NumOfProducts=products, HasCrCard=1 if has_card == "Yes" else 0,
                        IsActiveMember=1 if active == "Yes" else 0, EstimatedSalary=salary)
        prob = predict_proba(customer)
        leaves = prob >= threshold
        css, head = ("leave", "Likely to LEAVE") if leaves else ("stay", "Likely to STAY")
        color = "#c0392b" if leaves else "#1e8449"
        risk = "High" if prob >= 0.6 else ("Moderate" if prob >= 0.3 else "Low")

        st.markdown(
            f'<div class="verdict {css}"><div class="label">Verdict</div>'
            f'<div class="text">{head}</div>'
            f'<div class="sub">Churn probability {prob*100:.1f} percent against a {threshold*100:.0f} percent threshold.</div></div>',
            unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{prob*100:.1f}%")
        m2.metric("Risk level", risk)

        st.markdown(
            f'<div class="meter"><div class="fill" style="width:{prob*100:.1f}%;background:{color}"></div>'
            f'<div class="mark" style="left:{threshold*100:.0f}%"></div></div>'
            f'<div class="meter-legend"><span>0%</span><span>Threshold marker at {threshold*100:.0f}%</span><span>100%</span></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="verdict idle"><div style="font-weight:600;font-size:1.05rem">No prediction yet</div>'
            '<div style="margin-top:6px;font-size:.9rem">Enter the customer details and select Predict churn risk.</div></div>',
            unsafe_allow_html=True)

st.markdown("---")
st.caption("This tool supports retention planning and is not a substitute for human judgement. "
           "Predictions are statistical estimates produced by a model trained on 10,000 historical customers.")
