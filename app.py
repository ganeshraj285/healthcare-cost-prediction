import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
from reportlab.pdfgen import canvas
import os

st.set_page_config(page_title="Health Insurance App", layout="wide")

# ================= SESSION =================
if "pred" not in st.session_state:
    st.session_state.pred = None

# ===================== CLEAN CSS =====================
st.markdown("""
<style>

html, body {
    font-family: 'Segoe UI', Tahoma, Verdana, sans-serif;
    background:#F2F7FF;
}

/* Remove ugly white spacing */
.block-container {
    padding-top: 0.2rem !important;
    padding-bottom: 1rem !important;
}

/* HEADER */
.title-board{
    background: linear-gradient(120deg,#0028FF,#00C6FF);
    padding:24px;
    border-radius:22px;
    color:white;
    text-align:center;
    font-size:40px;
    font-weight:900;
    margin-bottom:8px;
}

/* SECTION CARD */
.section-box{
    background:white;
    padding:22px;
    border-radius:18px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.18);
    margin-top:14px;
}

/* heading */
.section-heading{
    font-size:26px;
    font-weight:900;
    color:#007BFF;
    margin-bottom:10px;
}

/* BMI CARD */
.bmi-card{
    background:#E8F5FF;
    padding:18px;
    border-radius:16px;
    border:2px solid #1E90FF;
    text-align:center;
    font-size:24px;
    font-weight:900;
}

/* PREMIUM RESULT CARD */
.premium-card{
    background:white;
    padding:22px;
    border-radius:16px;
    border:3px solid #1E90FF;
    text-align:center;
    font-size:28px;
    font-weight:900;
    color:#003B73;
}

/* Final Table Card */
.final-table{
    background:white;
    padding:18px;
    border-radius:16px;
    box-shadow:0px 8px 22px rgba(0,0,0,0.22);
}

/* Charts Card */
.chart-box{
    background:white;
    padding:14px;
    border-radius:16px;
    box-shadow:0 8px 25px rgba(0,0,0,0.2);
}

/* -------- BUTTON PERFECT CENTER -------- */
.center-btn{
    width:100%;
    display:flex;
    justify-content:center;
}

/* make button attractive */
.stButton>button {
    background:linear-gradient(120deg,#007BFF,#00C6FF);
    color:white;
    border-radius:10px;
    padding:10px 22px;
    border:none;
    font-size:18px;
    font-weight:700;
    box-shadow:0px 4px 10px rgba(0,0,0,0.25);
}

</style>
""",unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="title-board">🏥 Health Insurance Premium Prediction System</div>', unsafe_allow_html=True)

# Load Model
model = pickle.load(open("insurance_model.pkl","rb"))

# ================= USER INPUT =================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">✍ Step 1: Enter User Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("👤 Full Name")
    age = st.number_input("📅 Age",1,100)
    gender = st.selectbox("⚧ Gender",["Male","Female","Other"])
    smoker = st.selectbox("🚬 Smoking Status",["no","yes"])

with col2:
    height = st.number_input("📏 Height (cm)",50.0)
    weight = st.number_input("⚖ Weight (kg)",1.0)
    region = st.text_input("🌍 Region")

st.markdown('</div>', unsafe_allow_html=True)


# ================= BMI =================
if height and weight:
    bmi = round(weight / ((height/100)**2),2)
else:
    bmi = None


# ================= PREDICTION SECTION =================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">🚀 Step 2: Insurance Premium Prediction</div>', unsafe_allow_html=True)

if bmi:
    st.markdown(f'<div class="bmi-card">Your BMI : {bmi}</div>', unsafe_allow_html=True)

    st.markdown('<div class="center-btn">', unsafe_allow_html=True)
    predict_clicked = st.button("🎯 Predict Insurance Premium")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        smoker_val = 1 if smoker=="yes" else 0
        st.session_state.pred = round(model.predict(np.array([[age,bmi,smoker_val]]))[0],2)

        st.markdown(
            f'<div class="premium-card">Your Estimated Insurance Premium Cost : ₹ {st.session_state.pred}</div>',
            unsafe_allow_html=True
        )

        # -------- Final Summary Table --------
        df = pd.DataFrame([{
            "Name":name,
            "Age":age,
            "Gender":gender,
            "Height(cm)":height,
            "Weight(kg)":weight,
            "BMI":bmi,
            "Smoker":smoker,
            "Region":region,
            "Predicted Cost (₹)":st.session_state.pred
        }])

        st.markdown('<div class="final-table">', unsafe_allow_html=True)
        st.table(df)
        st.markdown('</div>', unsafe_allow_html=True)

        # Save History
        history_file = "prediction_history.csv"
        if os.path.exists(history_file):
            old = pd.read_csv(history_file)
            df = pd.concat([old,df],ignore_index=True)
        df.to_csv(history_file,index=False)

else:
    st.warning("Enter valid height & weight")

# ================= HISTORY TABLE =================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">📜 Saved Prediction History</div>', unsafe_allow_html=True)

if os.path.exists("prediction_history.csv"):
    hist = pd.read_csv("prediction_history.csv")
    st.dataframe(hist, use_container_width=True)
else:
    st.info("No history yet. Make predictions to build history.")

st.markdown('</div>', unsafe_allow_html=True)


# ================= DASHBOARD =================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

if os.path.exists("prediction_history.csv"):
    data = pd.read_csv("prediction_history.csv")
    st.success("Showing Insights from Saved Users")

    colA,colB = st.columns(2)

    with colA:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        chart1 = alt.Chart(data).mark_circle(size=120,color="#0077ff").encode(
            x="Age", y="Predicted Cost (₹)"
        )
        st.altair_chart(chart1,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        chart2 = alt.Chart(data).mark_circle(size=120,color="#ff0066").encode(
            x="BMI", y="Predicted Cost (₹)"
        )
        st.altair_chart(chart2,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.bar_chart(data.groupby("Smoker")["Predicted Cost (₹)"].mean())
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("No history yet. Make predictions to build dashboard.")


# ================= PDF DOWNLOAD =================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">📥 Download Report</div>', unsafe_allow_html=True)

def create_pdf(name,age,bmi,smoker,pred):
    file = "report.pdf"
    c = canvas.Canvas(file)
    c.setFont("Helvetica",16)
    c.drawString(100,800,"Health Insurance Prediction Report")
    c.setFont("Helvetica",12)
    c.drawString(100,760,f"Name: {name}")
    c.drawString(100,740,f"Age: {age}")
    c.drawString(100,720,f"BMI: {bmi}")
    c.drawString(100,700,f"Smoker: {smoker}")
    c.drawString(100,680,f"Predicted Cost: Rs {pred}")
    c.save()
    return file

st.markdown('<div class="center-btn">', unsafe_allow_html=True)
pdf_clicked = st.button("📄 Download PDF Report")
st.markdown('</div>', unsafe_allow_html=True)

if pdf_clicked:
    if bmi and st.session_state.pred:
        f = create_pdf(name,age,bmi,smoker,st.session_state.pred)
        with open(f,"rb") as file:
            st.download_button("Download Now",file,file_name="Insurance_Report.pdf")
    else:
        st.error("Please Predict Premium First!")

st.markdown('</div>', unsafe_allow_html=True)
