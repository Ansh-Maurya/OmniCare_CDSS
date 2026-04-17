import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import os
import base64
from io import BytesIO
from datetime import datetime
from fpdf import FPDF

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="OmniCare CDSS", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED MODERN STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main { background-color: #f1f5f9; }
    
    .clinical-card {
        background: white; 
        padding: 2rem; 
        border-radius: 16px;
        border: 1px solid #e2e8f0; 
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .brand-text {
        color: #0f172a;
        font-weight: 800;
        font-size: 1.8rem;
        letter-spacing: -1px;
    }
    
    .version-badge {
        background: #e2e8f0;
        color: #475569;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .stButton>button {
        width: 100%; 
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: white !important; 
        font-weight: 600; 
        border-radius: 10px; 
        border: none;
        height: 3.5rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .disclaimer-footer {
        font-size: 0.85rem; color: #94a3b8; text-align: center;
        padding: 3rem; border-top: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2.5 LOCAL IMAGE ENCODING ---
def get_base64_img(file_name):
    try:
        path = os.path.join("image", file_name)
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

img_doc = get_base64_img("doctor.png")
img_pdf = get_base64_img("pdf.png")
img_rep = get_base64_img("report.png")
img_hist = get_base64_img("medical_history.png")
img_det = get_base64_img("details.png")
img_pat = get_base64_img("patient.png")
img_find = get_base64_img("assessment.png")

# --- 2.8 PATIENT LOGGING SYSTEM ---
LOG_FILE = "patient_records.csv"

def log_patient_data(name, age, sex, results):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    res_str = " | ".join([f"{k}: {v*100:.1f}%" for k, v in results.items()])
    new_data = pd.DataFrame([[timestamp, name, age, sex, res_str]], 
                            columns=['Timestamp', 'Name', 'Age', 'Sex', 'Diagnostic Results'])
    
    if not os.path.isfile(LOG_FILE):
        new_data.to_csv(LOG_FILE, index=False)
    else:
        new_data.to_csv(LOG_FILE, mode='a', header=False, index=False)

def clear_all_logs():
    if os.path.exists("patient_records.csv"):
        os.remove("patient_records.csv")
        st.success("Log file deleted successfully!")
    else:
        st.info("No log file found to delete.")

# --- 3. HELPER FUNCTIONS ---
# Updated helper for better compatibility
def generate_clinical_pdf(patient_name, age, sex, results):
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # --- HEADER ---
        pdf.set_fill_color(15, 23, 42)  # Dark Blue/Slate
        pdf.rect(0, 0, 210, 40, 'F')
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(0, 20, "OMNICARE CDSS", ln=True, align='C')
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 5, "Official Clinical Assessment Report | Confidential", ln=True, align='C')
        
        pdf.ln(25) # Space after header
        
        # --- PATIENT INFO SECTION ---
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "1. Patient Information", ln=True)
        pdf.set_draw_color(226, 232, 240)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(40, 8, "Full Name:", 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"{patient_name}", ln=True)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(40, 8, "Age / Sex:", 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"{age} years old / {sex}", ln=True)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(40, 8, "Report Date:", 0)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"{datetime.now().strftime('%d %B %Y, %H:%M')}", ln=True)
        
        pdf.ln(10)
        
        # --- DIAGNOSTIC RESULTS SECTION ---
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "2. Diagnostic Risk Assessment", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Table Header
        pdf.set_fill_color(241, 245, 249)
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(80, 10, " Condition", 1, 0, 'L', True)
        pdf.cell(50, 10, " Probability (%)", 1, 0, 'C', True)
        pdf.cell(60, 10, " Risk Level", 1, 1, 'C', True)
        
        # Table Body
        pdf.set_font("Arial", '', 10)
        for disease, prob in results.items():
            prob_pct = f"{prob*100:.1f}%"
            if prob > 0.6:
                risk_lvl = "HIGH RISK"
            elif prob > 0.3:
                risk_lvl = "MODERATE RISK"
            else:
                risk_lvl = "LOW RISK"
                
            pdf.cell(80, 10, f" {disease}", 1)
            pdf.cell(50, 10, f"{prob_pct}", 1, 0, 'C')
            pdf.cell(60, 10, f"{risk_lvl}", 1, 1, 'C')
            
        pdf.ln(15)
        
        # --- FOOTER / DISCLAIMER ---
        pdf.set_y(-40)
        pdf.set_font("Arial", 'I', 8)
        pdf.set_text_color(148, 163, 184)
        pdf.multi_cell(0, 5, "Disclaimer: This report is generated by an Artificial Intelligence system for clinical decision support. It should be used as a supplementary tool and not as a final diagnosis. Final medical decisions must be made by a qualified healthcare professional.", align='C')
        
        return pdf.output(dest='S').encode('latin-1', 'ignore')
    except Exception as e:
        st.error(f"PDF Styling Error: {e}")
        return None

# --- 4. ASSET LOADING ---
@st.cache_resource
def load_ml_assets():
    try:
        d_model = joblib.load('models/diabetes_model_optimized.pkl')
        h_model = joblib.load('models/heart_model.pkl')
        k_model = joblib.load('models/ckd_model.pkl')
        d_scale = joblib.load('models/scaler.pkl')
        h_scale = joblib.load('models/heart_scaler.pkl')
        k_scale = joblib.load('models/ckd_scaler.pkl')
        return d_model, h_model, k_model, d_scale, h_scale, k_scale
    except Exception as e:
        st.error(f"Asset loading failed. Error: {e}")
        return [None]*6

d_model, h_model, k_model, d_scale, h_scale, k_scale = load_ml_assets()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="brand-text">OmniCare <span style="color:#10b981;">CDSS</span></div>', unsafe_allow_html=True)
    st.markdown('<span class="version-badge">v1.2.0 PROD</span>', unsafe_allow_html=True)
    st.write("")
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="data:image/png;base64,{img_doc}" width="30"/>
        <span style="font-size: 1.2rem; font-weight: 600;">Clinician Access</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Active Session: Admin")
    
    if st.button("Reset Diagnostics"):
        st.session_state.clear()
        st.rerun()
        
    if st.sidebar.button("🗑️ Wipe All Records"):
        clear_all_logs()
        st.rerun()
# --- 6. MAIN UI ---
st.markdown("<h2 style='margin-bottom:0;'>Clinical Decision Support Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b; margin-bottom:2rem;'>Multivariate risk assessment for metabolic and renal dysfunction.</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Patient Intake", "Evidence Analysis", "Patient History Log"])

with tab1:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    with st.form("main_assessment"):
        st.markdown(f'<div class="section-header"><img src="data:image/png;base64,{img_pat}" width="32"/> Patient Identification</div>', unsafe_allow_html=True)
        p_name = st.text_input("Full Patient Name", placeholder="Enter name for reporting...")
        
        st.markdown(f'<div class="section-header"><img src="data:image/png;base64,{img_det}" width="32"/> Demographics & Physicals</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Patient Age", 1, 120, 30)
            sex = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)
            bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 24.5)
        with col2:
            gen_hlth = st.select_slider("General Subjective Health", options=[1,2,3,4,5], help="1: Excellent, 5: Poor")
            phys_act = st.toggle("Regular Physical Activity", value=True)
            target_diseases = st.multiselect("Diagnostic Targets", ["Diabetes", "Heart Disease", "CKD"], default=["Diabetes"])
        with col3:
            st.write("**Pre-existing Conditions**")
            high_bp = st.checkbox("Hypertension")
            high_chol = st.checkbox("Hypercholesterolemia")
            smoker = st.checkbox("Nicotine Use")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header"><img src="data:image/png;base64,{img_hist}" width="32"/> Medical History Tags</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            stroke = st.checkbox("History of Stroke")
            hvy_alc = st.checkbox("High Alcohol Intake")
        with a2:
            diff_walk = st.checkbox("Mobility Impairment")
            healthcare = st.toggle("Insured Patient", value=True)
        with a3:
            phys_days = st.slider("Recent Physical Illness (Days)", 0, 30, 0)
            ment_days = st.slider("Recent Mental Stress (Days)", 0, 30, 0)

        if "CKD" in target_diseases:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><img src="https://img.icons8.com/fluency/32/test-tube.png"/> Nephrology Lab Panel</div>', unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            with k1:
                s_creat = st.number_input("Serum Creatinine (mg/dL)", 0.0, 15.0, 1.1)
                bun = st.number_input("Urea Nitrogen (BUN)", 1.0, 150.0, 18.0)
            with k2:
                gfr = st.number_input("Estimated GFR", 0.0, 200.0, 95.0)
                hbA1c = st.number_input("HbA1c Concentration (%)", 3.0, 15.0, 5.4)
            with k3:
                u_prot = st.select_slider("Proteinuria Level", options=[0, 1, 2, 3, 4, 5])
                fast_bs = st.number_input("Fasting Glucose", 50, 400, 100)

        submit = st.form_submit_button("PROCESS DIAGNOSTICS")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit:
        if not p_name:
            st.error("⚠️ Patient name is required for clinical reporting.")
        else:
            sex_num = 1 if sex == "Male" else 0
            age_mapped = min(13, max(1, (age - 18) // 5 + 1))
            st.session_state['scan_run'] = True
            st.session_state['selected_targets'] = target_diseases
            st.session_state['patient_name'] = p_name
            st.session_state['age'] = age
            st.session_state['sex'] = sex
            
            base_features = [int(high_bp), int(high_chol), 1, bmi, int(smoker), int(stroke), 0, int(phys_act), 1, 1, int(hvy_alc), int(healthcare), 0, gen_hlth, ment_days, phys_days, int(diff_walk), sex_num, age_mapped, 5, 6]
            base_arr = np.array(base_features).reshape(1, -1)
            
            report_results = {}
            if "Diabetes" in target_diseases and d_model:
                prob = d_model.predict_proba(d_scale.transform(base_arr))[0][1]
                st.session_state['d_res'] = prob
                report_results["Diabetes"] = prob
            if "Heart Disease" in target_diseases and h_model:
                prob = h_model.predict_proba(h_scale.transform(base_arr))[0][1]
                st.session_state['h_res'] = prob
                report_results["Heart Disease"] = prob
            if "CKD" in target_diseases and k_model:
                ckd_features = [age, bmi, int(smoker), int(hvy_alc), int(phys_act), 3, 3, 0, int(high_bp), 0, 0, 120, 80, fast_bs, hbA1c, s_creat, bun, gfr, u_prot, 1, 85]
                ckd_arr = np.array(ckd_features).reshape(1, -1)
                prob = k_model.predict_proba(k_scale.transform(ckd_arr))[0][1]
                st.session_state['k_res'] = prob
                report_results["CKD"] = prob

            st.session_state['report_results'] = report_results
            pdf_bytes = generate_clinical_pdf(p_name, age, sex, report_results)
            st.session_state['pdf_report_bytes'] = pdf_bytes
            log_patient_data(p_name, age, sex, report_results)

    if st.session_state.get('scan_run'):
        st.markdown(f'<div class="section-header"><img src="data:image/png;base64,{img_rep}" width="32"/> Clinical Output</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="clinical-card" style="text-align: center;">', unsafe_allow_html=True)
        
        # 1. Retrieve the PDF data from session state
        report_data = st.session_state.get('pdf_report_bytes')
        patient_name = st.session_state.get('patient_name', 'Patient')
        
        if report_data:
            st.subheader(f"Final Clinical Report: {patient_name}")
            st.info("Clinical assessment finalized. Click the button below to save the document.")
            
            # 2. Fix: Ensure key is unique and data is persistent
            st.download_button(
                label="📥 DOWNLOAD CLINICAL ASSESSMENT REPORT (PDF)",
                data=report_data,
                file_name=f"OmniCare_Report_{patient_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                key="persistent_download_btn",
                use_container_width=True
            )
            st.success("The report is encrypted and ready for local storage.")
        else:
            # 3. Fallback: Re-generate if lost (Safety Net)
            st.warning("Report data was refreshed. Please click 'PROCESS DIAGNOSTICS' again to re-bind the PDF.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- ASSESSMENT FINDINGS ---
        st.markdown(f'<div class="section-header"><img src="data:image/png;base64,{img_find}" width="32"/> Assessment Findings</div>', unsafe_allow_html=True)
        targets = st.session_state.get('selected_targets', [])
        if targets:
            cols = st.columns(len(targets))
            for i, disease in enumerate(targets):
                with cols[i]:
                    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
                    prob = st.session_state.get('d_res' if disease=="Diabetes" else 'h_res' if disease=="Heart Disease" else 'k_res', 0)
                    prob_pct = prob * 100
                    color = "#059669" if prob_pct <= 30 else "#d97706" if prob_pct <= 80 else "#dc2626"
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob_pct,
                        number={'suffix': "%", 'font': {'size': 35}},
                        title={'text': f"{disease} Risk", 'font': {'size': 16, 'color': '#475569'}},
                        gauge={'bar': {'color': color}, 'axis': {'range': [0, 100]}, 'bgcolor': "#f1f5f9", 'borderwidth': 0}
                    ))
                    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. ANALYTICS ---
with tab2:
    if st.session_state.get('scan_run'):
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True) 
        choice = st.selectbox("Select Diagnosis to Audit Feature Weights", st.session_state.get('selected_targets', []))
        if choice == "CKD":
            active_model, feat_names = k_model, ['Age', 'BMI', 'Smoking', 'Alcohol', 'Activity', 'Diet', 'Sleep', 'FamilyHist', 'HTN', 'Diabetes', 'UTI', 'SysBP', 'DiaBP', 'FastingBS', 'HbA1c', 'Creatinine', 'BUN', 'GFR', 'ProteinUrine', 'Fatigue', 'LifeQuality']
        else:
            active_model = d_model if choice == "Diabetes" else h_model
            feat_names = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDisease', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcohol', 'Healthcare', 'NoDocCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

        if active_model:
            df_imp = pd.DataFrame({'Factor': feat_names, 'Impact': active_model.feature_importances_}).sort_values('Impact', ascending=True).tail(10)
            fig_bar = go.Figure(go.Bar(x=df_imp['Impact'], y=df_imp['Factor'], orientation='h', marker_color='#10b981'))
            fig_bar.update_layout(title=f"Predictive Insights: {choice}", height=450, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run an assessment to view clinical evidence analysis.")

# --- 8. PATIENT HISTORY LOG TAB ---
with tab3:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    if os.path.exists(LOG_FILE):
        history_df = pd.read_csv(LOG_FILE)
        
        # Use data_editor to allow row deletion
        edited_df = st.data_editor(
            history_df, 
            num_rows="dynamic", 
            use_container_width=True,
            key="log_editor"
        )
        
        # Button to save changes after deleting rows
        if st.button("Save Changes to Log"):
            edited_df.to_csv(LOG_FILE, index=False)
            st.success("Log updated!")
            st.rerun()
    else:
        st.info("No records to display.")
    st.markdown('</div>', unsafe_allow_html=True)

# FINAL FOOTER
st.markdown('<div class="disclaimer-footer">🛡️ <strong>OmniCare CDSS</strong> | Clinical Intelligence Platform v4.5.2<br>Educational Purpose Only © 2026</div>', unsafe_allow_html=True)
