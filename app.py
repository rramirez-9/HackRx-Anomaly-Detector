"""Streamlit demonstration combining the local ECG and X-ray processors.

Run from the PTB-XL dataset root (where ptbxl_database.csv is located):
    streamlit run /absolute/path/to/outputs/app.py --theme.base=dark

Keep app.py, ecg_processor.py, and xray_processor.py together. The processors
save PNGs in the working directory; the app keeps a byte snapshot per session.
"""

from pathlib import Path
from threading import Lock
from hashlib import sha256

import streamlit as st

import ecg_processor
import xray_processor


@st.cache_resource
def processor_lock():
    """Serialize plotting and reads because both modules use fixed filenames."""
    return Lock()


def run_engine(ecg_id, sample_name):
    """Run both processors and copy image bytes before another run overwrites them."""
    result = {"images": {}, "errors": {}}
    with processor_lock():
        for modality, process in (
            ("ECG", lambda: ecg_processor.process_ecg(ecg_id, '.')),
            ("X-ray", lambda: xray_processor.process_xray(sample_name)),
        ):
            try:
                image_path = Path(process())
                image_data = image_path.read_bytes()
                if not image_data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("Processor output is not a PNG image")
                result["images"][modality] = image_data
            except Exception as exc:
                # A failed modality must not hide the other modality's output.
                result["errors"][modality] = f"{type(exc).__name__}: {exc}"
    return result


def generate_clinical_summary(age, sex, bmi, hr, sys_bp, dia_bp, o2, triage, complaint, ecg_id, xray_sample):
    # 1. ECG Analysis (Demo Magic mapping IDs to specific findings)
    if ecg_id == 15777:
        ecg_finding = "Atrial Fibrillation (AFib). Spectrogram shows absent P-waves and irregular R-R intervals."
        ecg_risk = "High"
    elif ecg_id == 15778:
        ecg_finding = "ST-Elevation suggestive of Myocardial Infarction (STEMI). High-frequency anomaly detected in ST segment."
        ecg_risk = "Critical"
    elif ecg_id == 15779:
        ecg_finding = "Premature Ventricular Contractions (PVCs) noted. Otherwise baseline stable."
        ecg_risk = "Moderate"
    elif hr > 100:
        ecg_finding = f"Sinus Tachycardia (HR {hr} bpm). Spectrogram baseline shifted but uniform."
        ecg_risk = "Moderate"
    elif hr < 60:
        ecg_finding = f"Sinus Bradycardia (HR {hr} bpm)."
        ecg_risk = "Low"
    else:
        ecg_finding = "Normal sinus rhythm. Spectrogram baseline stable."
        ecg_risk = "Low"

    # 2. X-Ray Analysis (Mapping Samples to specific findings)
    if xray_sample == "Sample 2":
        xray_finding = "CLAHE enhancement highlights bilateral basal opacities, suggestive of pulmonary edema."
        xray_risk = "High"
    elif xray_sample == "Sample 3":
        xray_finding = "Enhanced contrast reveals blunting of the costophrenic angle (possible pleural effusion)."
        xray_risk = "Moderate"
    elif xray_sample == "Sample 4":
        xray_finding = "Enlarged cardiac silhouette detected (Cardiomegaly)."
        xray_risk = "High"
    else:
        xray_finding = "CLAHE reveals clear lung fields. No acute cardiopulmonary process identified."
        xray_risk = "Low"

    # 3. Multimodal Synthesis (Combines Demographics, Vitals, ECG, and X-ray)
    synthesis = f"Patient is a {age}yo {sex} presenting with: {complaint}. "
    
    if ecg_risk == "Critical":
        synthesis += "CRITICAL CARDIAC EVENT: ECG findings indicate acute myocardial infarction. Immediate cardiology consult required. "
        risk_level = "CRITICAL"
    elif ecg_risk == "High" and xray_risk == "High":
        synthesis += "CRITICAL PRESENTATION: Concurrent cardiac arrhythmia (AFib) and pulmonary edema strongly suggests acute decompensated heart failure. "
        risk_level = "CRITICAL"
    elif xray_risk == "High" and o2 < 94:
        synthesis += f"HIGH RISK RESPIRATORY: Pulmonary opacities combined with hypoxia (O2 at {o2}%) suggests severe pneumonia or fluid overload. "
        risk_level = "HIGH"
    elif ecg_risk == "Moderate" or xray_risk == "Moderate" or sys_bp > 160:
        synthesis += "MODERATE RISK: Abnormalities detected in single modality or elevated vitals. Requires continuous monitoring and lab correlation. "
        risk_level = "MODERATE"
    else:
        synthesis += "LOW RISK: Vital signs and enhanced imaging do not indicate acute cardiopulmonary failure. Standard triage protocols apply. "
        risk_level = "LOW"
        
    return {
        "ecg_finding": ecg_finding,
        "xray_finding": xray_finding,
        "synthesis": synthesis,
        "risk_level": risk_level
    }

def main():
    st.set_page_config(
        page_title="Sub-Clinical Anomaly Detector",
        page_icon="🩺", layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(
        """<style>
        .stApp { background-color: #0b1220; color: #e6edf5; }
        [data-testid="stSidebar"] { background-color: #111c2e; color: #e6edf5; }
        [data-testid="stMetric"] {
            background-color: #152238; padding: 1.1rem;
            border: 1px solid #2b405b; border-radius: 12px; color: #e6edf5;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.title("Mix & Match Console")
        with st.expander("Data Sources", expanded=True):
            ecg_id = st.selectbox("Select ECG Record", range(15776, 15786))
            xray_sample = st.selectbox(
                "Select Chest X-Ray",
                ["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"],
            )
        with st.expander("Patient Demographics & Vitals", expanded=True):
            age = st.number_input("Age", min_value=18, max_value=100, value=65)
            sex = st.selectbox("Biological Sex", ["Male", "Female"])
            weight = st.number_input("Weight (lbs)", min_value=80, max_value=400, value=180)
            height = st.number_input("Height (inches)", min_value=36, max_value=96, value=68)
            bmi = (weight / (height ** 2)) * 703
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=85)
            systolic = st.number_input("Systolic BP", value=120, step=1)
            diastolic = st.number_input("Diastolic BP", value=80, step=1)
            oxygen = st.number_input("O2 Saturation (%)", min_value=70, max_value=100, value=98)
            triage = st.selectbox("Triage Level", ["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"], index=2)
        with st.expander("Clinical Context", expanded=True):
            clinical_history = st.text_area(
                "Chief Complaint & Past Medical History",
                placeholder=(
                    "e.g., 65yo male presenting with shortness of breath. "
                    "History of hypertension and smoking."
                ),
                key="clinical_history",
            )
            lab_report = st.file_uploader(
                "Upload External Lab Report (Optional)",
                type=["txt", "csv", "pdf"], key="lab_report",
            )
            st.caption("History is included in the rule-based summary; attachments are not analyzed.")
        st.caption("Demonstration workspace · synthetic triage data")
        st.divider()
        st.caption(
            "ECG source: local PTB-XL. X-ray source: keremberke/chest-xray-classification. "
            "These records are unpaired; the X-ray is not matched to the ECG subject."
        )

    selection = {
        "ecg_id": ecg_id, "xray_sample": xray_sample,
        "age": age, "sex": sex, "weight": weight,
        "height": height, "bmi": bmi, "clinical_history": clinical_history,
        "lab_report": (
            {"name": lab_report.name, "sha256": sha256(lab_report.getvalue()).hexdigest()}
            if lab_report is not None else None
        ),
        "heart_rate": heart_rate, "systolic": systolic, "diastolic": diastolic,
        "oxygen": oxygen, "triage": triage,
    }
    st.title("Sub-Clinical Anomaly Detector")
    st.caption("MULTIMODAL REVIEW · ECG + CHEST X-RAY · RESEARCH DEMONSTRATION")
    st.info(
        "Synthetic demonstration: summary text follows preset rules applied to "
        "entered vitals and sample names. ECG and X-ray findings are scripted; "
        "the processors visualize data and do not detect disease."
    )
    st.subheader("Patient Demographics")
    st.caption("Illustrative values only; no connection to Yale EM patient records.")
    for column, label, value in zip(
        st.columns(5),
        ("Age", "Biological Sex", "Height", "Weight", "BMI"),
        (f"{age} years", sex, f"{height} in", f"{weight} lbs", f"{bmi:.1f}"),
    ):
        column.metric(label, value)

    st.subheader("Custom Mock Yale EM Triage Vitals")
    for column, label, value in zip(
        st.columns(4),
        ("Heart Rate", "Blood Pressure", "O2 Saturation", "Triage Level"),
        (f"{heart_rate} bpm", f"{systolic}/{diastolic} mmHg", f"{oxygen}%", triage),
    ):
        column.metric(label, value)

    if st.button("Run Diagnostic Engine", type="primary"):
        with st.spinner("Processing ECG Lead I and enhancing the chest X-ray…"):
            result = run_engine(ecg_id, xray_sample)
            result["selection"] = selection.copy()
            st.session_state["last_analysis"] = result

    result = st.session_state.get("last_analysis")
    if result is None:
        st.caption("Choose records and vitals, then run the engine to view the analyses.")
        return

    if result["selection"] != selection:
        st.info("Inputs have changed. Images below are from the last run; run the engine to update them.")
    st.divider()
    for column, modality in zip(st.columns(2), ("ECG", "X-ray")):
        with column:
            st.subheader(f"{modality} Analysis")
            if modality in result["images"]:
                st.image(result["images"][modality], use_container_width=True)
                st.caption(
                    f"PTB-XL patient_id {result['selection']['ecg_id']} · Lead I"
                    if modality == "ECG"
                    else f"{result['selection']['xray_sample']} · chest X-ray dataset · unpaired with ECG"
                )
            else:
                st.error(f"{modality} processing failed.")
                with st.expander("Error details"):
                    st.code(result["errors"][modality], language=None)
                if modality == "ECG":
                    st.caption(
                        "Run Streamlit from the PTB-XL dataset root, containing "
                        "ptbxl_database.csv and records100/ with .hea and .dat files."
                    )
                else:
                    st.caption("The X-ray stream requires internet access and HTTP range support.")

    summary = generate_clinical_summary(
        age, sex, bmi, heart_rate, systolic, diastolic, oxygen, triage,
        clinical_history, ecg_id, xray_sample,
    )
    st.divider()
    st.subheader("Clinical Summary · Rule-Based Demonstration")
    st.caption(
        "Uses the current sidebar inputs and selected sample name, not image "
        "interpretation. High/Low is a demo rule classification, not a validated "
        "clinical risk estimate; Low does not establish clinical stability."
    )
    if result["errors"]:
        st.warning("Some images are unavailable. The summary below is generated only from sidebar inputs.")
    finding_columns = st.columns(2)
    with finding_columns[0]:
        st.info(f"**ECG Finding · Scripted**\n\n{summary['ecg_finding']}")
    with finding_columns[1]:
        st.info(f"**X-Ray Finding · Scripted**\n\n{summary['xray_finding']}")
    if summary["risk_level"] == "High":
        st.error(f"**High Risk · Demo Rules**\n\n{summary['synthesis']}")
    else:
        st.success(f"**Low Risk · Demo Rules**\n\n{summary['synthesis']}")


if __name__ == "__main__":
    main()
