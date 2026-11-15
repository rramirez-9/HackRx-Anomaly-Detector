from pathlib import Path
from threading import Lock

import streamlit as st

import ecg_processor
import xray_processor


PROFILES = {
    "Patient A": {
        "patient_id": 15776,
        "heart_rate": "104 bpm",
        "blood_pressure": "142/88 mmHg",
        "oxygen": "94%",
        "triage": "ESI 2",
        "risk": "High Risk — simulated scenario",
        "prediction": (
            "Illustrative scenario: sub-clinical frequency anomalies in ECG "
            "Lead I alongside early pulmonary opacities. These findings are "
            "scripted for this profile; they are not detected from the images."
        ),
    },
    "Patient B": {
        "patient_id": 15777,
        "heart_rate": "76 bpm",
        "blood_pressure": "118/76 mmHg",
        "oxygen": "98%",
        "triage": "ESI 4",
        "risk": "Lower Risk — simulated scenario",
        "prediction": (
            "Illustrative scenario: no additional anomaly flags assigned to "
            "this synthetic profile. This scripted result does not establish "
            "that the ECG or X-ray is normal."
        ),
    },
}


@st.cache_resource
def processor_lock():
    """Serialize plotting and reads because both modules use fixed filenames."""
    return Lock()


def run_engine(patient_id):
    """Run both processors and copy image bytes before another run overwrites them."""
    result = {"images": {}, "errors": {}}
    with processor_lock():
        for modality, process in (
            ("ECG", lambda: ecg_processor.process_ecg(patient_id, '.')),
            ("X-ray", xray_processor.process_xray),
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
        st.title("Patient Console")
        selected = st.selectbox("Synthetic Patient Profiles", list(PROFILES))
        st.caption("Demonstration workspace · synthetic triage data")
        st.divider()
        st.caption(
            "ECG source: local PTB-XL. X-ray source: streamed NIH example. "
            "These records are unpaired; the same X-ray is used for both profiles."
        )

    profile = PROFILES[selected]
    st.title("Sub-Clinical Anomaly Detector")
    st.caption("MULTIMODAL REVIEW · ECG + CHEST X-RAY · RESEARCH DEMONSTRATION")
    st.info(
        "Synthetic demonstration: the risk text is predefined. The processors "
        "filter and visualize data; they do not detect disease or predict risk."
    )
    st.subheader(f"{selected} · Mock Yale EM triage vitals")
    st.caption("Illustrative values only; no connection to Yale EM patient records.")
    for column, label, value in zip(
        st.columns(4),
        ("Heart Rate", "Blood Pressure", "O2 Saturation", "Triage Level"),
        (profile["heart_rate"], profile["blood_pressure"], profile["oxygen"], profile["triage"]),
    ):
        column.metric(label, value)

    if st.button("Run Diagnostic Engine", type="primary"):
        with st.spinner("Processing ECG Lead I and enhancing the chest X-ray…"):
            result = run_engine(profile["patient_id"])
            st.session_state.setdefault("patient_results", {})[selected] = result

    result = st.session_state.get("patient_results", {}).get(selected)
    if result is None:
        st.caption("Select a profile and run the engine to view its analyses.")
        return

    st.divider()
    for column, modality in zip(st.columns(2), ("ECG", "X-ray")):
        with column:
            st.subheader(f"{modality} Analysis")
            if modality in result["images"]:
                st.image(result["images"][modality], use_container_width=True)
                st.caption(
                    f"PTB-XL patient_id {profile['patient_id']} · Lead I"
                    if modality == "ECG"
                    else "NIH sample · original and CLAHE-enhanced · unpaired with ECG"
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

    st.subheader("Clinical Risk Prediction")
    if result["errors"]:
        st.warning("Analysis incomplete. The simulated risk output is withheld until both processors succeed.")
    else:
        st.warning(profile["risk"])
        st.write(profile["prediction"])
        st.caption("Simulated output · not a clinical assessment or a cross-modal correlation.")


if __name__ == "__main__":
    main()