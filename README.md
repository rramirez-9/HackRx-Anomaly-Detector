# Multi-Modal Clinical Triage Dashboard

**A Deterministic Clinical Decision Support System (CDSS) for the Emergency Department**

![UI Preview](https://img.shields.io/badge/UI-Streamlit-FF4B4B) ![Compute](https://img.shields.io/badge/Compute-Local_CPU-blue) ![Math](https://img.shields.io/badge/Math-Deterministic-brightgreen)

## The Clinical Problem
In a fast-paced Emergency Room, critical patient data is siloed. Vitals live on a tablet, ECGs are printed on carts, and chest X-rays require dedicated radiology monitors. Furthermore, modern hospitals are hesitant to adopt "black-box" Generative AI that is prone to hallucination, requires heavy GPU compute, and poses HIPAA risks via third-party API calls. 

## Our Solution
We built a unified **Multi-Modal Triage Dashboard** designed specifically for the ER workflow. Instead of relying on unpredictable LLMs, our engine uses **explainable, deterministic mathematics** to extract hidden physiological features, instantly cross-referencing them against a rule-based expert system to provide physicians with a highly accurate, zero-hallucination diagnostic synthesis.

---

## Core Clinical Features

### 1. Acoustic/Electrical Signal Processing (ECG)
* **The Math:** Fast Fourier Transforms (FFT).
* **The Impact:** Translates raw digital time-series ECG data (PTB-XL dataset) into high-resolution frequency spectrograms. This exposes microscopic electrical flutters and anomalies (like AFib or STEMI markers) that are completely invisible on a standard time-domain line graph.

### 2. Radiological Computer Vision (Chest X-Ray)
* **The Math:** Contrast Limited Adaptive Histogram Equalization (CLAHE).
* **The Impact:** Standard X-rays often hide early-stage fluid or tissue abnormalities behind dense bone structures. By mathematically recalculating localized pixel grids, our engine equalizes lighting contrast to reveal hidden pulmonary opacities without altering the underlying anatomy.

### 3. Cross-Modal Synthesis Engine (CDSS)
* **The Math:** Clinical Rule-Based Expert System.
* **The Impact:** Simulates advanced clinical reasoning by cross-referencing the patient's demographics (Age, BMI), live vitals (HR, BP, SpO2), and mathematical imaging enhancements. It instantly generates a categorized Risk Level and multi-modal synthesis (e.g., flagging the dangerous combination of high heart rate and pulmonary edema) while keeping the human doctor as the ultimate decision-maker.

---

## Why Deterministic Math Over Generative AI?
To maximize **Best Clinical Impact**, this project was architected strictly around patient safety and hospital deployment realities:
1. **Zero Hallucinations:** Rule-based logic guarantees 100% reproducibility.
2. **Edge-Compute Ready:** Optimized algorithms run locally on a standard laptop CPU in milliseconds. No $10,000 GPUs required.
3. **Absolute Privacy:** Requires zero internet connection and makes zero third-party API calls, ensuring strict HIPAA compliance.
4. **Physician Trust:** We don't replace the doctor; we provide highly explainable visual insights to accelerate human triage.

---

## Tech Stack
* **Frontend/UI:** Streamlit
* **Signal Processing:** SciPy, NumPy, Matplotlib
* **Computer Vision:** OpenCV (cv2)
* **Datasets:** PTB-XL (ECG), Hugging Face (Chest X-Ray)

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/HackRx-Anomaly-Detector.git](https://github.com/YourUsername/HackRx-Anomaly-Detector.git)
   cd HackRx-Anomaly-Detector
2. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
3. **Launch the clinical dashboard**
   ```bash
   streamlit run app.py
