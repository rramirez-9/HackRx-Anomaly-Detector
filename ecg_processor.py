from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")  # Save figures even on machines without a display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, spectrogram
import wfdb

def load_lead_i(patient_id, base_path):
    """Return Lead I in physical units, sample rate, unit, and ECG ID."""
    base_path = Path(base_path).expanduser()
    database = pd.read_csv(base_path / "ptbxl_database.csv")
    required = {"patient_id", "ecg_id", "filename_lr"}
    missing = required.difference(database.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    records = database.loc[database["patient_id"] == patient_id]
    if records.empty:
        raise ValueError(f"No ECG found for patient_id={patient_id}")
    row = records.sort_values("ecg_id").iloc[0]
    if pd.isna(row["filename_lr"]):
        raise ValueError("Selected record has no filename_lr path")

    record_path = base_path / str(row["filename_lr"])
    record = wfdb.rdrecord(str(record_path), physical=True)
    if record.p_signal is None:
        raise ValueError("The selected record has no physical waveform data")
    names = [name.strip().upper() for name in record.sig_name]
    if "I" not in names:
        raise ValueError(f"Lead I is missing; available leads: {record.sig_name}")
    lead_index = names.index("I")
    lead_i = np.asarray(record.p_signal[:, lead_index], dtype=float)
    if not np.isfinite(lead_i).all():
        raise ValueError("Lead I contains missing or nonfinite samples")
    fs = float(record.fs)
    if not np.isfinite(fs) or not np.isclose(fs, 100.0):
        raise ValueError(f"Expected 100 Hz filename_lr data; received {fs} Hz")
    unit = record.units[lead_index] if record.units else "physical units"
    return lead_i, fs, unit, int(row["ecg_id"])

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """Apply a zero-phase Butterworth bandpass in forward and reverse time."""
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("Cutoffs must satisfy 0 < lowcut < highcut < fs/2")
    b, a = butter(order, [lowcut, highcut], btype="bandpass", fs=fs)
    padlen = 3 * max(len(a), len(b))
    if len(signal) <= padlen:
        raise ValueError(f"Filtering requires more than {padlen} samples")
    return filtfilt(b, a, signal, padlen=padlen)

def process_ecg(patient_id, base_path):
    """Process one patient and save ecg_analysis.png in the working directory."""
    original, fs, unit, ecg_id = load_lead_i(patient_id, base_path)
    filtered = bandpass_filter(original, fs)
    time = np.arange(original.size) / fs

    nperseg = min(int(2 * fs), filtered.size)
    frequencies, segment_times, power = spectrogram(
        filtered,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=int(0.75 * nperseg),
        detrend=False,
        scaling="density",
        mode="psd",
    )
    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
    visible = (frequencies >= 0.5) & (frequencies <= 40)

    # ADDED: Plotting and saving logic
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # 1. Original Waveform
    axes[0].plot(time, original, color='gray')
    axes[0].set_title(f"Patient {patient_id} (ECG {ecg_id}) - Original Lead I")
    axes[0].set_ylabel(unit)
    
    # 2. Filtered Waveform
    axes[1].plot(time, filtered, color='blue')
    axes[1].set_title("Filtered Signal (0.5 - 40 Hz)")
    axes[1].set_ylabel(unit)
    
    # 3. Spectrogram
    im = axes[2].pcolormesh(segment_times, frequencies[visible], power_db[visible, :], 
                            shading='gouraud', cmap='viridis')
    axes[2].set_title("Spectrogram (Hidden Frequencies)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes[2], label="Power (dB)")

    plt.tight_layout()
    save_path = Path("ecg_analysis.png").resolve()
    plt.savefig(save_path)
    plt.close()
    print(f"Success! Image saved to {save_path}")
    return save_path

# ADDED: Execution block to run from the terminal
if __name__ == "__main__":
    import sys
    # Defaults to patient 1 and current directory if arguments aren't passed
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    path = sys.argv[2] if len(sys.argv) > 2 else '.'
    process_ecg(pid, path)