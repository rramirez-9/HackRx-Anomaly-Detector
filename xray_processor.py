"""Stream a selected chest X-ray and save its CLAHE comparison.

Install: pip install datasets scikit-image numpy matplotlib
Source: https://huggingface.co/datasets/keremberke/chest-xray-classification
License: CC BY 4.0. Original dataset: Mohamed Traore / Roboflow,
https://universe.roboflow.com/mohamed-traore-2ekkp/chest-x-rays-qjmia/dataset/2

The Hub's data-only Parquet export of the full training split avoids executing
this repository's legacy Python loading script. Streaming reads the requested
row plus metadata/buffered data, without preparing the full dataset locally.
"""

from pathlib import Path

from datasets import load_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


DATASET_ID = "keremberke/chest-xray-classification"
TRAIN_PARQUET_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
    "refs%2Fconvert%2Fparquet/full/train/0000.parquet"
)
SAMPLE_INDICES = {
    "Sample 1": 0,
    "Sample 2": 15,
    "Sample 3": 30,
    "Sample 4": 45,
    "Sample 5": 60,
}


def load_xray(sample_name="Sample 1"):
    """Return the decoded image at a zero-based row in the full train split."""
    if sample_name not in SAMPLE_INDICES:
        raise ValueError(
            f"Unknown sample {sample_name!r}; choose one of {list(SAMPLE_INDICES)}"
        )
    row_index = SAMPLE_INDICES[sample_name]
    stream = load_dataset(
        "parquet", data_files={"train": TRAIN_PARQUET_URL},
        split="train", streaming=True,
    )
    example = next(iter(stream.skip(row_index).take(1)), None)
    if example is None:
        raise ValueError(f"Training row {row_index} is unavailable for {sample_name}")
    return example["image"]


def load_first_xray():
    """Backward-compatible helper returning Sample 1."""
    return load_xray("Sample 1")


def enhance_xray(image):
    """Convert a decoded PIL image to grayscale and apply local CLAHE."""
    original = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    enhanced = exposure.equalize_adapthist(
        original, clip_limit=0.03, nbins=256,
    )
    return original, enhanced


def save_comparison(original, enhanced, output_path="xray_analysis.png"):
    """Save two panels with the same grayscale display range."""
    output_path = Path(output_path).resolve()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    try:
        for ax, array, title in zip(
            axes, (original, enhanced), ("Original X-ray", "CLAHE-enhanced X-ray"),
        ):
            ax.imshow(array, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)
    return output_path


def process_xray(sample_name="Sample 1"):
    """Stream the selected training row, enhance it, and save a PNG.

    CLAHE changes local image contrast throughout the image; it does not
    segment the lungs or recover detail absent from the source resolution.
    Returns the absolute path to xray_analysis.png in the working directory.
    """
    original, enhanced = enhance_xray(load_xray(sample_name))
    return save_comparison(original, enhanced)


if __name__ == "__main__":
    print(f"Saved: {process_xray()}")
