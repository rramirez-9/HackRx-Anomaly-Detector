from pathlib import Path

from datasets import load_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure


DATA_URL = (
    "https://huggingface.co/datasets/alkzar90/"
    "NIH-Chest-X-ray-dataset/resolve/main/data"
)


def load_first_xray():
    """Stream the first training-list image via a seekable remote ZIP member.

    The first training-list entry is in images_001.zip. Selecting one explicit
    member avoids scanning/decompressing the other images. The legacy loader's
    archive iteration order may differ from this training-list order.
    """
    filenames = load_dataset(
        "text", data_files={"train": f"{DATA_URL}/train_val_list.txt"},
        split="train", streaming=True,
    )
    first = next(iter(filenames), None)
    if first is None:
        raise ValueError("The training image list is empty")
    filename = first["text"].strip()
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Unexpected training image filename: {filename!r}")

    image_url = (
        f"zip://images/{filename}::{DATA_URL}/images/images_001.zip"
    )
    subset = load_dataset(
        "imagefolder", data_files={"train": image_url},
        split="train", streaming=True,
    )
    example = next(iter(subset), None)
    if example is None:
        raise ValueError("The image stream is empty")
    return example["image"]


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


def process_xray():
    """Stream one training image, enhance it, and save a PNG.

    CLAHE changes local image contrast throughout the image; it does not
    segment the lungs or recover detail absent from the source resolution.
    Returns the absolute path to xray_analysis.png in the working directory.
    """
    original, enhanced = enhance_xray(load_first_xray())
    return save_comparison(original, enhanced)


if __name__ == "__main__":
    print(f"Saved: {process_xray()}")