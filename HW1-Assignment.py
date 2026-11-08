import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity


ROOT_DIR = Path.cwd()


# initialize the directory of the images

Input_dir = Path(".\images\original")
print(Input_dir)
Degraded_dir = Path("images/degraded")
Result_dir = Path("results")

Degraded_dir.mkdir(parents=True, exist_ok=True)
Result_dir.mkdir(parents=True, exist_ok=True)

SEED = 42
ALPHA = 0.6 
NOISE_STD = 5
GAMMA = 0.5

#------------------------------
# Degradation
#------------------------------

def degrade_image(image):
    rng = np.random.default_rng(SEED)

    degraded = image.astype(np.float32) * ALPHA
    noise = rng.normal(0, NOISE_STD, image.shape)

    degraded = np.clip(
        degraded + noise, 0, 255
    )

    return degraded.astype(np.uint8)

# --------------------------------  


# -------------------------------
# Contrast stretching
# -------------------------------

def contrast_stretch(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.normalize(
        l, None, 0, 255, cv2.NORM_MINMAX
    )
    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )

# -------------------------------
# Histogram equalization
# -------------------------------


def histogram_equalization(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.equalizeHist(l)

    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )


# -------------------------------
# Gamma correction
# -------------------------------

def gamma_correction(image, gamma=GAMMA):
    table = np.array([
        255 * (i / 255.0) ** gamma
        for i in range(256)
    ], dtype=np.uint8)

    return cv2.LUT(image, table)


#-------------------------------
# Save comparison
#-------------------------------   


def save_comparison(images, titles, image_name):
    plt.figure(figsize=(18, 5))

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image_rgb)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()

    # Save directly in project root
    output_path = ROOT_DIR / f"{image_name}_comparison.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison saved: {output_path}")

# -------------------------------
# Main pipeline
# -------------------------------

metrics = []

for image_path in Input_dir.glob("*"):
    original = cv2.imread(str(image_path))

    if original is None:
        continue
    

    name = image_path.stem

    degraded = degrade_image(original)

    cv2.imwrite(
        str(Degraded_dir / f"{name}.png"),
        degraded
    )

    methods = {
        "Baseline": degraded,
        "Contrast Stretching": contrast_stretch(degraded),
        "Histogram Equalization": histogram_equalization(degraded),
        "Gamma Correction": gamma_correction(degraded)
    }

    for method_name, result in methods.items():

        method_dir = Result_dir / method_name
        method_dir.mkdir(exist_ok=True)

        cv2.imwrite(
            str(method_dir / f"{name}.png"),
            result
        )

        psnr = cv2.PSNR(original, result)

        ssim = structural_similarity(
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
            channel_axis=2,
            data_range=255
        )

        metrics.append({
            "Image": name,
            "Method": method_name,
            "PSNR": psnr,
            "SSIM": ssim
        })
    save_comparison(
    [
        original,
        degraded,
        contrast_stretch(degraded),
        histogram_equalization(degraded),
        gamma_correction(degraded)
    ],
    [
        "Original",
        "Degraded",
        "Contrast Stretching",
        "Histogram Equalization",
        "Gamma Correction"
    ],
    name
)


df = pd.DataFrame(metrics)
df.to_csv("metrics.csv", index=False)


print(df)
print("Pipeline completed!")



