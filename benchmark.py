import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import pipeline and visualizer
try:
    from neuro_symbolic import NeuroSymbolicInductor
    from viz import visualize_scene_graph
except ImportError:
    raise ImportError(
        "Please ensure 'neuro_symbolic.py' and 'viz.py' are in the current directory "
        "and importable (PYTHONPATH)."
    )

# --- Configuration ---
DATASET_ROOT = "datasets/MVTec-loco-AD"  # Point this to your dataset root
CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

OUTPUT_DIR = "results/nsad_mvtec_loco"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SAMPLES_PER_CLASS = 20  # or None for all images


# --- Helper Functions ---


def set_seed(seed: int = 0):
    """Optional: make runs more reproducible."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def get_image_paths(category: str) -> dict:
    """
    Returns dictionary of paths for train and test splits for one category.
    """
    cat_path = os.path.join(DATASET_ROOT, category)

    paths = {
        "train": glob.glob(os.path.join(cat_path, "train", "good", "*.png")),
        "test_good": glob.glob(os.path.join(cat_path, "test", "good", "*.png")),
        "test_logical": glob.glob(
            os.path.join(cat_path, "test", "logical_anomalies", "*.png")
        ),
        "test_structural": glob.glob(
            os.path.join(cat_path, "test", "structural_anomalies", "*.png")
        ),
    }
    return paths


def plot_histograms(y_true, y_scores, category, out_path):
    """Save score histograms for normal vs anomalous samples."""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    plt.figure()
    normal_scores = y_scores[y_true == 0]
    anomaly_scores = y_scores[y_true == 1]

    if len(normal_scores):
        plt.hist(
            normal_scores,
            bins=30,
            alpha=0.5,
            label="Normal",
            color="blue",
            density=True,
        )
    if len(anomaly_scores):
        plt.hist(
            anomaly_scores,
            bins=30,
            alpha=0.5,
            label="Anomaly",
            color="red",
            density=True,
        )

    plt.xlabel("Logical anomaly score (NS-AD)")
    plt.ylabel("Density")
    plt.title(f"Score distribution – {category}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(y_true, y_scores, category, auroc, out_path):
    """Save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUROC={auroc:.3f})", color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – {category}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_benchmark():
    # Optional but helpful for reproducibility of clustering / subsampling
    set_seed(0)
    rng = np.random.default_rng(0)

    results = {}
    all_records = []  # for global CSV across all categories

    print(f"[*] Starting MVTec LOCO AD Benchmark on: {DATASET_ROOT}")

    for category in CATEGORIES:
        print(f"\n=== Processing Category: {category} ===")
        paths = get_image_paths(category)

        if category == "breakfast_box":
            continue

        # Check if data exists
        if not paths["train"]:
            print(f"[!] Data not found for {category}. Skipping.")
            continue

        # 1. Initialize & Train
        nsad = NeuroSymbolicInductor()

        # Load Training Data (few shots for logic mining; adjust if needed)
        train_imgs = [load_image(p) for p in paths["train"][:16]]
        print(f"    Training on {len(train_imgs)} normal images...")
        nsad.train(train_imgs)

        # --- Quick balanced subsampling for test sets ---
        good_paths = paths["test_good"]
        log_paths = paths["test_logical"]

        if MAX_SAMPLES_PER_CLASS is not None:
            # balance between good and logical anomalies
            n_good = len(good_paths)
            n_log = len(log_paths)
            if n_good == 0 or n_log == 0:
                print("    [!] Missing good or logical images, skipping subsampling.")
            else:
                n = min(n_good, n_log, MAX_SAMPLES_PER_CLASS)
                good_paths = rng.choice(good_paths, size=n, replace=False).tolist()
                log_paths = rng.choice(log_paths, size=n, replace=False).tolist()
                print(
                    f"    Quick mode: using {n} good and {n} logical images "
                    f"(out of {n_good} / {n_log})"
                )

        # 2. Inference Loop
        y_true = []
        y_scores = []
        category_records = []  # per-category rows for CSV

        test_sets = [
            (good_paths, 0, "good"),  # Label 0 = Normal
            (log_paths, 1, "logical_anomaly"),  # Label 1 = Anomaly
            # Uncomment to benchmark structural anomalies too:
            # (paths["test_structural"], 1, "structural_anomaly"),
        ]

        for img_paths, label, subset_name in test_sets:
            if not img_paths:
                continue

            # Create Viz Directory: results/category/viz/subset_name/
            viz_dir = os.path.join(OUTPUT_DIR, category, "viz", subset_name)
            os.makedirs(viz_dir, exist_ok=True)

            desc = "Good" if label == 0 else subset_name

            for p in tqdm(img_paths, desc=f"    Testing {desc}", leave=False):
                rel_path = os.path.relpath(p, DATASET_ROOT)
                filename = os.path.basename(p)

                try:
                    img = load_image(p)
                    # Updated API: returns (graph, violations, calibrated_score)
                    graph, violations, anomaly_score = nsad.test(img, filename)

                    # --- Visualization Step ---
                    fig = visualize_scene_graph(img, graph, violations, show=False)
                    viz_save_path = os.path.join(viz_dir, filename)
                    fig.savefig(viz_save_path)
                    plt.close(fig)  # Critical: Close figure to prevent RAM blow-up

                    violations_str = (
                        "; ".join([v.description for v in violations])
                        if violations
                        else ""
                    )

                except Exception as e:
                    print(f"    [!] Inference failed for {p}: {e}")
                    # Use a large score so failures are treated as anomalous and visible
                    anomaly_score = 100.0
                    violations_str = f"INFERENCE_ERROR: {e}"

                y_true.append(label)
                y_scores.append(anomaly_score)

                record = {
                    "category": category,
                    "img_path": rel_path,
                    "subset": subset_name,
                    "label": label,
                    "anomaly_score": anomaly_score,
                    "violations": violations_str,
                }
                category_records.append(record)
                all_records.append(record)

        # 3. Save per-category CSV
        if category_records:
            df_cat = pd.DataFrame(category_records)
            csv_path = os.path.join(OUTPUT_DIR, f"{category}_predictions.csv")
            df_cat.to_csv(csv_path, index=False)
            print(f"    Saved predictions to {csv_path}")

        # 4. Calculate AUROC + plots (only if we have both classes)
        if len(set(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_scores)
            print(f"    -> I-AUROC (logical): {auroc:.4f}")
            results[category] = auroc

            # Plots
            roc_path = os.path.join(OUTPUT_DIR, f"{category}_roc.png")
            hist_path = os.path.join(OUTPUT_DIR, f"{category}_scores_hist.png")
            plot_roc_curve(y_true, y_scores, category, auroc, roc_path)
            plot_histograms(y_true, y_scores, category, hist_path)
        else:
            print(
                "    [!] Not enough data to calculate AUROC "
                "(need both normal and anomalous samples)."
            )
            results[category] = 0.0

    # --- Final Report ---
    print("\n" + "=" * 40)
    print("       FINAL BENCHMARK RESULTS       ")
    print("=" * 40)
    df = pd.DataFrame(list(results.items()), columns=["Category", "I-AUROC"])
    print(df)
    print("-" * 40)
    print(f"Mean I-AUROC: {df['I-AUROC'].mean():.4f}")
    print("=" * 40)

    if all_records:
        df_all = pd.DataFrame(all_records)
        all_csv_path = os.path.join(OUTPUT_DIR, "all_predictions.csv")
        df_all.to_csv(all_csv_path, index=False)
        print(f"[+] Saved all predictions to {all_csv_path}")


if __name__ == "__main__":
    run_benchmark()
