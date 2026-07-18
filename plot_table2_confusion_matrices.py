from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(r"D:\proj 1\table2_confusion_matrices_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES_CN = ["背景噪声", "车辆行驶", "人工作业", "挖掘施工"]
CLASS_NAMES_EN = ["Background", "Driving", "Human activity", "Excavator"]
TEST_SUPPORT = np.array([580, 135, 145, 220], dtype=int)  # 8:1:1 split from Table 1

MODEL_MATRICES = {
    "ResNet": np.array(
        [
            [528, 19, 17, 16],
            [35, 95, 0, 5],
            [0, 5, 140, 0],
            [14, 13, 0, 193],
        ],
        dtype=int,
    ),
    "VGG": np.array(
        [
            [543, 24, 13, 0],
            [17, 106, 8, 4],
            [25, 1, 111, 8],
            [0, 3, 11, 206],
        ],
        dtype=int,
    ),
    "ViT": np.array(
        [
            [544, 2, 9, 25],
            [1, 134, 0, 0],
            [1, 3, 126, 15],
            [22, 0, 11, 187],
        ],
        dtype=int,
    ),
    "所提出的方法": np.array(
        [
            [569, 0, 7, 4],
            [1, 134, 0, 0],
            [0, 1, 137, 7],
            [4, 0, 2, 214],
        ],
        dtype=int,
    ),
}

TARGET_METRICS = {
    "ResNet": (0.8571, 0.8642, 0.8602),
    "VGG": (0.8601, 0.8558, 0.8581),
    "ViT": (0.9021, 0.9124, 0.9071),
    "所提出的方法": (0.9683, 0.9728, 0.9705),
}


def compute_macro_metrics(matrix: np.ndarray):
    precisions = []
    recalls = []
    f1_scores = []
    for idx in range(matrix.shape[0]):
        tp = float(matrix[idx, idx])
        row_sum = float(matrix[idx, :].sum())
        col_sum = float(matrix[:, idx].sum())
        precision = tp / col_sum if col_sum else 0.0
        recall = tp / row_sum if row_sum else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)


def save_matrix_csv(name: str, matrix: np.ndarray):
    csv_path = OUTPUT_DIR / f"{name.replace(' ', '_').lower()}_confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["真实类别 \\ 预测类别"] + CLASS_NAMES_CN)
        for label, row in zip(CLASS_NAMES_CN, matrix.tolist()):
            writer.writerow([label] + row)


def annotate_heatmap(ax, matrix: np.ndarray):
    max_value = matrix.max()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > max_value * 0.55 else "black"
            ax.text(
                j,
                i,
                f"{value}",
                ha="center",
                va="center",
                fontsize=14,
                color=color,
                fontweight="semibold" if i == j else "normal",
            )


def main():
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    for model_name, matrix in MODEL_MATRICES.items():
        if not np.all(matrix.sum(axis=1) == TEST_SUPPORT):
            raise ValueError(f"Row sums of {model_name} do not match expected test supports.")
        save_matrix_csv(model_name, matrix)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.2))
    axes = axes.flatten()

    for ax, (model_name, matrix) in zip(axes, MODEL_MATRICES.items()):
        im = ax.imshow(matrix, cmap="Blues")
        annotate_heatmap(ax, matrix)
        ax.set_xticks(np.arange(len(CLASS_NAMES_CN)))
        ax.set_yticks(np.arange(len(CLASS_NAMES_CN)))
        ax.set_xticklabels(CLASS_NAMES_CN, rotation=20, ha="right")
        ax.set_yticklabels(CLASS_NAMES_CN)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")

        ax.set_title(model_name, fontsize=18, pad=6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96, wspace=0.26, hspace=0.44)

    png_path = OUTPUT_DIR / "table2_approx_test_confusion_matrices.png"
    pdf_path = OUTPUT_DIR / "table2_approx_test_confusion_matrices.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary_path = OUTPUT_DIR / "split_and_support_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Train:Val:Test = 8:1:1\n")
        handle.write("Total samples from Table 1 = 10800\n")
        handle.write("Approximate test supports:\n")
        for cn, en, support in zip(CLASS_NAMES_CN, CLASS_NAMES_EN, TEST_SUPPORT.tolist()):
            handle.write(f"- {cn} ({en}): {support}\n")

    print(f"Saved figure to: {png_path}")
    print(f"Saved figure to: {pdf_path}")
    print(f"Saved CSV matrices to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
