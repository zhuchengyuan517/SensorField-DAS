"""Build traceable figures and LaTeX tables for the SensorField-M3T paper."""

from __future__ import annotations

import shutil
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\proj 1")
OUTPUT = ROOT / "paper_assets" / "sensorfield_m3t_experiments"
BENCHMARK_CSV = (
    ROOT
    / "_tmp_sensorfield_benchmark_tables"
    / "20260518_taskwise_v3_full8"
    / "benchmark_wide-final-with-scores.csv"
)
MULTISEED_CSV = (
    ROOT
    / "_tmp_sensorfield_mtl43_multiseed_analysis"
    / "analysis"
    / "protocol_results_multiseed.csv"
)
CHECKPOINT_SELECTION_CSV = (
    ROOT
    / "_tmp_sensorfield_mtl43_multiseed_analysis"
    / "analysis"
    / "checkpoint_selection_metrics.csv"
)
VIEW_FIGURE = (
    ROOT
    / "_tmp_sensorfield_fig_z1"
    / "analysis"
    / "fig_z1_view_contributions.pdf"
)
VIEW_FIGURE_PNG = VIEW_FIGURE.with_suffix(".png")

# Direct trainable-parameter counts supersede THOP's module accounting for these
# two hybrid models; FLOPs remain the original THOP-derived values.
PARAMETER_OVERRIDES = {"MultiModN": 11.683723, "M4oE": 11.668383}
MODEL_LABELS = {"ConvNeXt": "ConvNeXt-Small"}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def pm(mean: float, std: float, digits: int = 4) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return r"--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def scalar(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return r"--"
    return f"{value:.{digits}f}"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&")


def source_pm(value: object) -> str:
    values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(value))
    if len(values) != 2:
        raise ValueError(f"Expected mean and standard deviation in: {value!r}")
    return f"{float(values[0]):.4f} $\\pm$ {float(values[1]):.4f}"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark = pd.read_csv(BENCHMARK_CSV)
    for model, params in PARAMETER_OVERRIDES.items():
        benchmark.loc[benchmark["Model"] == model, "Params(M)"] = params

    protocol = pd.read_csv(MULTISEED_CSV)
    protocol = protocol.loc[protocol["split"] == "test"].copy()
    seed_metrics = pd.read_csv(CHECKPOINT_SELECTION_CSV)
    return benchmark, protocol, seed_metrics


def write_sota_table(benchmark: pd.DataFrame) -> None:
    columns = [
        "Task1 ACC",
        "Task1 F1",
        "Task1 AUC",
        "Task1 FAR",
        "Task1Score",
        "Task2 ACC",
        "Task2 F1",
        "Task2 AUC",
        "Task2 FAR",
        "Task2Score",
        "MTLScore",
        "Params(M)",
        "FLOPs(G)",
    ]
    raw = benchmark.copy()
    scores = raw[columns].copy()
    for column in columns:
        if column.startswith("Task") and column not in {"Task1Score", "Task2Score"}:
            scores[column] = raw[column].str.split(" ").str[0].astype(float)
        else:
            scores[column] = pd.to_numeric(raw[column])

    higher_is_better = {
        "Task1 ACC",
        "Task1 F1",
        "Task1 AUC",
        "Task1Score",
        "Task2 ACC",
        "Task2 F1",
        "Task2 AUC",
        "Task2Score",
        "MTLScore",
    }
    best = {
        column: scores[column].max() if column in higher_is_better else scores[column].min()
        for column in columns
    }

    body = []
    for row_index, row in raw.iterrows():
        fields = [latex_escape(MODEL_LABELS.get(str(row["Model"]), str(row["Model"])))]
        for column in columns:
            if column.startswith("Task") and column not in {"Task1Score", "Task2Score"}:
                display = source_pm(row[column])
            elif column in {"Task1Score", "Task2Score", "MTLScore"}:
                display = scalar(float(row[column]))
            else:
                display = scalar(float(row[column]))
            if np.isclose(scores.loc[row_index, column], best[column], atol=1e-8):
                display = r"\textbf{" + display + "}"
            fields.append(display)
        body.append(" & ".join(fields) + r" \\")

    table = r"""\begin{table*}[t]
\centering
\caption{Comparison with eight baselines on the clean three-view imagefork benchmark. Each row is one selected checkpoint; values are mean $\pm$ standard deviation over five stratified partitions of its common held-out predictions, rather than over independent retraining runs. TaskScore is the mean of ACC, macro-F1, AUC, and $1-\mathrm{FAR}$; MTLScore averages the two task scores. Higher is better except FAR, Params, and FLOPs.}
\label{tab:sota_comparison}
\scriptsize
\setlength{\tabcolsep}{2.1pt}
\renewcommand{\arraystretch}{1.12}
\resizebox{\textwidth}{!}{
\begin{tabular}{l|ccccc|ccccc|ccc}
\hline
Model & \multicolumn{5}{c|}{Event task} & \multicolumn{5}{c|}{Location task} & \multicolumn{3}{c}{Overall} \\
& ACC & F1 & AUC & FAR & TaskScore & ACC & F1 & AUC & FAR & TaskScore & MTLScore & Params (M) & FLOPs (G) \\
\hline
""" + "\n".join(body) + r"""
\hline
\end{tabular}}
\end{table*}
"""
    (OUTPUT / "table_sota_comparison.tex").write_text(table, encoding="utf-8")


def write_sota_table_split(benchmark: pd.DataFrame) -> None:
    """Write readable event and location tables instead of one 14-column table."""
    all_columns = [
        "Task1 ACC", "Task1 F1", "Task1 AUC", "Task1 FAR", "Task1Score",
        "Task2 ACC", "Task2 F1", "Task2 AUC", "Task2 FAR", "Task2Score",
        "MTLScore", "Params(M)", "FLOPs(G)",
    ]
    raw = benchmark.copy()
    scores = raw[all_columns].copy()
    for column in all_columns:
        if column.startswith("Task") and column not in {"Task1Score", "Task2Score"}:
            scores[column] = raw[column].str.split(" ").str[0].astype(float)
        else:
            scores[column] = pd.to_numeric(raw[column])

    higher_is_better = {
        "Task1 ACC", "Task1 F1", "Task1 AUC", "Task1Score",
        "Task2 ACC", "Task2 F1", "Task2 AUC", "Task2Score", "MTLScore",
    }
    best = {
        column: scores[column].max() if column in higher_is_better else scores[column].min()
        for column in all_columns
    }
    labels = {
        **MODEL_LABELS,
        "DAS-MAE + downstream fine-tuning head": "DAS-MAE + FT head",
    }

    def build_body(selected_columns: list[str]) -> list[str]:
        rows = []
        for row_index, row in raw.iterrows():
            values = [latex_escape(labels.get(str(row["Model"]), str(row["Model"]))) ]
            for column in selected_columns:
                if column.startswith("Task") and column not in {"Task1Score", "Task2Score"}:
                    display = source_pm(row[column])
                else:
                    display = scalar(float(row[column]))
                if np.isclose(scores.loc[row_index, column], best[column], atol=1e-8):
                    display = r"\textbf{" + display + "}"
                values.append(display)
            rows.append(" & ".join(values) + r" \\")
        return rows

    event_columns = ["Task1 ACC", "Task1 F1", "Task1 AUC", "Task1 FAR", "Task1Score", "Params(M)", "FLOPs(G)"]
    location_columns = ["Task2 ACC", "Task2 F1", "Task2 AUC", "Task2 FAR", "Task2Score", "MTLScore", "Params(M)", "FLOPs(G)"]
    table = r"""\begin{table*}[t]
\centering
\caption{Event-task comparison on the clean three-view imagefork benchmark. Each row is one selected checkpoint; values are mean $\pm$ standard deviation over five stratified partitions of its common held-out predictions, rather than over independent retraining runs. TaskScore is the mean of ACC, macro-F1, AUC, and $1-\mathrm{FAR}$. Higher is better except FAR, Params, and FLOPs.}
\label{tab:sota_event}
\small
\setlength{\tabcolsep}{4.4pt}
\renewcommand{\arraystretch}{1.14}
\resizebox{\textwidth}{!}{
\begin{tabular}{l|ccccc|cc}
\hline
Model & \multicolumn{5}{c|}{Event task} & \multicolumn{2}{c}{Efficiency} \\
& ACC & F1 & AUC & FAR & TaskScore & Params (M) & FLOPs (G) \\
\hline
""" + "\n".join(build_body(event_columns)) + r"""
\hline
\end{tabular}
}
\end{table*}

\begin{table*}[t]
\centering
\caption{Location-task and aggregate comparison under the same fixed-checkpoint protocol as Table~\ref{tab:sota_event}. MTLScore averages the event and location TaskScores. Higher is better except FAR, Params, and FLOPs.}
\label{tab:sota_location}
\small
\setlength{\tabcolsep}{3.8pt}
\renewcommand{\arraystretch}{1.14}
\resizebox{\textwidth}{!}{
\begin{tabular}{l|ccccc|c|cc}
\hline
Model & \multicolumn{5}{c|}{Location task} & Overall & \multicolumn{2}{c}{Efficiency} \\
& ACC & F1 & AUC & FAR & TaskScore & MTLScore & Params (M) & FLOPs (G) \\
\hline
""" + "\n".join(build_body(location_columns)) + r"""
\hline
\end{tabular}
}
\end{table*}
"""
    (OUTPUT / "table_sota_comparison.tex").write_text(table, encoding="utf-8")


def write_cross_condition_table(protocol: pd.DataFrame) -> None:
    case_order = [
        "full_three_view",
        "region_generalization",
        "soil_generalization",
        "acquisition_generalization",
    ]
    labels = {
        "full_three_view": "In-distribution three-view",
        "region_generalization": "Region-level",
        "soil_generalization": "Soil-level",
        "acquisition_generalization": "Acquisition-level",
    }
    selected = protocol.set_index("case_name").loc[case_order]
    rows = []
    for case_name, row in selected.iterrows():
        fields = [latex_escape(labels[case_name])]
        for prefix in ("event", "location"):
            fields.extend(
                [
                    pm(row[f"{prefix}_acc_mean"], row[f"{prefix}_acc_std"]),
                    pm(row[f"{prefix}_f1_mean"], row[f"{prefix}_f1_std"]),
                    pm(row[f"{prefix}_task_score_mean"], row[f"{prefix}_task_score_std"]),
                ]
            )
        fields.append(pm(row["mtl_score_mean"], row["mtl_score_std"]))
        rows.append(" & ".join(fields) + r" \\")

    table = r"""\begin{table*}[t]
\centering
\caption{Five-seed cross-condition generalization results. All partitions are condition-disjoint. A dash indicates that the metric is undefined because at least one evaluation class has zero support; no imputation is used.}
\label{tab:cross_condition}
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.12}
\resizebox{\textwidth}{!}{
\begin{tabular}{l|ccc|ccc|c}
\hline
Setting & \multicolumn{3}{c|}{Event task} & \multicolumn{3}{c|}{Location task} & Overall \\
& ACC & Macro-F1 & TaskScore & ACC & Macro-F1 & TaskScore & MTLScore \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
}
\end{table*}
"""
    (OUTPUT / "table_cross_condition.tex").write_text(table, encoding="utf-8")


def write_ablation_table(protocol: pd.DataFrame) -> None:
    case_order = ["full_three_view", "wo_fac", "wo_taef", "wo_gcti", "wo_all"]
    labels = {
        "full_three_view": "Full SensorField-M3T",
        "wo_fac": "w/o FAC",
        "wo_taef": "w/o TAEF",
        "wo_gcti": "w/o GCTI",
        "wo_all": "w/o All",
    }
    selected = protocol.set_index("case_name").loc[case_order]
    rows = []
    for case_name, row in selected.iterrows():
        fields = [labels[case_name]]
        for prefix in ("event", "location"):
            fields.extend(
                [
                    pm(row[f"{prefix}_acc_mean"], row[f"{prefix}_acc_std"]),
                    pm(row[f"{prefix}_task_score_mean"], row[f"{prefix}_task_score_std"]),
                ]
            )
        fields.append(pm(row["mtl_score_mean"], row["mtl_score_std"]))
        rows.append(" & ".join(fields) + r" \\")

    table = r"""\begin{table}[t]
\centering
\caption{Five-seed ablation results under the strict three-view protocol. Scores are reported as mean $\pm$ standard deviation.}
\label{tab:ablation}
\scriptsize
\setlength{\tabcolsep}{3.6pt}
\renewcommand{\arraystretch}{1.10}
\resizebox{\columnwidth}{!}{
\begin{tabular}{l|cc|cc|c}
\hline
Variant & \multicolumn{2}{c|}{Event task} & \multicolumn{2}{c|}{Location task} & Overall \\
& ACC & TaskScore & ACC & TaskScore & MTLScore \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
}
\end{table}
"""
    (OUTPUT / "table_ablation.tex").write_text(table, encoding="utf-8")


def write_checkpoint_table(seed_metrics: pd.DataFrame) -> None:
    selection_order = ["current_score", "mean_task_score", "pareto_balanced", "last"]
    labels = {
        "current_score": "Validation MTLScore",
        "mean_task_score": "Mean validation task score",
        "pareto_balanced": "Pareto-balanced",
        "last": "Last checkpoint",
    }
    data = seed_metrics.loc[
        (seed_metrics["case_name"] == "full_three_view")
        & (seed_metrics["split"] == "test")
        & (seed_metrics["selection_method"].isin(selection_order))
    ].copy()
    rows = []
    for method in selection_order:
        subset = data.loc[data["selection_method"] == method]
        fields = [labels[method]]
        for metric in ("event_acc", "location_acc", "mtl_score"):
            fields.append(pm(subset[metric].mean(), subset[metric].std(ddof=0)))
        fields.append(f"{subset['epoch'].mean():.1f}")
        rows.append(" & ".join(fields) + r" \\")

    table = r"""\begin{table}[t]
\centering
\caption{Checkpoint-selection audit for the full model over five seeds. Earlier stopping selected by validation MTLScore remains preferable to using the final epoch.}
\label{tab:checkpoint_audit}
\scriptsize
\setlength{\tabcolsep}{3.3pt}
\renewcommand{\arraystretch}{1.10}
\resizebox{\columnwidth}{!}{
\begin{tabular}{l|ccc|c}
\hline
Selection rule & Event ACC & Location ACC & MTLScore & Mean epoch \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
}
\end{table}
"""
    (OUTPUT / "table_checkpoint_audit.tex").write_text(table, encoding="utf-8")


def plot_sota_efficiency(benchmark: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.75))
    colors = {
        "SensorField-M3T": "#B3322C",
        "ConvNeXt": "#356AA0",
        "MultiModN": "#5D8E46",
        "M4oE": "#8B5A2B",
        "DAS-MAE + downstream fine-tuning head": "#8E6C8A",
        "PipelineADWinT": "#5D5D5D",
        "Aligned-MTL": "#C87C21",
        "MoCo-weighting": "#3C8D8D",
    }
    labels = {
        "ConvNeXt": "ConvNeXt-S",
        "DAS-MAE + downstream fine-tuning head": "DAS-MAE",
    }
    for _, row in benchmark.iterrows():
        name = row["Model"]
        x = float(row["Params(M)"])
        y = float(row["MTLScore"])
        size = 35 + 12 * float(row["FLOPs(G)"])
        ax.scatter(
            x,
            y,
            s=size,
            color=colors[name],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.96,
            zorder=3,
        )
        dx, dy = (0.7, 0.006)
        if name == "SensorField-M3T":
            dx, dy = (-11.0, 0.010)
        elif name == "ConvNeXt":
            dx, dy = (-12.5, -0.016)
        elif name in {"MultiModN", "M4oE", "DAS-MAE + downstream fine-tuning head"}:
            dx, dy = (0.6, -0.017)
        annotation = {"fontsize": 7.8}
        if name in {"SensorField-M3T", "ConvNeXt"}:
            annotation["ha"] = "right"
        ax.annotate(
            labels.get(name, name),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            **annotation,
        )

    ax.axhline(
        benchmark.loc[benchmark["Model"] == "SensorField-M3T", "MTLScore"].iloc[0],
        color="#B3322C",
        linestyle="--",
        linewidth=0.8,
        alpha=0.55,
    )
    ax.set_xlabel("Trainable parameters (M)")
    ax.set_ylabel("MTLScore")
    ax.set_ylim(0.65, 0.97)
    ax.set_xlim(7, 55)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        0.02,
        "Marker area is proportional to FLOPs.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, "fig_sota_efficiency")


def plot_cross_condition(protocol: pd.DataFrame) -> None:
    case_order = [
        "full_three_view",
        "region_generalization",
        "soil_generalization",
        "acquisition_generalization",
    ]
    labels = ["In-dist.", "Region", "Soil", "Acquisition"]
    selected = protocol.set_index("case_name").loc[case_order]
    panels = [
        ("event_acc", "Event ACC"),
        ("event_f1", "Event macro-F1"),
        ("location_acc", "Location ACC"),
        ("location_f1", "Location macro-F1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7), sharex=True)
    palette = ["#3E6B89", "#5B8F6A", "#B07C38", "#7A5B8C"]
    x = np.arange(len(case_order))
    for ax, (metric, title) in zip(axes.flat, panels):
        means = selected[f"{metric}_mean"].to_numpy(dtype=float)
        stds = selected[f"{metric}_std"].to_numpy(dtype=float)
        ax.bar(x, means, yerr=stds, capsize=2.5, color=palette, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylim(0.2, 1.02)
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        for idx, value in enumerate(means):
            ax.text(idx, min(value + stds[idx] + 0.025, 1.0), f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xticks(x, labels)
    fig.text(0.01, 0.5, "Score", va="center", rotation="vertical", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.02, 1, 1))
    save_figure(fig, "fig_cross_condition")


def plot_ablation(protocol: pd.DataFrame) -> None:
    case_order = ["full_three_view", "wo_fac", "wo_taef", "wo_gcti", "wo_all"]
    labels = ["Full", "w/o FAC", "w/o TAEF", "w/o GCTI", "w/o All"]
    selected = protocol.set_index("case_name").loc[case_order]
    panels = [
        ("event_task_score", "Event TaskScore"),
        ("location_task_score", "Location TaskScore"),
        ("mtl_score", "MTLScore"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.85))
    x = np.arange(len(case_order))
    colors = ["#B3322C"] + ["#78909C"] * (len(case_order) - 1)
    for ax, (metric, title) in zip(axes, panels):
        means = selected[f"{metric}_mean"].to_numpy(dtype=float)
        stds = selected[f"{metric}_std"].to_numpy(dtype=float)
        ax.bar(x, means, yerr=stds, capsize=2.5, color=colors, edgecolor="white", linewidth=0.7)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        for idx, value in enumerate(means):
            ax.text(idx, value + stds[idx] + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=6.8)
    fig.tight_layout()
    save_figure(fig, "fig_ablation")


def copy_view_figure() -> None:
    for source in (VIEW_FIGURE, VIEW_FIGURE_PNG):
        if source.exists():
            shutil.copy2(source, OUTPUT / source.name)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    benchmark, protocol, seed_metrics = load_data()
    benchmark.to_csv(OUTPUT / "sota_paper_values.csv", index=False)
    protocol.to_csv(OUTPUT / "multiseed_test_protocol_values.csv", index=False)

    write_sota_table_split(benchmark)
    write_cross_condition_table(protocol)
    write_ablation_table(protocol)
    write_checkpoint_table(seed_metrics)
    plot_sota_efficiency(benchmark)
    plot_cross_condition(protocol)
    plot_ablation(protocol)
    copy_view_figure()
    print(f"Wrote paper assets to: {OUTPUT}")


if __name__ == "__main__":
    main()
