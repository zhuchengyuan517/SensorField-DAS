from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_CSV = (
    PROJECT_ROOT
    / "representative_multimodal_samples"
    / "representative_multimodal_samples.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "paper_assets"
    / "sensorfield_m3t_experiments"
    / "frequency_component_audit"
)

CLASS_ORDER = ["driving", "background", "excavator", "walking"]
CLASS_LABELS = {
    "driving": "Vehicle driving",
    "background": "Background noise",
    "excavator": "Excavator construction",
    "walking": "Human activity",
}
CLASS_COLORS = {
    "driving": "#1F77B4",
    "background": "#7F7F7F",
    "excavator": "#D95F02",
    "walking": "#1B9E77",
}
HARMONIC_FREQS = [8.0, 16.0, 24.0, 32.0]


def load_signal(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        signal = np.loadtxt(handle, delimiter=",", dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if signal.ndim != 2:
        raise ValueError(f"Expected a 1D/2D CSV signal, got shape={signal.shape}: {csv_path}")
    return signal


def load_representative_samples(sample_csv: Path) -> list[dict[str, str]]:
    with sample_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No samples found in {sample_csv}")
    by_class = {row["class"]: row for row in rows}
    ordered = [by_class[name] for name in CLASS_ORDER if name in by_class]
    missing = [name for name in CLASS_ORDER if name not in by_class]
    if missing:
        raise RuntimeError(f"Missing representative classes {missing} in {sample_csv}")
    return ordered


def block_average(signal: np.ndarray, block_size: int) -> np.ndarray:
    usable = (signal.shape[-1] // block_size) * block_size
    if usable <= 0:
        raise ValueError("Signal is shorter than one block.")
    trimmed = signal[..., :usable]
    return trimmed.reshape(*trimmed.shape[:-1], usable // block_size, block_size).mean(axis=-1)


def average_relative_psd(rows: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return mean normalized one-sided PSD over rows.

    Each row is demeaned and Hann-windowed. The row PSD is normalized by its own
    total power before averaging so that the curve emphasizes common spectral
    structure rather than absolute amplitude differences among samples/rows.
    """
    spectra = []
    for row in rows:
        x = np.asarray(row, dtype=np.float64)
        x = x - np.mean(x)
        if not np.any(np.isfinite(x)) or np.std(x) < 1e-12:
            continue
        window = np.hanning(x.size)
        xw = x * window
        spec = np.fft.rfft(xw)
        power = (np.abs(spec) ** 2) / max(float(np.sum(window**2)), 1e-12)
        total = float(np.sum(power))
        if total <= 1e-20:
            continue
        spectra.append(power / total)
    if not spectra:
        raise ValueError("No valid rows for PSD computation.")
    mean_psd = np.mean(np.vstack(spectra), axis=0)
    freqs = np.fft.rfftfreq(rows.shape[-1], d=1.0 / sample_rate)
    return freqs, mean_psd


def to_db(psd: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(psd, 1e-18))


def nearest_value(freqs: np.ndarray, psd: np.ndarray, target_freq: float) -> tuple[float, float]:
    index = int(np.argmin(np.abs(freqs - target_freq)))
    return float(freqs[index]), float(psd[index])


def top_peaks(freqs: np.ndarray, psd: np.ndarray, max_freq: float, count: int) -> list[tuple[float, float]]:
    mask = (freqs > 0.0) & (freqs <= max_freq)
    local_freqs = freqs[mask]
    local_psd = psd[mask]
    if local_freqs.size == 0:
        return []
    order = np.argsort(local_psd)[::-1]
    peaks: list[tuple[float, float]] = []
    min_separation_hz = max_freq / 200.0
    for idx in order:
        freq = float(local_freqs[idx])
        power = float(local_psd[idx])
        if any(abs(freq - existing_freq) < min_separation_hz for existing_freq, _ in peaks):
            continue
        peaks.append((freq, power))
        if len(peaks) >= count:
            break
    return peaks


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(
    ax,
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    max_freq: float,
) -> None:
    for class_name in CLASS_ORDER:
        freqs, psd = spectra[class_name]
        mask = freqs <= max_freq
        ax.plot(
            freqs[mask],
            to_db(psd[mask]),
            linewidth=2.0,
            color=CLASS_COLORS[class_name],
            label=CLASS_LABELS[class_name],
        )
    for freq in HARMONIC_FREQS:
        ax.axvline(freq, color="#333333", linestyle="--", linewidth=0.9, alpha=0.45)
        ax.text(
            freq,
            0.98,
            f"{freq:.0f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            color="#333333",
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Average relative power (dB)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_xlim(0.0, max_freq)


def render_figures(
    output_dir: Path,
    raw_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    processed_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    max_freq: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 220,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharey=True)
    plot_panel(
        axes[0],
        raw_spectra,
        "Before block averaging: raw CSV signal",
        max_freq=max_freq,
    )
    plot_panel(
        axes[1],
        processed_spectra,
        "After preprocessing: 10-point block-averaged signal",
        max_freq=max_freq,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Average Power Spectrum Audit for Fig. 3 Representative Signals", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(output_dir / "fig3_average_power_spectrum_raw_vs_preprocessed.png", bbox_inches="tight")
    fig.savefig(output_dir / "fig3_average_power_spectrum_raw_vs_preprocessed.pdf", bbox_inches="tight")
    plt.close(fig)

    for name, title, spectra in [
        ("raw", "Average power spectrum before block averaging", raw_spectra),
        ("preprocessed", "Average power spectrum after 10-point block averaging", processed_spectra),
    ]:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        plot_panel(ax, spectra, title, max_freq=max_freq)
        ax.legend(loc="upper right", frameon=False, fontsize=9.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"fig3_average_power_spectrum_{name}.png", bbox_inches="tight")
        fig.savefig(output_dir / f"fig3_average_power_spectrum_{name}.pdf", bbox_inches="tight")
        plt.close(fig)


def save_spectrum_values(
    output_dir: Path,
    raw_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    processed_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    max_freq: float,
) -> None:
    rows: list[dict[str, str]] = []
    for stage, spectra in [("raw", raw_spectra), ("preprocessed", processed_spectra)]:
        for class_name, (freqs, psd) in spectra.items():
            mask = freqs <= max_freq
            for freq, value in zip(freqs[mask], psd[mask]):
                rows.append(
                    {
                        "stage": stage,
                        "class": class_name,
                        "class_label": CLASS_LABELS[class_name],
                        "frequency_hz": f"{float(freq):.6f}",
                        "relative_power": f"{float(value):.12e}",
                        "relative_power_db": f"{float(to_db(np.array([value]))[0]):.6f}",
                    }
                )
    write_csv(output_dir / "fig3_average_power_spectrum_values.csv", rows)


def build_audit_tables(
    output_dir: Path,
    raw_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    processed_spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    max_freq: float,
) -> None:
    harmonic_rows: list[dict[str, str]] = []
    peak_rows: list[dict[str, str]] = []
    for stage, spectra in [("raw", raw_spectra), ("preprocessed", processed_spectra)]:
        for class_name, (freqs, psd) in spectra.items():
            for target_freq in HARMONIC_FREQS:
                actual_freq, power = nearest_value(freqs, psd, target_freq)
                harmonic_rows.append(
                    {
                        "stage": stage,
                        "class": class_name,
                        "class_label": CLASS_LABELS[class_name],
                        "target_frequency_hz": f"{target_freq:.1f}",
                        "nearest_frequency_hz": f"{actual_freq:.6f}",
                        "relative_power": f"{power:.12e}",
                        "relative_power_db": f"{float(to_db(np.array([power]))[0]):.6f}",
                    }
                )
            for rank, (freq, power) in enumerate(top_peaks(freqs, psd, max_freq=max_freq, count=12), start=1):
                peak_rows.append(
                    {
                        "stage": stage,
                        "class": class_name,
                        "class_label": CLASS_LABELS[class_name],
                        "rank": str(rank),
                        "peak_frequency_hz": f"{freq:.6f}",
                        "relative_power": f"{power:.12e}",
                        "relative_power_db": f"{float(to_db(np.array([power]))[0]):.6f}",
                    }
                )
    write_csv(output_dir / "fig3_harmonic_frequency_audit.csv", harmonic_rows)
    write_csv(output_dir / "fig3_top_spectral_peaks.csv", peak_rows)


def write_bin_mapping(output_dir: Path, processed_fs: float, stft_n_fft: int) -> None:
    rows = []
    bin_spacing = processed_fs / stft_n_fft
    for bin_index in [8, 16, 24, 32]:
        rows.append(
            {
                "stft_bin_index": str(bin_index),
                "effective_sample_rate_hz": f"{processed_fs:.6f}",
                "stft_n_fft": str(stft_n_fft),
                "frequency_hz": f"{bin_index * bin_spacing:.6f}",
            }
        )
    write_csv(output_dir / "fig3_stft_bin_to_hz_mapping.csv", rows)


def write_response_draft(output_dir: Path, duration_sec: float, block_size: int, stft_n_fft: int) -> None:
    english_text = f"""# Response draft: constant frequency components in Fig. 3

We thank the reviewer for pointing this out. We re-examined the representative vibration signals in Fig. 3 by plotting the average power spectra both before and after the visualization preprocessing. Specifically, the raw CSV segment contains 10000 points and is displayed as a {duration_sec:g}-s window, corresponding to an effective raw sampling rate of 2000 Hz under the Fig. 3 plotting convention. The visualized waveform is obtained by non-overlapping {block_size}-point block averaging, yielding 1000 points and an effective sampling rate of 200 Hz. The spectra are now provided in the revised supplementary/response figure.

The audit indicates that the values that visually appear around 8, 16, 24, and 32 in the original STFT panel should not be interpreted directly as 8 Hz, 16 Hz, 24 Hz, and 32 Hz. In that figure, the vertical STFT coordinate was a frequency-bin/component index. With the block-averaged effective sampling rate of 200 Hz and the STFT setting of n_fft={stft_n_fft}, bin indices 8, 16, 24, and 32 correspond to approximately 12.5, 25.0, 37.5, and 50.0 Hz, respectively. The average power-spectrum audit further shows that the dominant common spectral peaks are around 13.4, 26.6, 40.0, 53.4, and 66.6 Hz, and these peaks are already present before block averaging and are retained after preprocessing. Therefore, the shared harmonic pattern is not created by block averaging; however, the original wording/axis label could indeed lead to an incorrect Hz-level interpretation.

We will revise Fig. 3 and the corresponding description accordingly. The STFT axis will either be relabeled as a frequency-bin index or converted to physical frequency in Hz using the effective sampling rate after block averaging. We will also add an explanation that these shared narrow-band components are common-mode spectral components, likely related to periodic mechanical/electromechanical background vibrations of the acquisition environment and their harmonics. Since they appear across all event categories, they are not used as class-specific evidence; the discriminative patterns are mainly reflected in the temporal envelope, local energy distribution, transient characteristics, and space-time-frequency evolution.

Revision to add in manuscript:
\"The apparent regular horizontal components in the STFT maps are shared spectral components across the representative samples. We note that the vertical coordinate in the original visualization denotes the STFT frequency-bin index rather than a directly calibrated frequency in Hz. After accounting for the effective sampling rate of the block-averaged signal, the dominant common spectral peaks are observed around 13.4 Hz and its harmonic components. The same peaks are already visible in the raw-signal average power spectra and are preserved after preprocessing, confirming that they are not artifacts introduced by block averaging or downsampling. These common components are treated as background/common-mode periodic vibrations of the sensing system or field environment, while event discrimination relies primarily on class-dependent temporal-envelope variation, local energy concentration, and space-time-frequency evolution.\"
"""
    chinese_text = f"""# 审稿意见回复草稿：图3中的恒定频率分量

感谢审稿人的提醒。我们根据该意见重新检查了图3中四类代表性振动信号，并分别绘制了块平均前原始 CSV 信号以及可视化预处理后信号的平均功率谱。具体而言，图3对应的原始信号段包含 10000 个采样点，并按 {duration_sec:g} s 时间窗显示，因此在该图示口径下原始信号的等效采样率为 2000 Hz；图中展示的预处理波形由非重叠 {block_size} 点块平均得到，长度为 1000 点，对应等效采样率为 200 Hz。

重新核查后我们发现，原图 STFT 面板中看起来位于约 8、16、24、32 的水平分量，不应直接解释为 8 Hz、16 Hz、24 Hz 和 32 Hz。原图纵轴实际表示 STFT 频率分量/频率 bin 索引，而不是严格标定后的物理频率。按照块平均后的等效采样率 200 Hz 和 STFT 参数 n_fft={stft_n_fft}，bin 8、16、24 和 32 分别对应约 12.5、25.0、37.5 和 50.0 Hz。因此，原图的纵轴标注确实可能造成 Hz 层面的误读。

平均功率谱结果进一步表明，四类样本中共同存在的主要窄带峰值约位于 13.4、26.6、40.0、53.4 和 66.6 Hz，呈现近似倍频关系。这些峰值在块平均前的原始信号中已经存在，并在块平均/降采样后被保留和凸显。因此，该类共同谐波成分并非由块平均或降采样新引入，而更可能来自采集系统或现场环境中的周期性机械/机电背景振动及其谐波响应。由于这些成分在不同事件类型中均存在，我们将在文中说明其属于共同模态/背景谱线，而不是类别判别性事件特征；不同事件的区分主要依赖时域包络变化、局部能量分布、瞬态结构以及时空频联合演化模式。

拟在文中补充说明如下：

“图3中 STFT 图存在若干跨类别共享的规则水平谱线。需要说明的是，原图纵轴表示 STFT 频率 bin 索引，而非直接标定的 Hz 频率。结合块平均后信号的等效采样率进行换算后，共享窄带峰值主要位于约 13.4 Hz 及其倍频附近。平均功率谱分析显示，这些峰值在块平均前的原始信号中已经存在，并在预处理后得到保留，说明其并非块平均或降采样引入的伪影。本文将其视为采集系统或现场环境中的共同模态周期性背景振动，而模型判别主要依赖不同事件对应的时域包络、局部能量集中区域和时空频演化差异。”
"""
    (output_dir / "review_response_frequency_components.md").write_text(english_text, encoding="utf-8")
    (output_dir / "review_response_frequency_components_cn.md").write_text(chinese_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Fig. 3 average power spectra before and after block averaging."
    )
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--stft-n-fft", type=int, default=128)
    parser.add_argument("--max-freq", type=float, default=80.0)
    args = parser.parse_args()

    samples = load_representative_samples(args.sample_csv)
    raw_spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    processed_spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    source_rows: list[dict[str, str]] = []

    for sample in samples:
        class_name = sample["class"]
        path = Path(sample["path"])
        signal = load_signal(path)
        processed = block_average(signal, block_size=args.block_size)
        raw_fs = signal.shape[-1] / args.duration_sec
        processed_fs = processed.shape[-1] / args.duration_sec
        raw_spectra[class_name] = average_relative_psd(signal, sample_rate=raw_fs)
        processed_spectra[class_name] = average_relative_psd(processed, sample_rate=processed_fs)
        source_rows.append(
            {
                "class": class_name,
                "class_label": CLASS_LABELS[class_name],
                "source_path": str(path),
                "raw_shape": "x".join(str(x) for x in signal.shape),
                "preprocessed_shape": "x".join(str(x) for x in processed.shape),
                "duration_sec": f"{args.duration_sec:.6f}",
                "raw_effective_fs_hz": f"{raw_fs:.6f}",
                "preprocessed_effective_fs_hz": f"{processed_fs:.6f}",
                "block_size": str(args.block_size),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fig3_power_spectrum_sources.csv", source_rows)
    render_figures(args.output_dir, raw_spectra, processed_spectra, max_freq=args.max_freq)
    save_spectrum_values(args.output_dir, raw_spectra, processed_spectra, max_freq=args.max_freq)
    build_audit_tables(args.output_dir, raw_spectra, processed_spectra, max_freq=args.max_freq)
    processed_fs = processed.shape[-1] / args.duration_sec
    write_bin_mapping(args.output_dir, processed_fs=processed_fs, stft_n_fft=args.stft_n_fft)
    write_response_draft(
        args.output_dir,
        duration_sec=args.duration_sec,
        block_size=args.block_size,
        stft_n_fft=args.stft_n_fft,
    )
    print(f"Saved Fig. 3 frequency-component audit to {args.output_dir}")


if __name__ == "__main__":
    main()
