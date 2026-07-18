from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_CSV = (
    PROJECT_ROOT
    / "representative_multimodal_samples"
    / "representative_multimodal_samples.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "paper_assets" / "sensorfield_m3t_experiments" / "fig3_revised"
)

CLASS_ORDER = ["driving", "background", "excavator", "walking"]
CLASS_NAMES = {
    "driving": ("车辆行驶", "Vehicle driving"),
    "background": ("背景噪声", "Background noise"),
    "excavator": ("挖掘施工", "Excavator construction"),
    "walking": ("人工作业", "Human activity"),
}
CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
]


def configure_matplotlib() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in CHINESE_FONT_CANDIDATES if name in available), None)
    if selected:
        plt.rcParams["font.sans-serif"] = [selected]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 240
    plt.rcParams["savefig.dpi"] = 300
    return selected or "default"


def load_signal(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        signal = np.loadtxt(handle, delimiter=",", dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if signal.ndim != 2:
        raise ValueError(f"Expected 1D/2D CSV signal, got shape={signal.shape}: {path}")
    return signal


def load_samples(sample_csv: Path) -> dict[str, dict[str, str]]:
    with sample_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {row["class"]: row for row in csv.DictReader(handle)}
    missing = [name for name in CLASS_ORDER if name not in rows]
    if missing:
        raise RuntimeError(f"Missing classes {missing} in {sample_csv}")
    return rows


def block_average(signal: np.ndarray, block_size: int) -> np.ndarray:
    usable = (signal.shape[-1] // block_size) * block_size
    trimmed = signal[..., :usable]
    return trimmed.reshape(*trimmed.shape[:-1], usable // block_size, block_size).mean(axis=-1)


def select_representative_row(signal: np.ndarray) -> tuple[int, np.ndarray]:
    centered = signal - signal.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=-1))
    row_index = int(np.argmax(rms))
    return row_index, np.asarray(signal[row_index], dtype=np.float32)


def compute_stft(trace: np.ndarray, sample_rate: float, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    tensor = torch.from_numpy(trace.astype(np.float32))
    tensor = tensor - tensor.mean()
    window = torch.hann_window(win_length)
    spec = torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
    )
    magnitude = torch.log1p(torch.abs(spec)).cpu().numpy()
    return magnitude


def compute_gaf(trace: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(trace.astype(np.float32)).view(1, 1, -1)
    pooled = F.adaptive_avg_pool1d(tensor, size).view(-1)
    min_val = pooled.amin()
    max_val = pooled.amax()
    scaled = 2.0 * (pooled - min_val) / (max_val - min_val + 1e-6) - 1.0
    scaled = scaled.clamp(-0.999999, 0.999999)
    phase = torch.acos(scaled)
    return torch.cos(phase[:, None] + phase[None, :]).cpu().numpy()


def robust_limits(values: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    lo, hi = np.percentile(values[np.isfinite(values)], [lower, upper])
    if abs(float(hi - lo)) < 1e-9:
        hi = lo + 1.0
    return float(lo), float(hi)


def render_figure(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    font_name = configure_matplotlib()
    samples = load_samples(args.sample_csv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(CLASS_ORDER),
        3,
        figsize=(14.6, 12.8),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.05], "wspace": 0.30, "hspace": 0.44},
    )

    column_titles = [
        "块平均后波形",
        "STFT 时频图（Hz 标定）",
        "格拉姆角场（GAF）图像",
    ]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=17, fontweight="bold", pad=12)

    source_rows: list[dict[str, str]] = []
    for row, class_name in enumerate(CLASS_ORDER):
        item = samples[class_name]
        source_path = Path(item["path"])
        raw_signal = load_signal(source_path)
        processed_signal = block_average(raw_signal, args.block_size)
        selected_row, trace = select_representative_row(processed_signal)
        raw_fs = raw_signal.shape[-1] / args.duration_sec
        processed_fs = processed_signal.shape[-1] / args.duration_sec
        time_axis = np.linspace(0.0, args.duration_sec, trace.shape[-1], endpoint=False)

        stft_map = compute_stft(
            trace,
            sample_rate=processed_fs,
            n_fft=args.stft_n_fft,
            hop_length=args.stft_hop_length,
            win_length=args.stft_win_length,
        )
        gaf = compute_gaf(trace, size=args.gaf_size)

        ax_wave, ax_stft, ax_gaf = axes[row]
        ax_wave.plot(time_axis, trace, color="#1F77B4", linewidth=1.0)
        max_abs = float(np.max(np.abs(trace)))
        if max_abs > 0:
            ax_wave.set_ylim(-1.08 * max_abs, 1.08 * max_abs)
        ax_wave.set_xlim(0.0, args.duration_sec)
        ax_wave.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax_wave.set_xlabel("时间 (s)", fontsize=12)
        ax_wave.set_ylabel("幅值", fontsize=12)
        ax_wave.tick_params(labelsize=10)

        vmin, vmax = robust_limits(stft_map)
        ax_stft.imshow(
            stft_map,
            aspect="auto",
            origin="lower",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            extent=[0.0, args.duration_sec, 0.0, processed_fs / 2.0],
        )
        ax_stft.set_xlabel("时间 (s)", fontsize=12)
        ax_stft.set_ylabel("频率 (Hz)", fontsize=12)
        ax_stft.set_ylim(0.0, processed_fs / 2.0)
        ax_stft.set_yticks([0, 25, 50, 75, 100])
        ax_stft.tick_params(labelsize=10)

        ax_gaf.imshow(gaf, aspect="auto", origin="lower", cmap="viridis", vmin=-1.0, vmax=1.0)
        ax_gaf.set_xlabel("时间索引", fontsize=12)
        ax_gaf.set_ylabel("时间索引", fontsize=12)
        ax_gaf.tick_params(labelsize=10)

        class_cn, class_en = CLASS_NAMES[class_name]
        ax_wave.text(
            -0.28,
            0.5,
            f"{class_cn}\n{class_en}",
            transform=ax_wave.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

        source_rows.append(
            {
                "class": class_name,
                "class_cn": class_cn,
                "class_en": class_en,
                "source_path": str(source_path),
                "raw_shape": "x".join(str(v) for v in raw_signal.shape),
                "processed_shape": "x".join(str(v) for v in processed_signal.shape),
                "selected_processed_row_index": str(selected_row),
                "duration_sec": f"{args.duration_sec:.6f}",
                "raw_effective_fs_hz": f"{raw_fs:.6f}",
                "processed_effective_fs_hz": f"{processed_fs:.6f}",
                "block_size": str(args.block_size),
                "stft_n_fft": str(args.stft_n_fft),
                "stft_hop_length": str(args.stft_hop_length),
                "stft_win_length": str(args.stft_win_length),
                "gaf_size": str(args.gaf_size),
            }
        )

    fig.suptitle("四类典型事件的多模态表征", fontsize=20, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.012,
        "注：STFT 纵轴已按块平均后等效采样率 200 Hz 转换为 Hz；原图中的 8/16/24/32 为频率 bin 索引，不应直接解释为 Hz。",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.920, bottom=0.060)

    png_path = output_dir / "fig3_revised_multimodal_grid.png"
    pdf_path = output_dir / "fig3_revised_multimodal_grid.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    source_csv = output_dir / "fig3_revised_sources.csv"
    with source_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)

    note = {
        "font": font_name,
        "main_change": [
            "The waveform column is labeled as block-averaged/preprocessed waveform instead of raw waveform.",
            "The STFT y-axis is converted to physical frequency in Hz using the post-block-averaging effective sampling rate.",
            "A note is added to avoid interpreting STFT bin indices 8/16/24/32 as Hz.",
        ],
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "source_csv": str(source_csv),
    }
    (output_dir / "fig3_revised_note.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    return png_path, pdf_path, source_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot revised Fig. 3 with corrected STFT frequency axis.")
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--stft-n-fft", type=int, default=128)
    parser.add_argument("--stft-hop-length", type=int, default=24)
    parser.add_argument("--stft-win-length", type=int, default=128)
    parser.add_argument("--gaf-size", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    png_path, pdf_path, source_csv = render_figure(parse_args())
    print(f"Saved revised Fig. 3 PNG: {png_path}")
    print(f"Saved revised Fig. 3 PDF: {pdf_path}")
    print(f"Saved source audit CSV: {source_csv}")


if __name__ == "__main__":
    main()
