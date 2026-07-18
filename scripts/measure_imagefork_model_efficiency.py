from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = PROJECT_ROOT / "libmtl_das_patch" / "examples" / "das_csv" / "compare_imagefork_suite.py"
OLD_EVAL_SCRIPT = PROJECT_ROOT / "libmtl_das_patch" / "examples" / "das_csv" / "evaluate_old_libtl_imagefork_run.py"
OLD_SOTA_SCRIPT = Path(r"D:\github\LibMTL-main\LibMTL-main\examples\das_csv\sota_multitask_imagefork_benchmark.py")
OUTPUT_DIR = PROJECT_ROOT / "paper_assets" / "sensorfield_m3t_experiments" / "model_efficiency_retest_20260715"

EVENT_CLASSES = ["walking", "excavator", "driving", "background"]
DISTANCE_CLASSES = ["Alarm area", "Tracking area", "No-threat area"]

HYBRID_RUNS = {
    "MultiModN": PROJECT_ROOT / "_tmp_added4_bench" / "multimodn" / "20260518_215955",
    "M4oE": PROJECT_ROOT / "_tmp_added4_bench" / "m4oe" / "20260518_220341",
    "DAS-MAE": PROJECT_ROOT / "_tmp_added4_bench" / "dasmae" / "20260518_220731",
    "PipelineADWinT": PROJECT_ROOT / "_tmp_added4_bench" / "pipelineadwint" / "20260518_221136",
}

PREVIOUS_TABLE_VALUES = {
    "ConvNeXt-Small": {"previous_params_m": 49.4195, "previous_flops_g": 17.3660},
    "MultiModN": {"previous_params_m": 46.0000, "previous_flops_g": 14.0500},
    "M4oE": {"previous_params_m": 45.0000, "previous_flops_g": 12.2100},
    "DAS-MAE": {"previous_params_m": 38.0000, "previous_flops_g": 8.7000},
    "PipelineADWinT": {"previous_params_m": 78.0000, "previous_flops_g": 47.0000},
    "Aligned-MTL": {"previous_params_m": 22.0000, "previous_flops_g": 11.5500},
    "MoCo-weighting": {"previous_params_m": 22.0000, "previous_flops_g": 11.5500},
}


def dynamic_import(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def count_total_params(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def normalize_profiled_config(config: dict) -> dict:
    """Disable pretrained downloads for deterministic structure-only profiling."""
    copied = dict(config)
    copied["location_image_backbone_pretrained"] = False
    return copied


def patch_adapted_timm_pretrained(compare_module):
    adapted_module = sys.modules.get("LibMTL.model.adapted_benchmark_imagefork")
    if adapted_module is None or getattr(adapted_module, "timm", None) is None:
        return lambda: None
    original_create_model = adapted_module.timm.create_model

    def create_model_without_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    adapted_module.timm.create_model = create_model_without_pretrained

    def restore() -> None:
        adapted_module.timm.create_model = original_create_model

    return restore


def clear_libtl_module_cache() -> None:
    for module_name in list(sys.modules):
        if module_name == "LibMTL" or module_name.startswith("LibMTL."):
            del sys.modules[module_name]


def profile_row(model_name: str, model: nn.Module, profile_fn, device: torch.device, input_protocol: str, source: str) -> dict:
    model = model.to(device)
    model.eval()
    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)
    with torch.inference_mode():
        gflops, thop_params_m = profile_fn(model, device)
    previous = PREVIOUS_TABLE_VALUES.get(model_name, {})
    row = {
        "model": model_name,
        "input_protocol": input_protocol,
        "source": source,
        "params_m": f"{total_params / 1e6:.6f}",
        "trainable_params_m": f"{trainable_params / 1e6:.6f}",
        "thop_params_m": "" if thop_params_m is None else f"{float(thop_params_m):.6f}",
        "flops_g": "" if gflops is None else f"{float(gflops):.6f}",
        "macs_g": "" if gflops is None else f"{float(gflops) / 2.0:.6f}",
        "flop_convention": "THOP MACs x 2, batch=1/profile branch",
        "previous_params_m": "" if "previous_params_m" not in previous else f"{previous['previous_params_m']:.6f}",
        "previous_flops_g": "" if "previous_flops_g" not in previous else f"{previous['previous_flops_g']:.6f}",
        "delta_params_m": ""
        if "previous_params_m" not in previous
        else f"{total_params / 1e6 - previous['previous_params_m']:.6f}",
        "delta_flops_g": ""
        if gflops is None or "previous_flops_g" not in previous
        else f"{float(gflops) - previous['previous_flops_g']:.6f}",
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def build_old_libtl_trainer_model(old_eval_module, weighting: str, device: torch.device) -> nn.Module:
    old_main = old_eval_module.old_main
    args = SimpleNamespace(
        gpu_id=device.index if device.type == "cuda" and device.index is not None else 0,
        weighting=weighting,
        arch="LTB",
        lr=2e-5,
        weight_decay=2e-4,
        step_size=10,
        gamma=0.7,
        image_size=224,
        dataset_path=str(PROJECT_ROOT / "converted_csv" / "MTL43_imagefork_dedup_clean"),
        mtl43_root=str(PROJECT_ROOT / "converted_csv" / "MTL43"),
        image_root=str(PROJECT_ROOT / "converted_csv" / "flower_data_rl_dedup_clean"),
        bs=32,
        num_workers=0,
        csv_input_height=6,
        csv_input_width=10000,
        normalize="none",
        event_classes=",".join(EVENT_CLASSES),
        distance_classes=",".join(DISTANCE_CLASSES),
    )
    params = old_eval_module.build_params(args)
    kwargs, optim_param, scheduler_param = old_main.prepare_args(params)
    task_dict = {
        "event_type": {
            "metrics": ["Acc"],
            "metrics_fn": old_main.MaskedAccMetric(),
            "loss_fn": old_main.SafeCELoss(weight=None, ignore_index=None, label_smoothing=0.0),
            "weight": [1],
        },
        "distance_cls": {
            "metrics": ["Acc"],
            "metrics_fn": old_main.MaskedAccMetric(ignore_index=old_main.DISTANCE_IGNORE_INDEX),
            "loss_fn": old_main.SafeCELoss(
                weight=None,
                ignore_index=old_main.DISTANCE_IGNORE_INDEX,
                label_smoothing=0.05,
            ),
            "weight": [1],
        },
    }
    decoders = nn.ModuleDict(
        {
            "event_type": old_main.GlobalPoolDecoder(512, len(EVENT_CLASSES)),
            "distance_cls": old_main.GlobalPoolDecoder(512, len(DISTANCE_CLASSES)),
        }
    )
    encoder_class = old_main.build_resnet18_encoder(in_channels=3, pretrained=False)
    trainer = old_main.DASPLETrainer(
        task_dict=task_dict,
        weighting=weighting,
        architecture="LTB",
        encoder_class=encoder_class,
        decoders=decoders,
        rep_grad=False,
        multi_input=False,
        optim_param=optim_param,
        scheduler_param=scheduler_param,
        save_path=None,
        load_path=None,
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        event_class_names=EVENT_CLASSES,
        distance_class_names=DISTANCE_CLASSES,
        **kwargs,
    )
    trainer.model.epoch = 0
    trainer.model.epochs = 1
    return trainer.model


def write_latex_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Measured model complexity of the seven comparison models. FLOPs are computed with THOP using MACs$\times$2.}",
        r"\label{tab:model_efficiency_retest}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & Params (M) & FLOPs (G) \\",
        r"\midrule",
    ]
    for row in rows:
        flops_text = "--" if row["flops_g"] == "" else f"{float(row['flops_g']):.2f}"
        lines.append(f"{row['model']} & {float(row['params_m']):.2f} & {flops_text} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-profile Params and FLOPs for seven ImageFork comparison models.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")

    if str(PROJECT_ROOT / "libmtl_das_patch") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "libmtl_das_patch"))
    if str(COMPARE_SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(COMPARE_SCRIPT.parent))

    compare_module = dynamic_import(COMPARE_SCRIPT, "sensorfield_compare_imagefork_suite_profile")
    old_sota_module = dynamic_import(OLD_SOTA_SCRIPT, "old_sota_imagefork_profile")
    restore_timm = patch_adapted_timm_pretrained(compare_module)
    clear_libtl_module_cache()
    old_eval_module = dynamic_import(OLD_EVAL_SCRIPT, "old_libtl_imagefork_profile")

    rows: list[dict] = []
    try:
        convnext = old_sota_module.build_model(
            "convnext_small",
            event_classes=len(EVENT_CLASSES),
            distance_classes=len(DISTANCE_CLASSES),
            pretrained=False,
            dropout=0.2,
        )
        rows.append(
            profile_row(
                "ConvNeXt-Small",
                convnext,
                compare_module.profile_dense_model,
                device,
                "image [1,3,224,224]",
                str(PROJECT_ROOT / "_tmp_compare_imagefork_sota_clean" / "20260518_133929" / "convnext_small"),
            )
        )

        hybrid_builders = {
            "MultiModN": compare_module.build_multimodn_from_config,
            "M4oE": compare_module.build_m4oe_from_config,
            "DAS-MAE": compare_module.build_dasmae_from_config,
            "PipelineADWinT": compare_module.build_pipelineadwint_from_config,
        }
        for model_name, run_dir in HYBRID_RUNS.items():
            config = normalize_profiled_config(load_json(run_dir / "run_config.json"))
            model = hybrid_builders[model_name](config, EVENT_CLASSES, DISTANCE_CLASSES)
            rows.append(
                profile_row(
                    model_name,
                    model,
                    compare_module.profile_hybrid_model,
                    device,
                    "hybrid: csv [1,1,6,10000] + image [1,3,224,224]",
                    str(run_dir),
                )
            )

        for model_name, weighting in (("Aligned-MTL", "Aligned_MTL"), ("MoCo-weighting", "MoCo")):
            model = build_old_libtl_trainer_model(old_eval_module, weighting=weighting, device=device)
            rows.append(
                profile_row(
                    model_name,
                    model,
                    old_eval_module.profile_dense_model,
                    device,
                    "LibMTL unified image [1,3,224,224]",
                    str(
                        PROJECT_ROOT
                        / ("_tmp_compare_aligned_clean" if weighting == "Aligned_MTL" else "_tmp_compare_moco_weighting_clean")
                    ),
                )
            )
    finally:
        restore_timm()

    write_csv(output_dir / "seven_model_efficiency_retest.csv", rows)
    write_json(output_dir / "seven_model_efficiency_retest.json", rows)
    write_latex_table(output_dir / "seven_model_efficiency_retest.tex", rows)

    print(f"Device: {device}")
    print(f"Saved CSV: {output_dir / 'seven_model_efficiency_retest.csv'}")
    print(f"Saved LaTeX: {output_dir / 'seven_model_efficiency_retest.tex'}")
    print("")
    print("model,params_m,flops_g,thop_params_m,macs_g")
    for row in rows:
        print(f"{row['model']},{row['params_m']},{row['flops_g']},{row['thop_params_m']},{row['macs_g']}")


if __name__ == "__main__":
    main()
