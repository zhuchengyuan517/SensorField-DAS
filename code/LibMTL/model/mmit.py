from LibMTL.model.das_multimodal_net import (
    DASMultiModalNet,
    DASLossOutput,
    FocalLoss,
    compute_total_loss,
)


class MMIT(DASMultiModalNet):
    """MMIT: multimodal multimodal identification transformer-like network for DAS event recognition."""


def build_mmit(**kwargs) -> MMIT:
    return MMIT(**kwargs)


__all__ = [
    "MMIT",
    "build_mmit",
    "FocalLoss",
    "compute_total_loss",
    "DASLossOutput",
]
