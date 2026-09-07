"""Local LibMTL package used by the DAS experiment scripts."""

try:
    from LibMTL.trainer import Trainer
except ModuleNotFoundError as exc:  # pragma: no cover - optional training stack
    _trainer_import_error = exc

    class Trainer:  # type: ignore[no-redef]
        """Lazy placeholder when optional LibMTL trainer dependencies are absent."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "LibMTL.Trainer requires optional dependencies that are not installed in this environment."
            ) from _trainer_import_error

__all__ = ["Trainer"]
