"""Model identity used by runtime transactions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    """An immutable identity for one loaded model artifact."""

    model_id: str
    revision: str
    backend: str
    fingerprint: str

    def __post_init__(self) -> None:
        for field_name in ("model_id", "revision", "backend", "fingerprint"):
            value = getattr(self, field_name)
            if not value or value.isspace():
                raise ValueError(f"{field_name} must not be empty")
