from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelLimits:
    """描述模型或 Provider 声明的上下文硬限制。"""

    context_window_tokens: int
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be at least 1")
        if self.max_input_tokens is not None and self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be at least 1")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")



@dataclass(frozen=True, slots=True)
class ModelInputCapabilities:
    """Native model inputs; the active API can impose narrower restrictions."""

    image_mime_types: tuple[str, ...] = ()
    pdf: bool = False
    audio: bool = False
