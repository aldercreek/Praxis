"""Model failures shared by the request loop and the independent summarizer."""

import re


class ModelClientError(RuntimeError):
    """A model request or response failed; code retains provider error identity."""

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


class PromptTooLongError(ModelClientError):
    """The provider rejected the input length, not a network or output-limit error."""

    def __init__(self, message: str, *, code: str | None = None,
                 requested_tokens: int | None = None, limit_tokens: int | None = None):
        super().__init__(message, code=code)
        self.requested_tokens = requested_tokens
        self.limit_tokens = limit_tokens

    @property
    def required_reduction(self) -> int | None:
        if self.requested_tokens is None or self.limit_tokens is None:
            return None
        if self.limit_tokens < 1 or self.requested_tokens < self.limit_tokens:
            return None
        return self.requested_tokens - self.limit_tokens + 1


class IncompleteResponseError(ModelClientError):
    """No complete usable response was received; the request can be regenerated."""


class OutputTruncatedError(IncompleteResponseError):
    """Generation reached its output limit before producing a complete response."""


class SummaryError(ModelClientError):
    """Summarization failed; the main request must not retry it as a PTL."""


class SummaryTooLongError(SummaryError):
    """A complete candidate is retained for a bounded refinement attempt."""

    def __init__(self, summary: str, required_reduction: int):
        super().__init__("compacted context still exceeds the budget; no operation was saved")
        self.summary = summary
        self.required_reduction = required_reduction


def model_request_error(message: str, code: str | None = None, *, details=None) -> ModelClientError:
    input_codes = {"context_length_exceeded", "context_window_exceeded", "input_too_long", "prompt_too_long"}
    input_message = re.search(
        r"(?:prompt|input)(?:\s+(?:is|length|tokens?)){0,3}\s+too\s+long"
        r"|maximum context length|context (?:window|length).{0,80}exceed"
        r"|(?:input|prompt).{0,60}exceeds? .{0,30}(?:limit|maximum|context)",
        message, re.IGNORECASE,
    )
    if code not in input_codes and not input_message:
        return ModelClientError(message, code=code)
    requested, limit = _context_token_counts(message, details)
    return PromptTooLongError(message, code=code, requested_tokens=requested, limit_tokens=limit)


def _context_token_counts(message: str, details) -> tuple[int | None, int | None]:
    """Only use paired counts with the same basis (input or total request tokens)."""
    if isinstance(details, dict):
        error = details.get("error", details)
        if isinstance(error, dict):
            for data in (error, error.get("details")):
                if not isinstance(data, dict):
                    continue
                for used_key, limit_key in (("input_tokens", "max_input_tokens"),
                                            ("requested_tokens", "max_context_tokens")):
                    used, limit = data.get(used_key), data.get(limit_key)
                    if type(used) is int and type(limit) is int and used >= 0 and limit > 0:
                        return used, limit
    number = r"(\d[\d,]*)"
    # Providers report either a total requested count or an explicit input comparison.
    maximum = re.search(r"maximum context length (?:is|of)\s*" + number, message, re.I)
    requested = re.search(r"(?:you\s+)?requested\s*" + number + r"\s*tokens", message, re.I)
    if maximum and requested:
        return int(requested[1].replace(",", "")), int(maximum[1].replace(",", ""))
    comparison = re.search(
        r"(?:prompt|input)\s+(?:is\s+)?too long\s*:\s*" + number
        + r"\s*tokens\s*>\s*" + number, message, re.I,
    )
    if comparison:
        return tuple(int(value.replace(",", "")) for value in comparison.groups())
    return None, None
