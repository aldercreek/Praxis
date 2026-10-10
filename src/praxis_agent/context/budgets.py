"""按上下文窗口选择默认预算；显式配置由调用方覆盖。"""

_FIELDS = (
    "desired_output_tokens", "summary_target_tokens", "summary_max_output_tokens",
    "max_tool_output_chars", "max_file_read_chars",
    "max_history_read_chars",
)

# 窗口上界、输出预留、摘要目标、摘要生成上限、工具/文件字符、历史字符。
_TIERS = (
    (32_000,  (4_000, 2_000, 4_000, 16_000, 16_000, 8_000)),
    (64_000,  (8_000, 3_000, 6_000, 24_000, 24_000, 12_000)),
    (128_000, (16_000, 4_000, 8_000, 32_000, 32_000, 16_000)),
    (256_000, (24_000, 6_000, 12_000, 48_000, 48_000, 24_000)),
    (512_000, (32_000, 8_000, 16_000, 64_000, 64_000, 32_000)),
    (float("inf"), (32_000, 12_000, 24_000, 96_000, 96_000, 48_000)),
)


def context_budgets(window_tokens: int) -> dict[str, int]:
    """返回独立字典；没有模型信息的独立工具使用最小档。"""
    values = next(values for ceiling, values in _TIERS if window_tokens <= ceiling)
    return dict(zip(_FIELDS, values))


DEFAULT_BUDGETS = context_budgets(32_000)
