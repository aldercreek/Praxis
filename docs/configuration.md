# Model configuration

Praxis requires a YAML file. Select it with `praxis --config /absolute/path/praxis.yaml`, `PRAXIS_CONFIG`, or the default `<PRAXIS_DATA_DIR>/praxis.yaml` (`~/.praxis/praxis.yaml`). The desktop Settings dialog selects the same file. No provider or model catalog is bundled into the loader; no old model environment settings are read.

```yaml
default_model: coding
models:
  coding:
    protocol: responses
    base_url: https://gateway.example/v1
    model: upstream-coder
    api_key_env: CODING_API_KEY
    context_window: 128000
    max_output_tokens: 8192
    capabilities:
      tool_calling: true
      image_input: false
      pdf_input: false
  summary:
    protocol: chat_completions
    base_url: https://another-gateway.example/v1
    model: upstream-summary
    api_key_env: SUMMARY_API_KEY
    context_window: 64000
    max_output_tokens: 4096
    capabilities:
      tool_calling: false

# Optional: without these selections, compression uses the current agent's model,
# and workers inherit their parent model.
compression:
  model: summary
subagents:
  model: coding
  max_concurrent: 4

timeout_seconds: 120
max_retries: 2
max_parallel_tools: 4
```

## Required fields and defaults

- `models` must contain at least one entry. Every entry requires `protocol`, `base_url`, `model`, `context_window`, and `max_output_tokens`. These are explicit local operating limits, not automatically discovered provider metadata. Keep them within your gateway's supported limits.
- `default_model` selects a configuration key, not an upstream model name. It can be omitted only when there is one model. `praxis --model coding` selects a model for a new session. A restored session uses its saved configuration key; unknown keys fail explicitly.
- Supported protocols are `responses` and `chat_completions`. Adding another endpoint with one of these protocols only requires configuration. New protocols require an adapter implementation.
- `api_key_env` references an environment variable in the service process. An absent reference means no Authorization header; a specified but missing/empty variable fails when that model is selected. Keys are never stored in the config or returned by the settings API.
- `max_input_tokens` is optional. `max_output_tokens` is sent on every request and bounds both main generation and summary generation. Requests for a smaller summary limit can lower it. The main context reserves that configured output budget.
- `capabilities.tool_calling` defaults to true; models used by agents must support tools. Image/PDF flags default to false. PDF input is only available through Responses. Neither adapter supports audio input.
- Per-model `parameters.reasoning_effort` and `parameters.temperature` are optional. Omitted parameters are not sent. Only specify values supported by that model and protocol. `compression.parameters` can override these parameters for summaries without changing agent calls; there is no hardcoded `reasoning_effort: none`.
- `timeout_seconds`, `max_retries`, `max_parallel_tools`, and `subagents.max_concurrent` default to 120, 2, 4, and 4 respectively. These are generic program defaults, not provider presets.
- `context` can override generic `ContextPolicy` fields such as `safety_ratio`, `auto_compact_ratio`, `summary_output_tokens`, and tool/file/history character budgets. Invalid types, unknown fields, inconsistent limits, and duplicate YAML keys fail during loading.

## Compression and workers

A worker selects its explicit `spawn_agent.model`, then `subagents.model`, then the parent model. `compression.model` applies to both the main agent and workers. Without it, each agent uses its own model for compression. Summaries use dedicated instructions and no tools. Chunking uses the compression model's own capacity, including when it is smaller than the main model.

The service loads a validated snapshot and injects it into agents. Multiple sessions can use different model entries simultaneously. A file edit does not mutate an existing snapshot. Reload using `--config`, desktop Save settings, or `PUT /api/settings` with `config_path`; reload requires all sessions to be idle and disposes their clients. It does not modify process model environment variables. The settings API accepts only `config_path` and `bypass_approvals`; it no longer accepts inline model credentials or addresses.

Existing history is not converted between protocols. Keep the protocol of a configuration key stable when restoring sessions. Use a new configuration key and new session to change protocols. Both clients and service must use protocol version 2; older running services must be stopped before starting this version.
