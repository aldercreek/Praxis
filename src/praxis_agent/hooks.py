"""Ordered tool hooks with explicit argument and output replacement."""
import inspect
import json
from copy import deepcopy
from dataclasses import replace
from fnmatch import fnmatchcase

from praxis_agent.extension_loading import extension_modules, marked_functions
from praxis_agent.hook_types import HookDecision, HookOutput, HookResult
from praxis_agent.sdk import HOOK_EVENTS, HookMetadata, _HOOK_ATTRIBUTE


class HookError(ValueError):
    pass


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


class HookManager:
    def __init__(self, configurations=()):
        handlers = []
        with extension_modules(configurations) as modules:
            for config, module in modules:
                for function in marked_functions(module, _HOOK_ATTRIBUTE):
                    label = f"{config.id}.{function.__name__}"
                    metadata = getattr(function, _HOOK_ATTRIBUTE)
                    if not isinstance(metadata, HookMetadata) or metadata.event not in HOOK_EVENTS:
                        raise ValueError(f"Hook {label}: invalid metadata")
                    if (inspect.iscoroutinefunction(function) or inspect.isgeneratorfunction(function)
                            or inspect.isasyncgenfunction(function)):
                        raise ValueError(f"Hook {label}: ordinary synchronous functions are required")
                    try:
                        inspect.signature(function).bind(object())
                    except TypeError as error:
                        raise ValueError(f"Hook {label}: expected one event argument") from error
                    handlers.append((label, metadata, function))
        self.handlers = tuple(handlers)

    def _matching(self, event):
        return ((label, function) for label, metadata, function in self.handlers
                if metadata.event == event.event and any(fnmatchcase(event.tool_name, p) for p in metadata.tools))

    def handles_results(self, tool_name):
        return any(metadata.event != "before_tool_call" and any(fnmatchcase(tool_name, p) for p in metadata.tools)
                   for _, metadata, _ in self.handlers)

    def before(self, event):
        current = event
        for label, function in self._matching(event):
            try:
                decision = function(deepcopy(current))
                if decision is None:
                    continue
                if (not isinstance(decision, HookDecision) or type(decision.allowed) is not bool
                        or type(decision.requires_approval) is not bool):
                    raise ValueError("expected HookDecision or None")
                if not decision.allowed:
                    if not isinstance(decision.reason, str) or not decision.reason.strip():
                        raise ValueError("denial requires a reason")
                    return current, f"Hook {label} denied tool execution: {decision.reason}"
                arguments = current.arguments
                if decision.arguments is not None:
                    if not isinstance(decision.arguments, dict):
                        raise ValueError("replacement arguments must be a JSON object")
                    arguments = _json_copy(decision.arguments)
                current = replace(current, arguments=arguments,
                                  requires_approval=current.requires_approval or decision.requires_approval,
                                  approval_requested=current.approval_requested or decision.requires_approval,
                                  approval_hooks=current.approval_hooks + ((label,) if decision.requires_approval else ()))
            except Exception as error:
                # Do not echo arbitrary callback exceptions, which may contain payload secrets.
                raise HookError(f"Hook {label} failed before execution ({type(error).__name__})") from error
        return current, None

    def after(self, event, result, duration_ms):
        snapshot = HookResult(result.success, result.output, result.structured_output, result.model_content)
        current = replace(event, event="after_tool_call" if result.success else "on_tool_error",
                          original_result=snapshot, result=snapshot, duration_ms=duration_ms)
        output = result
        for label, function in self._matching(current):
            try:
                update = function(deepcopy(current))
                if update is None:
                    continue
                if not isinstance(update, HookOutput) or not isinstance(update.output, str):
                    raise ValueError("expected HookOutput or None")
                if update.structured_output is not None and not isinstance(update.structured_output, dict):
                    raise ValueError("structured_output must be an object")
                if update.model_content is not None and (not isinstance(update.model_content, list)
                        or any(not isinstance(block, dict) for block in update.model_content)):
                    raise ValueError("model_content must be a list of objects")
                data = _json_copy([update.structured_output, update.model_content])
                output = replace(output, output=update.output, structured_output=data[0], model_content=data[1],
                                 file_change=None, presentation={**(output.presentation or {}), "result_processing": "modified"})
                current = replace(current, result=HookResult(output.success, output.output,
                                                            output.structured_output, output.model_content))
            except Exception:
                # Preserve execution status/callbacks, but never fall back to unfiltered content.
                return replace(result, output=f"Hook {label} failed while processing the result. "
                               "Tool execution has already finished; do not retry automatically.",
                               structured_output=None, model_content=None, file_change=None,
                               presentation={**(result.presentation or {}), "result_processing": "error"})
        return output
