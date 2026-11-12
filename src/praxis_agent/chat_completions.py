"""Chat Completions wire adapter for Praxis conversation items."""
from httpx2 import TransportError
from openai import Omit

from praxis_agent.model_errors import IncompleteResponseError, ModelClientError, OutputTruncatedError
from openai import APIConnectionError


def _content(value):
    if isinstance(value, str):
        return value
    blocks = []
    for block in value:
        kind = block.get("type")
        if kind in {"input_text", "output_text"}:
            blocks.append({"type": "text", "text": block["text"]})
        elif kind == "input_image":
            image = {"url": block["image_url"]}
            if "detail" in block:
                image["detail"] = block["detail"]
            blocks.append({"type": "image_url", "image_url": image})
        else:
            raise ModelClientError(f"Unsupported Chat Completions content: {kind}")
    return blocks


def chat_messages(instructions, items, capabilities):
    from praxis_agent.model import _response_input

    messages = [{"role": "system", "content": instructions}]
    for item in items:
        data = _response_input(item, capabilities)
        kind = data.get("type", "message")
        if kind == "function_call":
            call = {"id": data["call_id"], "type": "function",
                    "function": {"name": data["name"], "arguments": data["arguments"]}}
            if messages[-1].get("role") != "assistant":
                messages.append({"role": "assistant", "content": None})
            messages[-1].setdefault("tool_calls", []).append(call)
        elif kind == "function_call_output":
            output = data["output"]
            # Chat tool messages are text-only. Images are a following user attachment.
            if isinstance(output, list):
                blocks = _content(output)
                text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
                images = [b for b in blocks if b["type"] == "image_url"]
            else:
                text, images = output, []
            messages.append({"role": "tool", "tool_call_id": data["call_id"], "content": text})
            if images:
                # Keep all tool results adjacent before appending their attachments below.
                messages[-1]["_images"] = images
        elif kind == "reasoning":
            if data.get("encrypted_content"):
                raise ModelClientError("Encrypted Responses reasoning cannot be sent to Chat Completions")
            text = "".join(b.get("text", "") for b in data.get("content", []))
            if text:
                messages.append({"role": "assistant", "content": None, "reasoning_content": text})
        elif kind == "message":
            content = _content(data.get("content", ""))
            if data["role"] == "assistant" and messages[-1].get("role") == "assistant" and messages[-1].get("content") is None:
                messages[-1]["content"] = content
            else:
                messages.append({"role": data["role"], "content": content})
        else:
            raise ModelClientError(f"Unsupported conversation item: {kind}")
    result, images = [], []
    for message in messages:
        if message["role"] != "tool" and images:
            result.append({"role": "user", "content": images})
            images = []
        images.extend(message.pop("_images", []))
        result.append(message)
    if images:
        result.append({"role": "user", "content": images})
    return result


async def create_chat_response(client, config, instructions, items, tools, on_delta, output_limit):
    from praxis_agent.model import FunctionCall, ModelResponse, TokenUsage

    options = dict(config.parameters)
    if not config.api_key:
        options["extra_headers"] = {"Authorization": Omit()}
    if tools:
        options["tools"] = [{"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}
                            for tool in tools]
    stream = await client.chat.completions.create(
        model=config.model, messages=chat_messages(instructions, items, config.input_capabilities),
        stream=True, stream_options={"include_usage": True}, max_completion_tokens=output_limit,
        **options,
    )
    texts, reasoning, calls = [], [], {}
    finish = response_id = model = usage = None
    try:
        async for chunk in stream:
            response_id, model = chunk.id, chunk.model
            if chunk.usage:
                usage = TokenUsage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens, chunk.usage.total_tokens)
            for choice in chunk.choices:
                if choice.index != 0:
                    raise ModelClientError("Expected one completion choice")
                delta = choice.delta
                if delta.content:
                    texts.append(delta.content)
                    if on_delta:
                        on_delta(delta.content)
                if text := getattr(delta, "reasoning_content", None):
                    reasoning.append(text)
                for part in delta.tool_calls or ():
                    call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                    if part.id:
                        call["id"] += part.id
                    if part.function:
                        call["name"] += part.function.name or ""
                        call["arguments"] += part.function.arguments or ""
                if choice.finish_reason:
                    finish = choice.finish_reason
    except (TransportError, APIConnectionError) as error:
        raise IncompleteResponseError("model stream interrupted") from error
    finally:
        await stream.close()
    if finish == "length":
        raise OutputTruncatedError("model response exceeded output limit", code=finish)
    if finish == "content_filter":
        raise ModelClientError("model response was filtered", code=finish)
    if finish not in {"stop", "tool_calls"}:
        raise IncompleteResponseError("model stream ended without a completed response")
    if any(not c["id"] or not c["name"] for c in calls.values()) or (finish == "tool_calls" and not calls):
        raise IncompleteResponseError("model returned incomplete tool calls")
    function_calls = tuple(FunctionCall(c["id"], c["name"], c["arguments"]) for _, c in sorted(calls.items()))
    output = []
    if reasoning:
        output.append({"type": "reasoning", "content": [{"type": "reasoning_text", "text": "".join(reasoning)}]})
    text = "".join(texts)
    if text:
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    output.extend({"type": "function_call", "call_id": c.call_id, "name": c.name, "arguments": c.arguments}
                  for c in function_calls)
    return ModelResponse(tuple(output), text.strip(), function_calls, response_id, model, usage)
