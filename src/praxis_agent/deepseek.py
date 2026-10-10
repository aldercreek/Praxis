from openai.types.responses import Response


class DeepSeekResponseNormalizer:
    """处理 DeepSeek 网关把同一段 reasoning 重复附在工具调用上的情况。"""

    def normalize(self, response: Response) -> Response:
        """保留独立 reasoning；只移除后续调用中完全相同的冗余副本。"""
        reasoning_texts: set[str] = set()
        output = []
        for item in response.output:
            payload = item.model_dump(mode="json")
            if item.type == "reasoning":
                reasoning_texts = set()
                for field in ("summary", "content"):
                    blocks = payload.get(field) or []
                    text = "".join(block.get("text", "") for block in blocks)
                    if text:
                        reasoning_texts.add(text)
            elif item.type == "function_call":
                text = payload.get("reasoning_content")
                if isinstance(text, str) and text in reasoning_texts:
                    item = item.model_copy(deep=True)
                    item.model_extra.pop("reasoning_content", None)
            output.append(item)
        return response.model_copy(update={"output": output})
