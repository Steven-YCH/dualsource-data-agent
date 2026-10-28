"""LangChain chat-model adapter for the project's Foundry wire protocol.

Optional: imported only by the opt-in LangChain harness. The existing client owns
URL/header/model compatibility; LangChain owns callback/model run lifecycle.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


def wire_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Preserve tool IDs/arguments through the Foundry OpenAI-compatible API."""
    out = []
    roles = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}
    for message in messages:
        row: dict[str, Any] = {"role": roles[message.type], "content": message.content}
        if message.type == "tool":
            row["tool_call_id"] = message.tool_call_id
        if isinstance(message, AIMessage) and message.tool_calls:
            row["tool_calls"] = [
                {"id": call["id"], "type": "function", "function": {
                    "name": call["name"], "arguments": json.dumps(call["args"]),
                }} for call in message.tool_calls
            ]
        out.append(row)
    return out


class FoundryChatModel(BaseChatModel):
    """Real BaseChatModel callbacks without changing Foundry deployments/auth."""
    client: Any = Field(exclude=True, repr=False)
    model_name: str

    @property
    def _llm_type(self) -> str:
        return "dualsource-foundry"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "provider": "foundry",
                "reasoning_effort_policy": getattr(self.client, "reasoning_effort", None)}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any):
        return self.bind(
            tools=[convert_to_openai_tool(tool) for tool in tools],
            tool_choice=tool_choice,
            **kwargs,
        )

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> ChatResult:
        if stop:
            raise ValueError("FoundryChatModel does not support stop sequences")
        result = self.client.chat(
            model=self.model_name, messages=wire_messages(messages),
            tools=kwargs.get("tools"), tool_choice=kwargs.get("tool_choice"),
            temperature=0.1, max_tokens=1400,
        )
        raw = result.get("raw") or {}
        usage = raw.get("usage") or {}
        tool_calls = []
        for call in result.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("Foundry tool arguments must be an object")
            tool_calls.append({"name": fn["name"], "args": args, "id": call["id"], "type": "tool_call"})
        usage_metadata = None
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            usage_metadata = {
                "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                "total_tokens": usage.get("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"]),
            }
        reasoning_settings = getattr(self.client, "reasoning_settings", None)
        message = AIMessage(
            content=result.get("content") or "", tool_calls=tool_calls,
            usage_metadata=usage_metadata,
            response_metadata={"model_name": self.model_name, "provider": "foundry", "id": raw.get("id"),
                               "reasoning": reasoning_settings(self.model_name) if callable(reasoning_settings) else {}},
        )
        return ChatResult(generations=[ChatGeneration(message=message)], llm_output={"token_usage": usage, "model_name": self.model_name})
