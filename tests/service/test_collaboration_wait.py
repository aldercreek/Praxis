"""A durable HTTP input wakes an agent blocked in the collaboration tool."""
import json
from threading import Condition, Event

from support.service import Harness, answer
from praxis_agent.model import FunctionCall, ModelResponse
from praxis_agent.multi_agent.models import MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.multi_agent.tools import register_parent_tools


def test_http_input_is_consumed_in_same_turn_after_wait(tmp_path):
    waiting = Event()
    captured = []

    class ObservedCondition(Condition):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)

    class Model:
        def create_response(self, items, tools, **kwargs):
            captured.append(str(items))
            if len(captured) == 1:
                call = FunctionCall("wait", "wait_agents", json.dumps({"timeout": 60}))
                return ModelResponse(({"type": "function_call", "call_id": call.call_id,
                                       "name": call.name, "arguments": call.arguments},), "", (call,))
            assert "revised live instruction" in str(items)
            assert "user_input" in str(items)
            return answer("accepted revision")

    h = Harness(tmp_path, [Model()])
    base = h.manager.factory

    def factory(*args, **kwargs):
        agent = base(*args, **kwargs)
        runtime = AgentRuntime(None, MultiAgentConfig(), workspace=tmp_path)
        runtime.condition = ObservedCondition()
        agent.collaboration = runtime
        agent.receive_messages = lambda accept: runtime.receive("parent", accept)
        register_parent_tools(agent.tool_registry, runtime, agent._pump_worker_events,
                              has_user_input=agent._has_pending_user_input)
        return agent

    h.manager.factory = factory
    try:
        session = h.create()
        h.submit(session, "start waiting")
        assert waiting.wait(3)
        h.submit(session, "revised live instruction")
        snapshot = h.settled(session)
        assert len(captured) == 2
        assert snapshot["pending_inputs"] == []
        user_messages = [m["content"] for m in snapshot["messages"] if m["role"] == "user"]
        assert user_messages == ["start waiting", "revised live instruction"]
    finally:
        h.close()
