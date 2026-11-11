from __future__ import annotations

from pathlib import Path

from praxis_agent.agent import Agent
from praxis_agent.config import ModelConfig, PraxisConfig, load_config
from praxis_agent.context import (
    ContextManager,
)
from praxis_agent.context.summarizer import SUMMARY_INSTRUCTIONS, ModelContextSummarizer
from praxis_agent.events import (
    EventSink,
)
from praxis_agent.mcp import MCPConnections
from praxis_agent.instructions import load_global_instructions
from praxis_agent.model import (
    ModelClient,
    OpenAIModelClient,
)
from praxis_agent.multi_agent.models import ExecutionControl, MultiAgentConfig
from praxis_agent.multi_agent.runtime import AgentRuntime
from praxis_agent.multi_agent.tools import (
    PARENT_INSTRUCTIONS,
    WORKER_INSTRUCTIONS,
    register_parent_tools,
    register_worker_tools,
)
from praxis_agent.prompts import build_instructions
from praxis_agent.runtime import collect_runtime_context
from praxis_agent.skills import create_skill_tool, discover_skills
from praxis_agent.tool_execution import (
    ToolApprovalHandler,
)
from praxis_agent.tools import (
    ToolRegistry,
    create_default_registry,
)


def create_default_agent(
    workspace: Path,
    *,
    event_sink: EventSink | None = None,
    approve_tool_call: ToolApprovalHandler | None = None,
    bypass_approvals: bool = False,
    resume_latest: bool = False,
    resume_session_id: str | None = None,
    session_database_path: Path | None = None,
    enable_multi_agent: bool = True,
    session_store=None,
    mcp: MCPConnections | None = None,
    configuration: PraxisConfig | None = None,
    model: str | None = None,
) -> Agent:
    """使用文件配置快照和内置工具创建默认 Agent。"""
    if resume_latest and resume_session_id is not None:
        raise ValueError("resume_latest and resume_session_id are mutually exclusive")

    from praxis_agent.sessions import (
        SessionNotFoundError,
        SQLiteSessionStore,
    )

    configuration = configuration or load_config()
    model_name = model or configuration.default_model
    collaboration_config = MultiAgentConfig(configuration.max_concurrent) if enable_multi_agent else None
    runtime_context = collect_runtime_context(workspace)
    skills = discover_skills(runtime_context.workspace_root)
    global_instructions = load_global_instructions()
    session_store = session_store or SQLiteSessionStore(session_database_path)
    history: list[object] = []

    if resume_session_id is not None:
        session = session_store.resolve_session(
            resume_session_id,
            workspace=runtime_context.workspace_root,
        )
    elif resume_latest:
        session = session_store.latest_session(runtime_context.workspace_root)
        if session is None:
            raise SessionNotFoundError("no session found for the current workspace")
    else:
        session = session_store.create_session(
            runtime_context.workspace_root,
            model_name,
        )

    if resume_latest or resume_session_id is not None:
        history = session_store.load_conversation_items(session.id)
        model_name = session.model

    config = configuration.resolve(model_name)
    if not config.tool_calling:
        raise ValueError(f"{model_name}: agents require tool_calling")
    summary_config = configuration.resolve(model_name, compression=True)
    instructions = build_instructions(
        runtime_context,
        skills=skills,
        collaboration_instructions=PARENT_INSTRUCTIONS if enable_multi_agent else "",
        context_policy=config.context_policy,
        global_instructions=global_instructions,
    )
    control = ExecutionControl()
    model_client = OpenAIModelClient(
        config, instructions=instructions, execution_control=control
    )
    summary_client = OpenAIModelClient(
        summary_config,
        instructions=SUMMARY_INSTRUCTIONS,
        execution_control=control,
    )
    agent = Agent(
        model_client=model_client,
        model_config=config,
        tool_registry=create_default_registry(
            config.context_policy,
            check_execution=control.check,
            command_shell=runtime_context.command_shell,
        ),
        execution_control=control,
        workspace=runtime_context.workspace_root,
        event_sink=event_sink,
        approve_tool_call=approve_tool_call,
        bypass_approvals=bypass_approvals,
        max_parallel_tools=config.max_parallel_tools,
        session_store=session_store,
        session_id=session.id,
        session_model=session.model,
        context_manager=_create_context_manager(config, instructions, summary_client, summary_config),
        close_callbacks=(model_client.close, summary_client.close),
    )

    from praxis_agent.tools.code import CodeRuntime

    code_runtime = CodeRuntime(
        environment_dir=lambda: Path.home() / ".praxis" / "python" / agent.session_id,
    )
    agent.tool_registry.register(code_runtime.tool("parent"))
    agent.reset_code_worker = code_runtime.reset_session
    agent.code_turn = lambda: code_runtime.agent_turn("parent")
    agent.close_callbacks = (code_runtime.close, *agent.close_callbacks)
    agent.tool_registry.register(
        create_skill_tool(skills, max_chars=config.context_policy.max_tool_output_chars)
    )
    owns_mcp = mcp is None
    try:
        if owns_mcp:
            mcp = MCPConnections()
            mcp.start()
        release_mcp_tools = mcp.register(
            agent.tool_registry, check_execution=control.check
        )
        agent.mcp_catalog = mcp.catalog
    except BaseException:
        try:
            if owns_mcp and mcp is not None:
                mcp.close()
        finally:
            agent.close()
        raise
    agent.close_callbacks = (release_mcp_tools, *((mcp.close,) if owns_mcp else ()), *agent.close_callbacks)

    if not enable_multi_agent:
        agent.restore_history(history)
        return agent

    def create_worker(
        agent_id: str, control: ExecutionControl, sink: EventSink
    ) -> Agent:
        handle = collaboration.handles[agent_id]
        worker_name = handle.model or configuration.subagent_model or model_name
        worker_config = configuration.resolve(worker_name)
        if not worker_config.tool_calling:
            raise ValueError(f"{worker_name}: agents require tool_calling")
        worker_summary_config = configuration.resolve(worker_name, compression=True)
        # Skill discovery belongs to the session project, not a worker's output directory.
        worker_context = collect_runtime_context(handle.workspace)
        worker_instructions = build_instructions(
            worker_context,
            skills=skills,
            collaboration_instructions=WORKER_INSTRUCTIONS,
            context_policy=worker_config.context_policy,
            global_instructions=load_global_instructions(),
        )
        registry = ToolRegistry()
        for tool in create_default_registry(
            worker_config.context_policy,
            check_execution=control.check,
            command_shell=worker_context.command_shell,
        ).tools:
            registry.register(tool)
        registry.register(code_runtime.tool(agent_id))
        registry.register(
            create_skill_tool(
                skills, max_chars=worker_config.context_policy.max_tool_output_chars
            )
        )
        release_worker_tools = mcp.register(registry, check_execution=control.check)
        register_worker_tools(registry, collaboration, agent_id)
        client = OpenAIModelClient(
            worker_config,
            instructions=worker_instructions,
            execution_control=control,
        )
        summary = OpenAIModelClient(
            worker_summary_config,
            instructions=SUMMARY_INSTRUCTIONS,
            execution_control=control,
        )
        return Agent(
            model_client=client,
            model_config=worker_config,
            tool_registry=registry,
            workspace=handle.workspace,
            name=agent_id,
            mcp_catalog=mcp.catalog,
            reset_code_worker=lambda: code_runtime.release_agent(agent_id),
            code_turn=lambda: code_runtime.agent_turn(agent_id),
            event_sink=sink,
            max_parallel_tools=worker_config.max_parallel_tools,
            execution_control=control,
            approve_tool_call=lambda name, arguments: collaboration.request_approval(
                agent_id, name, arguments
            ),
            bypass_approvals=agent.bypass_approvals,
            context_manager=_create_context_manager(
                worker_config, worker_instructions, summary, worker_summary_config
            ),
            receive_messages=lambda accept: collaboration.receive(agent_id, accept),
            team_context=lambda: collaboration.context_item(agent_id),
            close_callbacks=(
                lambda: code_runtime.release_agent(agent_id),
                release_worker_tools,
                client.close,
                summary.close,
            ),
        )

    collaboration = AgentRuntime(
        create_worker,
        collaboration_config,
        workspace=runtime_context.workspace_root,
        approve_tool_call=approve_tool_call,
        live_output_sink=event_sink,
    )
    agent.collaboration = collaboration
    agent.receive_messages = lambda accept: collaboration.receive("parent", accept)
    register_parent_tools(agent.tool_registry, collaboration, agent._pump_worker_events,
                          has_user_input=agent._has_pending_user_input,
                          models=tuple(name for name, entry in configuration.models.items() if entry.tool_calling))
    agent.restore_history(history)
    return agent


def _create_context_manager(
    config: ModelConfig,
    instructions: str,
    summary_client: ModelClient,
    summary_config: ModelConfig,
) -> ContextManager:
    return ContextManager(
        model_limits=config.model_limits,
        policy=config.context_policy,
        instructions=instructions,
        summarizer=ModelContextSummarizer(
            summary_client,
            model_limits=summary_config.model_limits,
            model_config=summary_config,
            safety_ratio=summary_config.context_policy.safety_ratio,
        ),
    )
