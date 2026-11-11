"""Run complex synthetic tasks through real Praxis agents and a live model.

Example (credentials stay in environment):
  python evals/multi_agent/run.py --output D:/temp/praxis-eval
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
from queue import SimpleQueue
from threading import Lock, Timer
from time import monotonic

from praxis_agent.events import ChildAgentEvent, ModelRequestCompleted, ModelRequestStarted, ToolCallPrepared, ToolFinished, ToolStarted
from praxis_agent.execution import Run
from praxis_agent.factory import create_default_agent
from praxis_agent.config import load_config
from praxis_agent.mcp import MCPConnections

from scenarios import SCENARIOS


class Trace:
    def __init__(self, path, scenario, deadline, max_calls):
        self.path = path
        self.scenario = scenario
        self.deadline = deadline
        self.max_calls = max_calls
        self.lock = Lock()
        self.parent = None
        self.calls = {}
        self.spawned = set()
        self.peer_messages = 0
        self.waits = {}
        self.wait_reasons = {}
        self.inherited_spawns = 0
        self.inherited_ids = set()
        self.worker_requests = set()
        self.inherited_seen = set()
        self.closed = 0
        self.models = set()
        self.tokens = dict(input_tokens=0, output_tokens=0, total_tokens=0)
        self.requests = 0
        self.injected = False
        self.tool_failures = []

    def emit(self, event):
        agent = "parent"
        if isinstance(event, ChildAgentEvent):
            agent, event = event.agent_id, event.event
        record = dict(time=round(monotonic(), 3), agent=agent, event=type(event).__name__)
        if isinstance(event, (ToolCallPrepared, ToolStarted, ToolFinished)):
            record["call_id"] = event.call_id
        with self.lock:
            if isinstance(event, ModelRequestStarted):
                self.requests += 1
                if agent != "parent" and agent not in self.worker_requests:
                    self.worker_requests.add(agent)
                    record["first_worker_request"] = True
                    record["context_marker_present"] = "ORCHID-714" in str(event.input_items)
                    if record["context_marker_present"]:
                        self.inherited_seen.add(agent)
                record["request_number"] = event.request_number
                if self.requests > self.max_calls or monotonic() > self.deadline:
                    raise RuntimeError("evaluation request/time limit reached")
            elif isinstance(event, ModelRequestCompleted) and event.response:
                response = event.response
                if response.model:
                    self.models.add(response.model)
                if response.usage:
                    for key, value in asdict(response.usage).items():
                        self.tokens[key] += value
                record.update(model=response.model, text=response.output_text)
            elif isinstance(event, ToolCallPrepared):
                self.calls[agent, event.call_id] = (event.tool_name, event.arguments)
                record.update(tool=event.tool_name, arguments=event.arguments)
            elif isinstance(event, ToolStarted):
                name, arguments = self.calls.get((agent, event.call_id), (event.tool_name, {}))
                if name == "wait_agents":
                    self.waits[agent] = self.waits.get(agent, 0) + 1
                    if agent == "parent" and self.scenario.followup and not self.injected:
                        self.injected = True
                        self.parent.user_input.put(self.scenario.followup)
                        self.parent.collaboration.wake_waiters()
                        record["injected_user_input"] = self.scenario.followup
            elif isinstance(event, ToolFinished):
                name, arguments = self.calls.get((agent, event.call_id), (event.tool_name, {}))
                record.update(tool=name, success=event.result.success, output=event.result.output)
                if not event.result.success:
                    self.tool_failures.append(dict(agent=agent, tool=name, output=event.result.output[:600]))
                elif name == "spawn_agent":
                    try:
                        spawned_id = json.loads(event.result.output)["agent_id"]
                        self.spawned.add(spawned_id)
                        if arguments.get("inherit_context"):
                            self.inherited_spawns += 1
                            self.inherited_ids.add(spawned_id)
                    except (ValueError, KeyError):
                        pass
                elif name == "wait_agents":
                    reason = json.loads(event.result.output)["reason"]
                    reasons = self.wait_reasons.setdefault(agent, {})
                    reasons[reason] = reasons.get(reason, 0) + 1
                elif name == "send_message" and agent != "parent" and arguments.get("target", "parent") != "parent":
                    self.peer_messages += 1
                elif name == "close_agent":
                    self.closed += 1
            else:
                return
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_case(name, args):
    case = args.output / name
    workspace = case / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    scenario = SCENARIOS[name](workspace)
    (case / "task.txt").write_text(scenario.task, encoding="utf-8")
    started = monotonic()
    trace = Trace(case / "trace.jsonl", scenario, started + args.seconds, args.max_calls)
    # Empty MCP config prevents the benchmark from connecting to user services.
    mcp = MCPConnections(case / "disabled-mcp.json")
    parent = create_default_agent(workspace, event_sink=trace, bypass_approvals=True,
                                  configuration=args.configuration, model=args.model,
                                  session_database_path=case / "session.sqlite3", mcp=mcp)
    parent.user_input = SimpleQueue()
    trace.parent = parent
    parent.restore_history([
        {"role": "user", "content": "Earlier agreed constraints: standard library only; all money in integer cents; "
         "do not change public APIs without explicit permission. CONTEXT_CODE=ORCHID-714. "
         "This code should be recovered by a worker through inheritance, not repeated in its task."},
        {"role": "assistant", "content": "I will preserve these constraints in subsequent work."},
    ])
    scope = Run("live multi-agent evaluation")
    timer = Timer(args.seconds, scope.cancel)
    timer.daemon = True
    timer.start()
    error = None
    final = ""
    try:
        task = scenario.task + (
            "\nUse the actual spawn_agent, send_message and wait_agents tools for collaboration, not simulated agents. "
            "Use this workspace as working_directory for workers; coordinate file ownership. "
            "Use short task_summary labels so peers can find each other in the team table. "
            "Use run_command with the absolute interpreter " + os.sys.executable + " for scripts/tests; "
            "prefer short tool calls over large inline shell strings. No external services, network, installing packages, "
            "or reading files outside this workspace. Create scripts to compute, then run them. "
            "Do not spend turns acknowledging messages without substantive progress. "
            "Consolidate result.json and supporting evidence before finalizing."
        )
        final = scope.execute(lambda: parent.run(task))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        timer.cancel()
        parent.close()
        mcp.close()
    (case / "final.txt").write_text(final, encoding="utf-8")
    try:
        data = json.loads((workspace / "result.json").read_text(encoding="utf-8-sig"))
        checks = scenario.check(data)
    except Exception as exc:
        checks = {"artifact_readable": False}
        if error is None:
            error = f"grader: {type(exc).__name__}: {exc}"
    inherited_workers = trace.inherited_seen & trace.inherited_ids
    collaboration = dict(at_least_three_workers=len(trace.spawned) >= 3,
                         at_least_two_peer_messages=trace.peer_messages >= 2,
                         inherited_context_reached_worker=bool(inherited_workers),
                         parent_used_wait=trace.waits.get("parent", 0) > 0)
    if scenario.followup:
        collaboration["live_user_update_injected"] = trace.injected
        collaboration["user_input_woke_parent_wait"] = trace.wait_reasons.get("parent", {}).get("user_input", 0) > 0
    if name == "schedule":
        collaboration["member_closed_and_replaced"] = trace.closed >= 1 and len(trace.spawned) >= 4
    report = dict(case=name, elapsed_seconds=round(monotonic() - started, 2),
                  requested_model=args.model, observed_models=sorted(trace.models),
                  requests=trace.requests, tokens=trace.tokens, workers=len(trace.spawned),
                  peer_messages=trace.peer_messages, waits=trace.waits,
                  wait_reasons=trace.wait_reasons,
                  inherited_spawns=trace.inherited_spawns,
                  inherited_context_workers=sorted(inherited_workers),
                  correctness=checks, collaboration=collaboration,
                  passed=error is None and all(checks.values()) and all(collaboration.values()),
                  error=error, tool_failures=trace.tool_failures)
    (case / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Model configuration file")
    parser.add_argument("--model", help="Model key from the configuration")
    parser.add_argument("--cases", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--max-calls", type=int, default=120)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.configuration = load_config(args.config)
    args.model = args.model or args.configuration.default_model
    # Sessions/kernels and logs belong to this evaluation, not a user's live service.
    os.environ["PRAXIS_DATA_DIR"] = str(args.output / "state")
    reports = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_case, name, args): name for name in args.cases}
        for future in as_completed(futures):
            try:
                reports.append(future.result())
            except Exception as exc:
                reports.append(dict(case=futures[future], passed=False, error=f"{type(exc).__name__}: {exc}"))
                print(json.dumps(reports[-1]), flush=True)
            (args.output / "summary.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
