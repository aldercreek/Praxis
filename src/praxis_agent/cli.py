import argparse
from collections.abc import Sequence
from pathlib import Path

from praxis_agent.agent import AgentError, create_default_agent
from praxis_agent.model import ModelClientError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="praxis",
        description="Run the Praxis coding agent.",
    )
    parser.add_argument("task", nargs="*", help="Task for the agent")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = " ".join(args.task).strip()

    if not task:
        task = input("praxis> ").strip()

    try:
        agent = create_default_agent(Path.cwd())
        print(agent.run(task))
    except (AgentError, ModelClientError, ValueError) as error:
        print(f"error: {error}")
        return 1

    return 0
