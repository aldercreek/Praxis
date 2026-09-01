import argparse
from collections.abc import Sequence

from praxis_agent.agent import Agent


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
        print(Agent().run(task))
    except ValueError as error:
        print(f"error: {error}")
        return 1

    return 0

