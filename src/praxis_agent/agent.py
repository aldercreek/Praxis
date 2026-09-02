from dataclasses import dataclass


@dataclass(slots=True)
class Agent:
    name: str = "Praxis"

    def run(self, task: str) -> str:
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("task must not be empty")

        return f"{self.name} received: {normalized_task}"

