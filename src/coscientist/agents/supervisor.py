from __future__ import annotations

from dataclasses import dataclass

from coscientist.schemas.run_state import RunState


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Supervisor:
    state: RunState

    def reserve(self, calls: int, phase: str) -> None:
        if self.state.llm_call_count + calls > self.state.maximum_llm_calls:
            message = f"Budget exhausted before {phase}: requested {calls}, available {self.remaining_calls}."
            self.state.errors.append(message)
            raise BudgetExhausted(message)
        self.state.llm_call_count += calls
        self.state.phase = phase

    @property
    def remaining_calls(self) -> int:
        return self.state.maximum_llm_calls - self.state.llm_call_count
