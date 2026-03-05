from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class SwitchState(Enum):
    IDLE = "idle"
    SWITCHING = "switching"
    READY = "ready"
    FAILED = "failed"


@dataclass
class MachineSwitchState:
    state: SwitchState = SwitchState.IDLE
    error: Optional[str] = None


def init_machine_states(machines) -> Dict[str, Dict[str, Optional[str]]]:
    return {
        str(m.get("machine_id") or f"machine-{idx}"): {"state": SwitchState.IDLE.value, "error": None}
        for idx, m in enumerate(machines)
    }
