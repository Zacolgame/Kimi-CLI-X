"""L5 Programmatic Memory: workflows, tasks, triggers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from kimix.memory.types import MemoryEntry, MemoryType


class TriggerType(Enum):
    SCHEDULE = "schedule"   # cron-like or interval
    EVENT = "event"         # fired by named event


@dataclass(slots=True)
class Trigger:
    """Workflow trigger: schedule or event."""

    trigger_type: TriggerType
    condition: str                    # cron expression, interval seconds, or event name
    payload: dict[str, Any] = field(default_factory=dict)
    last_fired: float = 0.0
    enabled: bool = True
    _interval: float = field(init=False, repr=False, compare=False, default=0.0)
    _is_interval: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(self) -> None:
        if self.trigger_type == TriggerType.SCHEDULE:
            try:
                self._interval = float(self.condition)
                self._is_interval = True
            except ValueError:
                self._is_interval = False

    def should_fire(self, now: float | None = None) -> bool:
        """For SCHEDULE triggers: check if interval has passed."""
        if not self._is_interval or not self.enabled:
            return False
        now = now or time.time()
        return (now - self.last_fired) >= self._interval

    def mark_fired(self, now: float | None = None) -> None:
        self.last_fired = now or time.time()


@dataclass(slots=True)
class Task:
    """Unit of work inside a workflow."""

    name: str
    status: str = "pending"           # pending | running | completed | failed
    payload: dict[str, Any] = field(default_factory=dict)
    result: Any = None


@dataclass(slots=True)
class Workflow:
    """Named workflow with triggers and tasks."""

    name: str
    triggers: list[Trigger] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def add_trigger(self, trigger: Trigger) -> None:
        self.triggers.append(trigger)

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)

    def pending_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status == "pending"]

    def to_memory_entry(self) -> MemoryEntry:
        return MemoryEntry(
            content=f"WORKFLOW: {self.name} ({len(self.tasks)} tasks, {len(self.triggers)} triggers)",
            memory_type=MemoryType.WORKFLOW,
            tags=["workflow", self.name],
            metadata={
                "workflow_name": self.name,
                "task_count": len(self.tasks),
                "trigger_count": len(self.triggers),
                **self.metadata,
            },
        )


class ProgrammaticMemory:
    """L5 memory: register and run workflows."""

    def __init__(self) -> None:
        self.workflows: dict[str, Workflow] = {}
        # event_name -> {workflow_name: [Trigger, ...]}
        self._event_index: dict[str, dict[str, list[Trigger]]] = {}
        # workflow_name -> [schedule Trigger, ...]
        self._schedule_index: dict[str, list[Trigger]] = {}

    def register_workflow(self, workflow: Workflow) -> None:
        """Register or overwrite a workflow."""
        old = self.workflows.get(workflow.name)
        if old is not None:
            self._remove_indices(old)
        self.workflows[workflow.name] = workflow
        schedule_triggers: list[Trigger] = []
        for trigger in workflow.triggers:
            if trigger.trigger_type == TriggerType.EVENT:
                self._event_index.setdefault(trigger.condition, {}).setdefault(workflow.name, []).append(trigger)
            elif trigger.trigger_type == TriggerType.SCHEDULE:
                schedule_triggers.append(trigger)
        if schedule_triggers:
            self._schedule_index[workflow.name] = schedule_triggers

    def unregister_workflow(self, name: str) -> bool:
        """Remove a workflow."""
        wf = self.workflows.pop(name, None)
        if wf is None:
            return False
        self._remove_indices(wf)
        return True

    def _remove_indices(self, wf: Workflow) -> None:
        """Remove a workflow from all internal indices."""
        self._schedule_index.pop(wf.name, None)
        for trigger in wf.triggers:
            if trigger.trigger_type == TriggerType.EVENT:
                event_map = self._event_index.get(trigger.condition)
                if event_map is not None:
                    event_map.pop(wf.name, None)
                    if not event_map:
                        del self._event_index[trigger.condition]

    def run_pending(
        self,
        task_runner: Callable[[Workflow, Task], Any] | None = None,
    ) -> dict[str, list[str]]:
        """Execute all workflows whose SCHEDULE triggers have fired.

        Returns a mapping of workflow_name -> list_of_task_names_executed.
        """
        results: dict[str, list[str]] = {}
        now = time.time()
        for name, triggers in self._schedule_index.items():
            wf = self.workflows.get(name)
            if wf is None or not wf.enabled:
                continue
            fired = False
            for trigger in triggers:
                if trigger.should_fire(now):
                    trigger.mark_fired(now)
                    fired = True
            if not fired:
                continue
            executed: list[str] = []
            for task in wf.tasks:
                if task.status != "pending":
                    continue
                task.status = "running"
                if task_runner is not None:
                    try:
                        task.result = task_runner(wf, task)
                        task.status = "completed"
                    except Exception:
                        task.status = "failed"
                else:
                    task.status = "completed"
                executed.append(task.name)
            if executed:
                results[name] = executed
        return results

    def trigger_event(
        self,
        event_name: str,
        event_payload: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        """Fire an event trigger and run matching workflows."""
        results: dict[str, list[str]] = {}
        event_map = self._event_index.get(event_name)
        if event_map is None:
            return results
        payload = event_payload or {}
        for wf_name, triggers in event_map.items():
            wf = self.workflows.get(wf_name)
            if wf is None or not wf.enabled:
                continue
            for trigger in triggers:
                trigger.mark_fired()
            executed: list[str] = []
            for task in wf.tasks:
                if task.status != "pending":
                    continue
                task.status = "running"
                task.result = payload
                task.status = "completed"
                executed.append(task.name)
            if executed:
                results[wf_name] = executed
        return results

    def list_workflows(self) -> list[str]:
        return list(self.workflows.keys())

    def to_entries(self) -> list[MemoryEntry]:
        """Export all workflows as MemoryEntries."""
        return [wf.to_memory_entry() for wf in self.workflows.values()]

    def reflect(self) -> str:
        total_tasks = sum(len(wf.tasks) for wf in self.workflows.values())
        return (
            f"Programmatic Memory: {len(self.workflows)} workflows, {total_tasks} tasks"
        )
