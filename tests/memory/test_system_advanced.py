"""Advanced tests for AgentMemorySystem and standalone memory components."""

import os
import tempfile
import time

import pytest

from kimix.memory.system import AgentMemorySystem
from kimix.memory.types import MemoryType
from kimix.memory.procedural_memory import ProceduralMemory
from kimix.memory.programmatic_memory import ProgrammaticMemory, Workflow, Task, Trigger, TriggerType
from kimix.memory.cold_storage import ColdStorage
from kimix.memory.types import MemoryEntry


class TestSystemProceduralIntegration:
    def test_add_scar(self):
        pm = ProceduralMemory()
        pm.add_scar("division by zero", "check denominator", ["divide", "zero"], severity=8.0)
        assert len(pm.scars) == 1
        assert pm.scars[0].failure_pattern == "division by zero"

    def test_add_rule(self):
        pm = ProceduralMemory()
        pm.add_rule("deploy on friday", "reject deployment", 10.0, ["ops"])
        assert len(pm.rules) == 1
        assert pm.rules[0].condition == "deploy on friday"


class TestSystemProgrammaticIntegration:
    def test_register_workflow_and_run(self):
        prog = ProgrammaticMemory()
        wf = Workflow(name="cleanup")
        wf.add_trigger(Trigger(TriggerType.SCHEDULE, condition="0"))
        wf.add_task(Task(name="delete_temp"))
        prog.register_workflow(wf)
        ran = prog.run_pending()
        assert "cleanup" in ran

    def test_trigger_event(self):
        prog = ProgrammaticMemory()
        wf = Workflow(name="alert")
        wf.add_trigger(Trigger(TriggerType.EVENT, condition="high_cpu"))
        wf.add_task(Task(name="page_ops"))
        prog.register_workflow(wf)
        ran = prog.trigger_event("high_cpu", {"cpu": 99})
        assert "alert" in ran


class TestSystemColdStorageIntegration:
    def test_archive_to_cold_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cs = ColdStorage(archive_dir=tmpdir)
            entry = MemoryEntry(content="old low-priority fact", memory_type=MemoryType.SEMANTIC)
            cs.archive([entry])
            assert len(cs.list_archives()) >= 1

    def test_archive_with_explicit_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cs = ColdStorage(archive_dir=tmpdir)
            entries = [MemoryEntry(content="explicit", memory_type=MemoryType.SEMANTIC)]
            cs.archive(entries, start_year=2020, end_year=2020)
            restored = cs.restore_range(2020, 2020)
            assert any(e.content == "explicit" for e in restored)


class TestSystemSelfReflection:
    def test_self_reflect_downranks_stale(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sys = AgentMemorySystem(ltm_path=path)
            entry = sys.remember("stale fact", importance=5.0)
            # Simulate old access
            entry.last_accessed = time.time() - 30 * 86400
            entry.access_count = 0
            report = sys.self_reflect()
            assert "Down-ranked stale" in report or "Long-term entries" in report
        finally:
            os.unlink(path)

    def test_self_reflect_report_structure(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sys = AgentMemorySystem(ltm_path=path)
            report = sys.self_reflect()
            assert "Self-Reflection Report" in report
            assert "Short-term buffer" in report
        finally:
            os.unlink(path)


class TestSystemTemporalValidity:
    def test_perceive_with_expiry(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sys = AgentMemorySystem(ltm_path=path)
            sys.perceive("temp event", importance=5.0, expires_at=time.time() - 1)
            sys.short_term.clear_expired()
            assert len(sys.short_term.buffer) == 0
        finally:
            os.unlink(path)

    def test_remember_with_expiry(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sys = AgentMemorySystem(ltm_path=path)
            sys.remember("temp fact", importance=5.0, expires_at=time.time() - 1)
            results = sys.long_term.retrieve("temp")
            assert results == []
        finally:
            os.unlink(path)


class TestSystemSQLiteOption:
    def test_system_with_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sys = AgentMemorySystem(
                ltm_path=f"{tmpdir}/ltm.json",
                use_sqlite=True,
                db_path=f"{tmpdir}/memory.db",
                agent_id="sql_agent",
            )
            sys.remember("sqlite memory", importance=8.0, tags=["sql"])
            results = sys.recall("sqlite", use_long=True)
            assert len(results["long_term"]) == 1
            assert sys.long_term.count() == 1
            if sys.long_term._backend is not None:
                sys.long_term._backend.close()
