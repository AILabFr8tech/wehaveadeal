"""Finding and loading community agents.

A community agent is any `DecisionEngine` subclass that declares which side
of the market it plays:

    class MyCarrier(DecisionEngine):
        role = "carrier"            # or "shipper"
        engine_name = "MyCarrier"
        def next_move(self, state): ...

Agents can be referenced three ways (the "agent spec" used by the CLI):

    path/to/my_agent.py             -- the single agent class defined in that file
    path/to/my_agent.py:MyCarrier   -- a specific class in that file
    my_package.module:MyCarrier     -- an importable module (pip-installed agents)

`discover_agents(dir)` scans a folder (the repo's `agents/` drop-in folder by
default) and returns every agent it finds, plus the files that failed to
load, so one broken file never hides the others.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type

from wehaveadeal.agents.base import DecisionEngine

VALID_ROLES = ("shipper", "carrier")
_USER_MODULE_PREFIX = "wehaveadeal_user_agents"


class AgentLoadError(Exception):
    """Raised when an agent spec cannot be turned into a usable agent class.
    The message is written for the person who wrote the agent."""


@dataclass(frozen=True)
class AgentSpec:
    cls: Type[DecisionEngine]
    role: str
    name: str  # display name, e.g. "MyCarrier"
    kind: str  # id-safe key, e.g. "u_mycarrier" -- used in agent ids "<role>-<kind>-<n>"
    source: str  # file path or module the class came from
    source_sha256: Optional[str] = None  # hash of the source file, recorded in benchmark reports

    def build(self, agent_id: str) -> DecisionEngine:
        try:
            return self.cls(agent_id=agent_id)
        except TypeError as exc:
            raise AgentLoadError(
                f"{self.name} could not be constructed as {self.name}(agent_id=...): {exc}. "
                "Agents must accept `agent_id` as a keyword argument (call super().__init__(agent_id, ...)) "
                "and give every other constructor parameter a default."
            ) from exc


@dataclass
class DiscoveryResult:
    agents: List[AgentSpec] = field(default_factory=list)
    errors: List[dict] = field(default_factory=list)  # [{"file": ..., "error": ...}]

    def for_role(self, role: str) -> List[AgentSpec]:
        return [a for a in self.agents if a.role == role]


def kind_for(name: str) -> str:
    """"My Carrier v2" -> "u_my_carrier_v2". Lowercase alnum/underscore only, so it can
    never contain the '-' separator of agent ids nor collide with a baseline kind."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "agent"
    return f"u_{slug}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _import_file(path: Path):
    path = path.resolve()
    if not path.exists():
        raise AgentLoadError(f"Agent file not found: {path}")
    if path.suffix != ".py":
        raise AgentLoadError(f"Agent file must be a .py file, got: {path}")
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    module_name = f"{_USER_MODULE_PREFIX}.{re.sub(r'[^A-Za-z0-9_]', '_', path.stem)}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AgentLoadError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    # Let the agent import sibling helper modules from its own folder.
    parent = str(path.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 -- any error in user code is reported, not raised raw
        sys.modules.pop(module_name, None)
        raise AgentLoadError(f"Importing {path.name} failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if added:
            sys.path.remove(parent)
    return module


def _agent_classes_in(module) -> List[type]:
    """DecisionEngine subclasses *defined* in `module` (not merely imported into it,
    so `from wehaveadeal.agents import CarrierR` inside a user file is not picked up)."""
    found = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if issubclass(obj, DecisionEngine) and obj is not DecisionEngine and not inspect.isabstract(obj):
            found.append(obj)
    return found


def _to_spec(cls: type, source: str, sha: Optional[str]) -> AgentSpec:
    role = getattr(cls, "role", None)
    if role not in VALID_ROLES:
        raise AgentLoadError(
            f"{cls.__name__} must declare which side it plays, as a class attribute: "
            f'role = "carrier" or role = "shipper" (got {role!r}).'
        )
    name = getattr(cls, "engine_name", None)
    if not name or name == "base":
        name = cls.__name__
    return AgentSpec(cls=cls, role=role, name=name, kind=kind_for(name), source=source, source_sha256=sha)


def load_agent(spec: str) -> AgentSpec:
    """Resolve one agent spec (see module docstring) into an AgentSpec."""
    target, _, class_name = spec.partition(":")
    # Windows drive letters ("C:\\...") also contain ':' -- re-split on the LAST colon then.
    if re.match(r"^[A-Za-z]$", target) and class_name:
        target, _, class_name = spec.rpartition(":")
        if not target.endswith(".py"):
            target, class_name = spec, ""

    if target.endswith(".py") or Path(target).suffix == ".py":
        path = Path(target)
        module = _import_file(path)
        source, sha = str(path), _sha256(path.resolve())
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise AgentLoadError(f"Could not import module {target!r}: {exc}") from exc
        source = target
        sha = _sha256(Path(module.__file__)) if getattr(module, "__file__", None) else None

    classes = _agent_classes_in(module)
    if class_name:
        matches = [c for c in classes if c.__name__ == class_name]
        if not matches:
            available = ", ".join(c.__name__ for c in classes) or "none"
            raise AgentLoadError(
                f"No DecisionEngine subclass named {class_name!r} in {source} (found: {available})."
            )
        return _to_spec(matches[0], source, sha)

    if not classes:
        raise AgentLoadError(f"No DecisionEngine subclass defined in {source}.")
    if len(classes) > 1:
        names = ", ".join(c.__name__ for c in classes)
        raise AgentLoadError(
            f"{source} defines several agents ({names}); pick one with {source}:ClassName."
        )
    return _to_spec(classes[0], source, sha)


def discover_agents(directory: Path | str) -> DiscoveryResult:
    """Every agent defined in `directory/*.py` (files starting with '_' are skipped, so
    helpers like `_utils.py` can live alongside agents). Never raises for user errors:
    broken files are reported in `.errors`."""
    result = DiscoveryResult()
    directory = Path(directory)
    if not directory.is_dir():
        return result
    seen_kinds = {}
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = _import_file(path)
            sha = _sha256(path.resolve())
            for cls in _agent_classes_in(module):
                try:
                    spec = _to_spec(cls, str(path), sha)
                except AgentLoadError as exc:
                    result.errors.append({"file": str(path), "error": str(exc)})
                    continue
                key = (spec.role, spec.kind)
                if key in seen_kinds:
                    result.errors.append({
                        "file": str(path),
                        "error": f"{spec.name} has the same name as an agent in {seen_kinds[key]}; "
                                 "rename one of them (engine_name) so they can be told apart.",
                    })
                    continue
                seen_kinds[key] = str(path)
                result.agents.append(spec)
        except AgentLoadError as exc:
            result.errors.append({"file": str(path), "error": str(exc)})
    return result
