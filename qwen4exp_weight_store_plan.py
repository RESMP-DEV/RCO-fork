#!/usr/bin/env python3
"""Host-safe Qwen4-Exp candidate-store planning seam for RCO.

Reads a compact JSON topology/config manifest and enumerates the exact
per-expert ``gate_proj`` / ``up_proj`` / ``down_proj`` quantization jobs plus
the WeightStore directory names that ``rco_search_quant.py`` expects for a
Qwen4-Exp model whose routed experts ship as fused 3-D parameters.

The seam is dry-run only and host-safe by construction: it never imports
Torch, never opens a checkpoint, never runs GPTQ or CUDA, never downloads
anything, and never contacts the network.  Its output is a planning receipt,
not permission to run an optimizer.

Manifest shape (format ``qwen4exp-weight-store-plan-v1``)::

    {
      "format": "qwen4exp-weight-store-plan-v1",
      "model": "Qwen/Qwen3.8-Flash-Next",
      "candidate_store": "/data/qwen4exp_candidates",
      "layer_prefix": "model.layers",
      "topology": {
        "num_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "hidden_size": 2560,
        "moe_intermediate_size": 640,
        "layer_types": ["linear_attention", ...]
      },
      "selection": {"layers": "all", "experts": "all"},
      "bitwidths": [4],
      "measurements": ["W4A16", "W4A4"],
      "cost_model": "exact",
      "costs": [
        {
          "scope": {"layers": "all", "experts": "all",
                    "projections": ["gate_proj", "up_proj", "down_proj"],
                    "bitwidths": [4], "measurements": ["W4A16"]},
          "kind": "exact",
          "bytes": 844800,
          "source": "gguf tensor bytes",
          "components": {"tensor": 819200, "scales": 25600}
        }
      ]
    }

A candidate is GPU-ready only when every enumerated candidate carries exact
artifact byte-cost metadata.  Nominal or missing costs fail closed.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMAT = "qwen4exp-weight-store-plan-v1"

PROJECTIONS: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

# Deployment measurement profiles the seam understands.  Each label maps to
# its (weight bits, activation bits) pair.
SUPPORTED_MEASUREMENTS: dict[str, tuple[int, int]] = {
    "W4A16": (4, 16),
    "W4A4": (4, 4),
}

EXACT_COST_KIND = "exact"

MIN_BITWIDTH = 2
MAX_BITWIDTH = 8

# RCO's src/grouping.py groups per-expert layers with this pattern.  A model
# whose module path differs (for example model.language_model.layers) still
# plans fine, but stock rco_search_quant.py --moe-per-expert will not group
# those experts into per-(layer, expert) groups.
RCO_MOE_GROUP_PATTERN = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

MAX_REPORTED_ISSUES = 20

NON_CLAIM = (
    "This plan enumerates candidate jobs and expected WeightStore paths only. It does "
    "not prove that any candidate artifact exists on disk, that the declared byte "
    "costs match a real file, that GPTQ or CUDA succeeded, or that an optimizer run "
    "is authorized."
)

GPU_READY_NON_CLAIM = (
    "GPU-ready here means the manifest passes host-safe validation and every planned "
    "candidate carries exact artifact byte-cost metadata. It is not permission to run "
    "GPTQ, CUDA, or an optimizer."
)


class PlanError(ValueError):
    """Raised when a manifest or plan is invalid. The seam fails closed."""


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{where} must be a JSON object")
    return value


def _require_int(
    value: Any,
    where: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise PlanError(f"{where} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise PlanError(f"{where} must be <= {maximum}, got {value}")
    return value


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{where} must be a non-empty string")
    return value.strip()


def _require_identifier(value: Any, where: str) -> str:
    text = _require_str(value, where)
    parts = text.split(".")
    if not all(_IDENTIFIER.match(part) for part in parts):
        raise PlanError(f"{where} must be a dotted Python identifier, got {text!r}")
    return text


def _require_non_empty_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise PlanError(f"{where} must be a non-empty list")
    return value


@dataclass(frozen=True)
class Topology:
    """Declared Qwen4-Exp geometry."""

    num_layers: int
    num_experts: int
    num_experts_per_tok: int
    hidden_size: int
    moe_intermediate_size: int
    layer_types: tuple[str, ...] | None
    experts_per_layer: tuple[int, ...]
    active_per_layer: tuple[int, ...]

    def expert_count(self, layer_index: int) -> int:
        return self.experts_per_layer[layer_index]

    def active_count(self, layer_index: int) -> int:
        return self.active_per_layer[layer_index]


@dataclass(frozen=True)
class CostEntry:
    """One scoped byte-cost record from the manifest."""

    index: int
    layers: frozenset[int] | None
    experts: frozenset[int] | None
    projections: frozenset[str]
    bitwidths: frozenset[int]
    measurements: frozenset[str]
    kind: str
    bytes: int | None
    source: str | None
    components: Mapping[str, int] | None


@dataclass(frozen=True)
class Manifest:
    """A validated manifest."""

    path: Path
    model: str
    candidate_store: str
    layer_prefix: str
    experts_prefix: str
    topology: Topology
    selected_layers: tuple[int, ...]
    selected_experts: tuple[int, ...] | None
    projections: tuple[str, ...]
    bitwidths: tuple[int, ...]
    measurements: tuple[str, ...]
    cost_model: str
    costs: tuple[CostEntry, ...]

    def layer_name(self, layer_index: int, expert_index: int, projection: str) -> str:
        """WeightStore directory name for one unfused expert projection."""
        return (
            f"{self.layer_prefix}.{layer_index}.mlp.{self.experts_prefix}."
            f"{expert_index}.{projection}"
        )

    def planned_experts(self, layer_index: int) -> tuple[int, ...]:
        """Expert indices planned for one layer, honouring the selection."""
        count = self.topology.expert_count(layer_index)
        if self.selected_experts is None:
            return tuple(range(count))
        return tuple(index for index in self.selected_experts if index < count)


@dataclass(frozen=True)
class ManifestScope:
    """Plan-level scope used to validate cost-entry scopes against the plan."""

    selected_layers: tuple[int, ...]
    selected_experts: tuple[int, ...] | None
    smallest_expert_count: int
    projections: tuple[str, ...]
    bitwidths: tuple[int, ...]
    measurements: tuple[str, ...]


@dataclass(frozen=True)
class Job:
    """One quantization job: one expert projection at one bitwidth."""

    layer_index: int
    expert_index: int
    projection: str
    bitwidth: int
    layer_name: str
    weight_artifact: str
    qparams_artifact: str


def _parse_topology(raw: Any) -> Topology:
    topology = _require_mapping(raw, "topology")
    num_layers = _require_int(topology.get("num_layers"), "topology.num_layers", minimum=1)
    num_experts = _require_int(topology.get("num_experts"), "topology.num_experts", minimum=1)
    active = _require_int(
        topology.get("num_experts_per_tok"),
        "topology.num_experts_per_tok",
        minimum=1,
    )
    if active > num_experts:
        raise PlanError(
            f"topology.num_experts_per_tok ({active}) must not exceed "
            f"topology.num_experts ({num_experts})"
        )
    hidden_size = _require_int(topology.get("hidden_size"), "topology.hidden_size", minimum=1)
    moe_intermediate_size = _require_int(
        topology.get("moe_intermediate_size"),
        "topology.moe_intermediate_size",
        minimum=1,
    )

    layer_types: tuple[str, ...] | None = None
    raw_layer_types = topology.get("layer_types")
    if raw_layer_types is not None:
        if not isinstance(raw_layer_types, list) or not raw_layer_types:
            raise PlanError("topology.layer_types must be a non-empty list of strings")
        layer_types = tuple(
            _require_str(item, f"topology.layer_types[{index}]")
            for index, item in enumerate(raw_layer_types)
        )
        if len(layer_types) != num_layers:
            raise PlanError(
                f"topology.layer_types has {len(layer_types)} entries but "
                f"topology.num_layers is {num_layers}"
            )

    experts_per_layer = _parse_per_layer_ints(
        topology.get("experts_per_layer"),
        "topology.experts_per_layer",
        num_layers=num_layers,
        default=num_experts,
        minimum=1,
        maximum=num_experts,
    )
    active_per_layer = _parse_per_layer_ints(
        topology.get("active_per_layer"),
        "topology.active_per_layer",
        num_layers=num_layers,
        default=active,
        minimum=1,
        maximum=num_experts,
    )
    for layer_index, count in enumerate(experts_per_layer):
        if active_per_layer[layer_index] > count:
            raise PlanError(
                f"topology.active_per_layer[{layer_index}] ({active_per_layer[layer_index]}) "
                f"must not exceed that layer's expert count ({count})"
            )

    return Topology(
        num_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_tok=active,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        layer_types=layer_types,
        experts_per_layer=experts_per_layer,
        active_per_layer=active_per_layer,
    )


def _parse_per_layer_ints(
    raw: Any,
    where: str,
    *,
    num_layers: int,
    default: int,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    if raw is None:
        return tuple(default for _ in range(num_layers))
    if isinstance(raw, list):
        items: list[tuple[int, Any]] = list(enumerate(raw))
    elif isinstance(raw, dict):
        items = []
        for key, value in raw.items():
            try:
                items.append((int(key), value))
            except (TypeError, ValueError) as error:
                raise PlanError(f"{where} keys must be layer indices: {error}") from error
    else:
        raise PlanError(f"{where} must be a list or an object keyed by layer index")
    if len(items) != num_layers:
        raise PlanError(
            f"{where} has {len(items)} entries but the topology declares {num_layers} layers"
        )
    values = [0] * num_layers
    seen: set[int] = set()
    for layer_index, value in items:
        if not 0 <= layer_index < num_layers:
            raise PlanError(f"{where} references out-of-range layer index {layer_index}")
        if layer_index in seen:
            raise PlanError(f"{where} lists layer {layer_index} more than once")
        seen.add(layer_index)
        values[layer_index] = _require_int(
            value,
            f"{where}[{layer_index}]",
            minimum=minimum,
            maximum=maximum,
        )
    return tuple(values)


def _parse_selection(
    raw: Any, topology: Topology
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    selection = _require_mapping(raw if raw is not None else {}, "selection")
    raw_layers = selection.get("layers", "all")
    if raw_layers == "all":
        selected_layers = tuple(range(topology.num_layers))
    else:
        if not isinstance(raw_layers, list) or not raw_layers:
            raise PlanError('selection.layers must be "all" or a non-empty list of integers')
        selected_layers = tuple(
            _require_int(
                item, f"selection.layers[{index}]", minimum=0, maximum=topology.num_layers - 1
            )
            for index, item in enumerate(raw_layers)
        )
        if len(set(selected_layers)) != len(selected_layers):
            raise PlanError("selection.layers contains duplicate layer indices")

    raw_experts = selection.get("experts", "all")
    selected_experts: tuple[int, ...] | None
    if raw_experts == "all":
        selected_experts = None
    else:
        if not isinstance(raw_experts, list) or not raw_experts:
            raise PlanError('selection.experts must be "all" or a non-empty list of integers')
        smallest = min(topology.expert_count(layer) for layer in selected_layers)
        selected_experts = tuple(
            _require_int(item, f"selection.experts[{index}]", minimum=0, maximum=smallest - 1)
            for index, item in enumerate(raw_experts)
        )
        if len(set(selected_experts)) != len(selected_experts):
            raise PlanError("selection.experts contains duplicate expert indices")
    return selected_layers, selected_experts


def _parse_projections(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return PROJECTIONS
    if not isinstance(raw, list) or not raw:
        raise PlanError("selection.projections must be a non-empty list of projection names")
    projections = tuple(
        _require_str(item, f"selection.projections[{index}]") for index, item in enumerate(raw)
    )
    unknown = [name for name in projections if name not in PROJECTIONS]
    if unknown:
        raise PlanError(
            f"selection.projections contains unsupported projections {unknown}; "
            f"expected a subset of {list(PROJECTIONS)}"
        )
    if len(set(projections)) != len(projections):
        raise PlanError("selection.projections contains duplicates")
    return projections


def _parse_measurements(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise PlanError("measurements must be a non-empty list of measurement labels")
    labels: list[str] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            item = item.get("label")
        label = _require_str(item, f"measurements[{index}]")
        if label not in SUPPORTED_MEASUREMENTS:
            raise PlanError(
                f"measurements[{index}] label {label!r} is not supported; expected a subset of "
                f"{sorted(SUPPORTED_MEASUREMENTS)}"
            )
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise PlanError("measurements contains duplicate labels")
    return tuple(labels)


def _parse_bitwidths(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise PlanError("bitwidths must be a non-empty list of integers")
    bitwidths = tuple(
        _require_int(item, f"bitwidths[{index}]", minimum=MIN_BITWIDTH, maximum=MAX_BITWIDTH)
        for index, item in enumerate(raw)
    )
    if len(set(bitwidths)) != len(bitwidths):
        raise PlanError("bitwidths contains duplicates")
    return bitwidths


def _parse_scope(
    raw: Any,
    manifest_bits: ManifestScope,
    where: str,
) -> tuple[
    frozenset[int] | None, frozenset[int] | None, frozenset[str], frozenset[int], frozenset[str]
]:
    scope = _require_mapping(raw if raw is not None else {}, where)
    unknown_keys = sorted(
        set(scope) - {"layers", "experts", "projections", "bitwidths", "measurements"}
    )
    if unknown_keys:
        raise PlanError(f"{where} has unsupported keys {unknown_keys}")

    raw_layers = scope.get("layers", "all")
    if raw_layers == "all":
        layers: frozenset[int] | None = None
    else:
        if not isinstance(raw_layers, list) or not raw_layers:
            raise PlanError(f'{where}.layers must be "all" or a non-empty list of integers')
        layers = frozenset(
            _require_int(item, f"{where}.layers[{index}]", minimum=0)
            for index, item in enumerate(raw_layers)
        )
        outside = sorted(layers - set(manifest_bits.selected_layers))
        if outside:
            raise PlanError(f"{where}.layers references layers outside the selection: {outside}")

    raw_experts = scope.get("experts", "all")
    if raw_experts == "all":
        experts: frozenset[int] | None = None
    else:
        if not isinstance(raw_experts, list) or not raw_experts:
            raise PlanError(f'{where}.experts must be "all" or a non-empty list of integers')
        experts = frozenset(
            _require_int(item, f"{where}.experts[{index}]", minimum=0)
            for index, item in enumerate(raw_experts)
        )
        smallest = manifest_bits.smallest_expert_count
        outside = sorted(index for index in experts if index >= smallest)
        if outside:
            raise PlanError(
                f"{where}.experts references expert indices that do not exist in every "
                f"selected layer: {outside}"
            )
        if manifest_bits.selected_experts is not None:
            outside = sorted(experts - set(manifest_bits.selected_experts))
            if outside:
                raise PlanError(
                    f"{where}.experts references experts outside the selection: {outside}"
                )

    projections = frozenset(
        _require_str(item, f"{where}.projections[{index}]")
        for index, item in enumerate(
            _require_non_empty_list(
                scope.get("projections", list(manifest_bits.projections)),
                f"{where}.projections",
            )
        )
    )
    unsupported = sorted(projections - set(manifest_bits.projections))
    if unsupported:
        raise PlanError(
            f"{where}.projections references projections outside the plan: {unsupported}"
        )

    bitwidths = frozenset(
        _require_int(item, f"{where}.bitwidths[{index}]")
        for index, item in enumerate(
            _require_non_empty_list(
                scope.get("bitwidths", list(manifest_bits.bitwidths)),
                f"{where}.bitwidths",
            )
        )
    )
    unsupported_bw = sorted(bitwidths - set(manifest_bits.bitwidths))
    if unsupported_bw:
        raise PlanError(
            f"{where}.bitwidths references bitwidths outside the plan: {unsupported_bw}"
        )

    measurements = frozenset(
        _require_str(item, f"{where}.measurements[{index}]")
        for index, item in enumerate(
            _require_non_empty_list(
                scope.get("measurements", list(manifest_bits.measurements)),
                f"{where}.measurements",
            )
        )
    )
    unsupported_measurements = sorted(measurements - set(manifest_bits.measurements))
    if unsupported_measurements:
        raise PlanError(
            f"{where}.measurements references measurement labels outside the plan: "
            f"{unsupported_measurements}"
        )
    return layers, experts, projections, bitwidths, measurements


def _parse_costs(raw: Any, scope: ManifestScope) -> tuple[CostEntry, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PlanError("costs must be a list of byte-cost records")
    entries: list[CostEntry] = []
    for index, item in enumerate(raw):
        where = f"costs[{index}]"
        record = _require_mapping(item, where)
        unsupported = sorted(set(record) - {"scope", "kind", "bytes", "source", "components"})
        if unsupported:
            raise PlanError(f"{where} has unsupported keys {unsupported}")
        layers, experts, projections, bitwidths, measurements = _parse_scope(
            record.get("scope"), scope, f"{where}.scope"
        )
        kind = record.get("kind")
        if kind is not None and not isinstance(kind, str):
            raise PlanError(f"{where}.kind must be a string")
        kind_text = kind.strip() if isinstance(kind, str) else ""
        byte_count: int | None = None
        if record.get("bytes") is not None:
            byte_count = _require_int(record["bytes"], f"{where}.bytes", minimum=1)
        source = record.get("source")
        if source is not None:
            source = _require_str(source, f"{where}.source")
        components: dict[str, int] | None = None
        if record.get("components") is not None:
            raw_components = _require_mapping(record["components"], f"{where}.components")
            components = {}
            for name, value in raw_components.items():
                components[_require_str(name, f"{where}.components key")] = _require_int(
                    value, f"{where}.components[{name}]", minimum=0
                )
            if byte_count is not None:
                total = sum(components.values())
                if total != byte_count:
                    raise PlanError(
                        f"{where}.components sum to {total} but {where}.bytes is {byte_count}"
                    )
        entries.append(
            CostEntry(
                index=index,
                layers=layers,
                experts=experts,
                projections=projections,
                bitwidths=bitwidths,
                measurements=measurements,
                kind=kind_text,
                bytes=byte_count,
                source=source,
                components=components,
            )
        )
    return tuple(entries)


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a manifest. Raises PlanError on any invalid input."""
    manifest_path = Path(path)
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise PlanError(f"{manifest_path}: cannot read manifest: {error}") from error
    try:
        document = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise PlanError(f"{manifest_path}: invalid JSON: {error}") from error
    data = _require_mapping(document, str(manifest_path))

    manifest_format = data.get("format")
    if manifest_format != FORMAT:
        raise PlanError(f"{manifest_path}: expected format {FORMAT}, got {manifest_format!r}")

    model = _require_str(data.get("model"), "model")
    candidate_store = _require_str(data.get("candidate_store"), "candidate_store")
    layer_prefix = _require_identifier(data.get("layer_prefix", "model.layers"), "layer_prefix")
    experts_prefix = _require_identifier(data.get("experts_prefix", "experts"), "experts_prefix")

    topology = _parse_topology(data.get("topology"))
    selection_raw = data.get("selection")
    selection = _require_mapping(selection_raw if selection_raw is not None else {}, "selection")
    selected_layers, selected_experts = _parse_selection(selection_raw, topology)
    projections = _parse_projections(selection.get("projections"))
    bitwidths = _parse_bitwidths(data.get("bitwidths"))
    measurements = _parse_measurements(data.get("measurements"))

    for label in measurements:
        weight_bits = SUPPORTED_MEASUREMENTS[label][0]
        if weight_bits not in bitwidths:
            raise PlanError(
                f"measurement {label} needs weight bitwidth {weight_bits}, which is not in "
                f"bitwidths {list(bitwidths)}"
            )

    cost_model = _require_str(data.get("cost_model", EXACT_COST_KIND), "cost_model")
    scope = ManifestScope(
        selected_layers=selected_layers,
        selected_experts=selected_experts,
        smallest_expert_count=min(topology.expert_count(layer) for layer in selected_layers),
        projections=projections,
        bitwidths=bitwidths,
        measurements=measurements,
    )
    costs = _parse_costs(data.get("costs"), scope)

    return Manifest(
        path=manifest_path,
        model=model,
        candidate_store=candidate_store,
        layer_prefix=layer_prefix,
        experts_prefix=experts_prefix,
        topology=topology,
        selected_layers=selected_layers,
        selected_experts=selected_experts,
        projections=projections,
        bitwidths=bitwidths,
        measurements=measurements,
        cost_model=cost_model,
        costs=costs,
    )


def enumerate_jobs(manifest: Manifest) -> list[Job]:
    """Enumerate every per-expert gate/up/down quantization job in the plan."""
    jobs: list[Job] = []
    for layer_index in manifest.selected_layers:
        for expert_index in manifest.planned_experts(layer_index):
            for projection in manifest.projections:
                layer_name = manifest.layer_name(layer_index, expert_index, projection)
                for bitwidth in manifest.bitwidths:
                    jobs.append(
                        Job(
                            layer_index=layer_index,
                            expert_index=expert_index,
                            projection=projection,
                            bitwidth=bitwidth,
                            layer_name=layer_name,
                            weight_artifact=f"{layer_name}/{bitwidth}.pth",
                            qparams_artifact=f"{layer_name}/{bitwidth}_qparams.pt",
                        )
                    )
    return jobs


@dataclass
class CostResolution:
    """Result of matching cost entries against planned candidates."""

    assigned: dict[tuple[int, int, str, int, str], int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    issue_count: int = 0

    def add_issue(self, message: str) -> None:
        self.issue_count += 1
        if len(self.issues) < MAX_REPORTED_ISSUES:
            self.issues.append(message)


def _describe(key: tuple[int, int, str, int, str]) -> str:
    layer_index, expert_index, projection, bitwidth, measurement = key
    return (
        f"layer {layer_index} expert {expert_index} {projection} @ {bitwidth}-bit ({measurement})"
    )


def measurement_matches_bitwidth(measurement: str, bitwidth: int) -> bool:
    """Whether a W*A* measurement label can refer to this weight bitwidth."""
    return SUPPORTED_MEASUREMENTS[measurement][0] == bitwidth


def resolve_costs(manifest: Manifest, jobs: Sequence[Job]) -> CostResolution:
    """Match cost entries to planned candidates and record every blocking gap."""
    resolution = CostResolution()
    if manifest.cost_model != EXACT_COST_KIND:
        resolution.add_issue(
            f"cost_model {manifest.cost_model!r} is not {EXACT_COST_KIND!r}: nominal costs "
            "cannot establish GPU readiness"
        )

    planned_experts = {
        layer_index: manifest.planned_experts(layer_index)
        for layer_index in manifest.selected_layers
    }
    for entry in manifest.costs:
        entry_layers = (
            manifest.selected_layers if entry.layers is None else tuple(sorted(entry.layers))
        )
        for layer_index in entry_layers:
            layer_experts = planned_experts[layer_index]
            if entry.experts is None:
                entry_experts = layer_experts
            else:
                entry_experts = tuple(index for index in layer_experts if index in entry.experts)
            for expert_index in entry_experts:
                for projection in sorted(entry.projections):
                    for bitwidth in sorted(entry.bitwidths):
                        for measurement in sorted(entry.measurements):
                            if not measurement_matches_bitwidth(measurement, bitwidth):
                                continue
                            key = (
                                layer_index,
                                expert_index,
                                projection,
                                bitwidth,
                                measurement,
                            )
                            previous = resolution.assigned.get(key)
                            if previous is not None:
                                resolution.add_issue(
                                    f"ambiguous byte-cost metadata for {_describe(key)}: "
                                    f"cost entries {previous} and {entry.index} both cover it"
                                )
                                continue
                            resolution.assigned[key] = entry.index

    coverage = Counter(resolution.assigned.values())
    for entry in manifest.costs:
        affected = coverage.get(entry.index, 0)
        if affected == 0:
            resolution.add_issue(
                f"cost entry {entry.index} covers no planned measurement-compatible candidate"
            )

    for job in jobs:
        for measurement in manifest.measurements:
            if not measurement_matches_bitwidth(measurement, job.bitwidth):
                continue
            key = (
                job.layer_index,
                job.expert_index,
                job.projection,
                job.bitwidth,
                measurement,
            )
            if key not in resolution.assigned:
                resolution.add_issue(f"missing byte-cost metadata for {_describe(key)}")

    for entry in manifest.costs:
        affected = coverage.get(entry.index, 0)
        if entry.kind != EXACT_COST_KIND:
            resolution.add_issue(
                f"cost entry {entry.index} declares kind={entry.kind!r}, not "
                f"{EXACT_COST_KIND!r}; {affected} candidate(s) it covers are not GPU-ready"
            )
        elif entry.bytes is None:
            resolution.add_issue(
                f"cost entry {entry.index} has no exact byte count; {affected} candidate(s) "
                "it covers are not GPU-ready"
            )
        elif not entry.source:
            resolution.add_issue(
                f"cost entry {entry.index} has no source provenance; {affected} candidate(s) "
                "it covers are not GPU-ready"
            )
    return resolution


def iter_candidate_rows(
    manifest: Manifest,
    jobs: Sequence[Job],
    resolution: CostResolution,
) -> Iterator[dict[str, Any]]:
    """Yield one row per planned candidate (job crossed with measurement label)."""
    for job in jobs:
        for measurement in manifest.measurements:
            if not measurement_matches_bitwidth(measurement, job.bitwidth):
                continue
            key = (
                job.layer_index,
                job.expert_index,
                job.projection,
                job.bitwidth,
                measurement,
            )
            entry_index = resolution.assigned.get(key)
            entry = manifest.costs[entry_index] if entry_index is not None else None
            yield {
                "layer_index": job.layer_index,
                "expert_index": job.expert_index,
                "projection": job.projection,
                "bitwidth": job.bitwidth,
                "measurement": measurement,
                "weightstore_dir": job.layer_name,
                "weight_artifact": job.weight_artifact,
                "qparams_artifact": job.qparams_artifact,
                "bytes": entry.bytes if entry is not None else None,
                "cost_kind": entry.kind if entry is not None else None,
                "cost_source": entry.source if entry is not None else None,
            }


def build_summary(
    manifest: Manifest,
    jobs: Sequence[Job],
    resolution: CostResolution,
) -> dict[str, Any]:
    """Build the machine-readable planning summary."""
    directories = sorted({job.layer_name for job in jobs})
    measurements: dict[str, dict[str, int]] = {}
    for measurement in manifest.measurements:
        count = 0
        total_bytes = 0
        for key, entry_index in resolution.assigned.items():
            if key[4] != measurement:
                continue
            count += 1
            total_bytes += manifest.costs[entry_index].bytes or 0
        measurements[measurement] = {"candidates": count, "bytes": total_bytes}

    warnings: list[str] = []
    grouping_compatible = True
    if jobs and not RCO_MOE_GROUP_PATTERN.search(jobs[0].layer_name):
        grouping_compatible = False
        warnings.append(
            f"expected WeightStore directory names do not match RCO's per-expert grouping "
            f"pattern {RCO_MOE_GROUP_PATTERN.pattern}; stock rco_search_quant.py "
            "--moe-per-expert will not group these experts"
        )

    gpu_ready = resolution.issue_count == 0
    return {
        "format": FORMAT,
        "dry_run": True,
        "executes_jobs": False,
        "gpu_ready": gpu_ready,
        "model": manifest.model,
        "manifest": str(manifest.path),
        "candidate_store": manifest.candidate_store,
        "candidate_store_present": Path(manifest.candidate_store).exists(),
        "cost_model": manifest.cost_model,
        "cost_entries": len(manifest.costs),
        "topology": {
            "num_layers": manifest.topology.num_layers,
            "num_experts": manifest.topology.num_experts,
            "num_experts_per_tok": manifest.topology.num_experts_per_tok,
            "hidden_size": manifest.topology.hidden_size,
            "moe_intermediate_size": manifest.topology.moe_intermediate_size,
            "layer_types_declared": manifest.topology.layer_types is not None,
            "experts_per_layer": list(manifest.topology.experts_per_layer),
            "active_per_layer": list(manifest.topology.active_per_layer),
        },
        "selection": {
            "layers": list(manifest.selected_layers),
            "experts": (
                list(manifest.selected_experts) if manifest.selected_experts is not None else "all"
            ),
            "projections": list(manifest.projections),
            "bitwidths": list(manifest.bitwidths),
            "measurements": list(manifest.measurements),
        },
        "counts": {
            "layers": len(manifest.selected_layers),
            "experts": sum(
                len(manifest.planned_experts(layer)) for layer in manifest.selected_layers
            ),
            "projections": len(manifest.projections),
            "bitwidths": len(manifest.bitwidths),
            "measurements": len(manifest.measurements),
            "jobs": len(jobs),
            "weightstore_directories": len(directories),
            "candidate_records": sum(
                sum(
                    measurement_matches_bitwidth(measurement, job.bitwidth)
                    for measurement in manifest.measurements
                )
                for job in jobs
            ),
            "artifact_files": len(jobs) * 2,
        },
        "measurements": measurements,
        "expected": {
            "weightstore_directory_pattern": (
                f"{manifest.layer_prefix}.<layer>.mlp.{manifest.experts_prefix}."
                "<expert>.<projection>"
            ),
            "weight_artifact_pattern": "<weightstore_dir>/<bitwidth>.pth",
            "qparams_artifact_pattern": "<weightstore_dir>/<bitwidth>_qparams.pt",
            "sample_directories": directories[:5],
        },
        "rco_moe_grouping_compatible": grouping_compatible,
        "blocking_issue_count": resolution.issue_count,
        "blocking_issues": list(resolution.issues),
        "warnings": warnings,
        "non_claim": NON_CLAIM,
        "gpu_ready_non_claim": GPU_READY_NON_CLAIM,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Host-safe Qwen4-Exp candidate-store planning seam for RCO."
    )
    subparsers = parser.add_subparsers(dest="verb", required=True)
    for name, help_text in (
        ("plan", "enumerate candidate jobs and expected WeightStore paths"),
        ("check", "fail-closed GPU-readiness gate for a candidate-store manifest"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--manifest", required=True, type=Path)
        subparser.add_argument(
            "--output", type=Path, default=None, help="write the summary JSON here"
        )
        subparser.add_argument(
            "--dry-run",
            dest="dry_run",
            action="store_true",
            default=True,
            help="dry-run only; this is the only supported mode",
        )
        subparser.add_argument(
            "--execute",
            dest="dry_run",
            action="store_false",
            help="rejected: this planner never executes quantization jobs",
        )
        if name == "plan":
            subparser.add_argument(
                "--emit-jobs",
                type=Path,
                default=None,
                help="write one JSON line per candidate; paths are candidate-store relative",
            )
    return parser


def write_jobs(
    path: Path,
    manifest: Manifest,
    jobs: Sequence[Job],
    resolution: CostResolution,
) -> int:
    """Write one JSON line per planned candidate. Returns the row count."""
    parent = path.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in iter_candidate_rows(manifest, jobs, resolution):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        print(
            json.dumps(
                {
                    "format": FORMAT,
                    "passed": False,
                    "gpu_ready": False,
                    "error": "this planner is dry-run only; it never executes quantization jobs",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    try:
        manifest = load_manifest(args.manifest)
        jobs = enumerate_jobs(manifest)
        resolution = resolve_costs(manifest, jobs)
        summary = build_summary(manifest, jobs, resolution)
    except PlanError as error:
        print(
            json.dumps(
                {
                    "format": FORMAT,
                    "passed": False,
                    "gpu_ready": False,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    if args.verb == "plan" and args.emit_jobs is not None:
        summary["jobs_file"] = str(args.emit_jobs)
        summary["jobs_file_rows"] = write_jobs(args.emit_jobs, manifest, jobs, resolution)
    if args.output is not None:
        parent = args.output.parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if args.verb == "check":
        receipt = {
            "format": FORMAT,
            "stage": "qwen4exp.rco-quant-aware.candidate-store",
            "passed": summary["gpu_ready"],
            "gpu_ready": summary["gpu_ready"],
            "counts": summary["counts"],
            "blocking_issue_count": summary["blocking_issue_count"],
            "blocking_issues": summary["blocking_issues"],
            "warnings": summary["warnings"],
            "non_claim": GPU_READY_NON_CLAIM,
        }
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["gpu_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
