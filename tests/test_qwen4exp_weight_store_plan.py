"""Tests for the host-safe Qwen4-Exp candidate-store planning seam.

The seam is deliberately import-light: these tests prove it never pulls in
Torch, never reaches the network, and never executes a quantization job.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANNER_PATH = REPO_ROOT / "qwen4exp_weight_store_plan.py"
GROUPING_PATH = REPO_ROOT / "src" / "grouping.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


planner = _load_module("qwen4exp_weight_store_plan", PLANNER_PATH)


def base_manifest() -> dict:
    """A small but fully valid Qwen4-Exp manifest."""
    scope = {
        "layers": "all",
        "experts": "all",
        "projections": ["gate_proj", "up_proj", "down_proj"],
        "bitwidths": [4, 8],
    }
    return {
        "format": planner.FORMAT,
        "model": "Qwen/Qwen3.8-Flash-Next",
        "candidate_store": "/data/qwen4exp_candidates",
        "topology": {
            "num_layers": 3,
            "num_experts": 5,
            "num_experts_per_tok": 2,
            "hidden_size": 64,
            "moe_intermediate_size": 32,
        },
        "selection": {"layers": "all", "experts": "all"},
        "bitwidths": [4, 8],
        "measurements": ["W4A16", "W4A4"],
        "cost_model": "exact",
        "costs": [
            {
                "scope": {**scope, "measurements": ["W4A16"]},
                "kind": "exact",
                "bytes": 4096,
                "source": "gguf tensor bytes",
                "components": {"tensor": 4096},
            },
            {
                "scope": {**scope, "measurements": ["W4A4"]},
                "kind": "exact",
                "bytes": 2048,
                "source": "gguf tensor bytes",
            },
        ],
    }


class ManifestCase(unittest.TestCase):
    """Base class with manifest helpers."""

    def write_manifest(self, manifest: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def plan_for(self, manifest: dict):
        path = self.write_manifest(manifest)
        loaded = planner.load_manifest(path)
        jobs = planner.enumerate_jobs(loaded)
        resolution = planner.resolve_costs(loaded, jobs)
        return loaded, jobs, resolution, planner.build_summary(loaded, jobs, resolution)


class ExactEnumerationTests(ManifestCase):
    def test_job_and_directory_counts_are_exact(self) -> None:
        manifest, jobs, resolution, summary = self.plan_for(base_manifest())
        layers = manifest.topology.num_layers
        experts = manifest.topology.num_experts
        projections = len(planner.PROJECTIONS)
        bitwidths = len(manifest.bitwidths)

        self.assertEqual(len(jobs), layers * experts * projections * bitwidths)
        self.assertEqual(len(jobs), 90)
        self.assertEqual(summary["counts"]["jobs"], 90)
        self.assertEqual(summary["counts"]["weightstore_directories"], 45)
        self.assertEqual(summary["counts"]["artifact_files"], 180)
        self.assertEqual(summary["counts"]["candidate_records"], 90)
        self.assertEqual(summary["counts"]["experts"], layers * experts)
        self.assertEqual(resolution.issue_count, 0)

    def test_every_job_has_three_projections_per_expert(self) -> None:
        manifest, jobs, _, _ = self.plan_for(base_manifest())
        keys = {(job.layer_index, job.expert_index, job.projection, job.bitwidth) for job in jobs}
        self.assertEqual(len(keys), len(jobs))
        for layer_index in range(manifest.topology.num_layers):
            for expert_index in range(manifest.topology.num_experts):
                for projection in planner.PROJECTIONS:
                    self.assertIn((layer_index, expert_index, projection, 4), keys)
                    self.assertIn((layer_index, expert_index, projection, 8), keys)

    def test_weightstore_names_and_artifact_paths(self) -> None:
        _, jobs, _, summary = self.plan_for(base_manifest())
        by_key = {
            (job.layer_index, job.expert_index, job.projection, job.bitwidth): job for job in jobs
        }
        job = by_key[(1, 2, "up_proj", 8)]
        self.assertEqual(job.layer_name, "model.layers.1.mlp.experts.2.up_proj")
        self.assertEqual(job.weight_artifact, "model.layers.1.mlp.experts.2.up_proj/8.pth")
        self.assertEqual(job.qparams_artifact, "model.layers.1.mlp.experts.2.up_proj/8_qparams.pt")
        self.assertEqual(
            summary["expected"]["weightstore_directory_pattern"],
            "model.layers.<layer>.mlp.experts.<expert>.<projection>",
        )
        self.assertIn(
            "model.layers.0.mlp.experts.0.gate_proj", summary["expected"]["sample_directories"]
        )

    def test_names_match_rco_moe_per_expert_grouping(self) -> None:
        grouping = _load_module("rco_grouping", GROUPING_PATH)
        manifest, jobs, _, _ = self.plan_for(base_manifest())
        names = [job.layer_name for job in jobs]
        unique_names = sorted(set(names))
        groups = grouping.build_moe_per_expert_groups(unique_names, [])
        self.assertEqual(len(groups), manifest.topology.num_layers * manifest.topology.num_experts)
        for group in groups:
            self.assertEqual(len(group.layer_names), len(planner.PROJECTIONS))
            self.assertRegex(group.pattern, r"^layer_\d+_expert_\d+$")
        self.assertTrue(summary_compatible := planner.RCO_MOE_GROUP_PATTERN.search(unique_names[0]))
        self.assertIsNotNone(summary_compatible)

    def test_selection_narrows_the_plan(self) -> None:
        manifest = base_manifest()
        manifest["selection"] = {"layers": [0, 2], "experts": [1, 3], "projections": ["gate_proj"]}
        manifest["costs"] = [
            {
                "scope": {
                    "layers": "all",
                    "experts": "all",
                    "projections": ["gate_proj"],
                    "bitwidths": [4, 8],
                    "measurements": ["W4A16", "W4A4"],
                },
                "kind": "exact",
                "bytes": 512,
                "source": "gguf tensor bytes",
            }
        ]
        _, jobs, resolution, summary = self.plan_for(manifest)
        self.assertEqual(len(jobs), 2 * 2 * 1 * 2)
        self.assertEqual(summary["counts"]["weightstore_directories"], 4)
        self.assertEqual(resolution.issue_count, 0)


class FailClosedCostTests(ManifestCase):
    def test_nominal_cost_kind_is_not_gpu_ready(self) -> None:
        manifest = base_manifest()
        manifest["costs"][0]["kind"] = "nominal"
        _, _, resolution, summary = self.plan_for(manifest)
        self.assertFalse(summary["gpu_ready"])
        self.assertGreater(resolution.issue_count, 0)
        self.assertTrue(
            any("not 'exact'" in issue for issue in resolution.issues),
            resolution.issues,
        )

    def test_nominal_cost_model_is_not_gpu_ready(self) -> None:
        manifest = base_manifest()
        manifest["cost_model"] = "nominal"
        _, _, resolution, summary = self.plan_for(manifest)
        self.assertFalse(summary["gpu_ready"])
        self.assertTrue(
            any("cost_model 'nominal'" in issue for issue in resolution.issues),
            resolution.issues,
        )

    def test_missing_byte_cost_is_not_gpu_ready(self) -> None:
        manifest = base_manifest()
        manifest["costs"][0]["scope"]["measurements"] = ["W4A16"]
        manifest["costs"].pop()
        _, _, resolution, summary = self.plan_for(manifest)
        self.assertFalse(summary["gpu_ready"])
        self.assertTrue(
            any("missing byte-cost metadata" in issue for issue in resolution.issues),
            resolution.issues,
        )

    def test_missing_byte_count_is_not_gpu_ready(self) -> None:
        manifest = base_manifest()
        del manifest["costs"][0]["bytes"]
        _, _, resolution, summary = self.plan_for(manifest)
        self.assertFalse(summary["gpu_ready"])
        self.assertTrue(
            any("no exact byte count" in issue for issue in resolution.issues),
            resolution.issues,
        )

    def test_missing_source_provenance_is_not_gpu_ready(self) -> None:
        manifest = base_manifest()
        del manifest["costs"][0]["source"]
        _, _, resolution, summary = self.plan_for(manifest)
        self.assertFalse(summary["gpu_ready"])
        self.assertTrue(
            any("no source provenance" in issue for issue in resolution.issues),
            resolution.issues,
        )

    def test_ambiguous_cost_coverage_is_rejected(self) -> None:
        manifest = base_manifest()
        duplicate = json.loads(json.dumps(manifest["costs"][0]))
        manifest["costs"].append(duplicate)
        _, _, resolution, summary = self.plan_for(manifest)
        self.assertFalse(summary["gpu_ready"])
        self.assertTrue(
            any("ambiguous byte-cost metadata" in issue for issue in resolution.issues),
            resolution.issues,
        )

    def test_components_must_sum_to_declared_bytes(self) -> None:
        manifest = base_manifest()
        manifest["costs"][0]["components"] = {"tensor": 1, "scales": 2}
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("components sum to 3", str(caught.exception))


class ManifestValidationTests(ManifestCase):
    def test_wrong_format_is_rejected(self) -> None:
        manifest = base_manifest()
        manifest["format"] = "something-else"
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("expected format", str(caught.exception))

    def test_active_count_cannot_exceed_expert_count(self) -> None:
        manifest = base_manifest()
        manifest["topology"]["num_experts_per_tok"] = 6
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("must not exceed", str(caught.exception))

    def test_layer_types_length_must_match_layer_count(self) -> None:
        manifest = base_manifest()
        manifest["topology"]["layer_types"] = ["linear_attention", "full_attention"]
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("layer_types has 2 entries", str(caught.exception))

    def test_experts_per_layer_must_cover_every_layer(self) -> None:
        manifest = base_manifest()
        manifest["topology"]["experts_per_layer"] = {"0": 5, "1": 5}
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("experts_per_layer has 2 entries", str(caught.exception))

    def test_active_per_layer_cannot_exceed_that_layer(self) -> None:
        manifest = base_manifest()
        manifest["topology"]["experts_per_layer"] = {"0": 5, "1": 2, "2": 5}
        manifest["topology"]["active_per_layer"] = {"0": 2, "1": 3, "2": 2}
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("must not exceed that layer's expert count", str(caught.exception))

    def test_unsupported_measurement_label_is_rejected(self) -> None:
        manifest = base_manifest()
        manifest["measurements"] = ["W8A16"]
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("is not supported", str(caught.exception))

    def test_duplicate_measurement_labels_are_rejected(self) -> None:
        manifest = base_manifest()
        manifest["measurements"] = ["W4A16", "W4A16"]
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("duplicate labels", str(caught.exception))

    def test_measurement_weight_bits_must_be_plannable(self) -> None:
        manifest = base_manifest()
        manifest["bitwidths"] = [8]
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("needs weight bitwidth 4", str(caught.exception))

    def test_selection_expert_outside_layer_is_rejected(self) -> None:
        manifest = base_manifest()
        manifest["selection"] = {"layers": "all", "experts": [9]}
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("selection.experts[0]", str(caught.exception))

    def test_cost_scope_outside_the_plan_is_rejected(self) -> None:
        manifest = base_manifest()
        manifest["costs"][0]["scope"]["layers"] = [7]
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("outside the selection", str(caught.exception))

    def test_unknown_cost_scope_key_is_rejected(self) -> None:
        manifest = base_manifest()
        manifest["costs"][0]["scope"]["tensors"] = ["x"]
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("unsupported keys", str(caught.exception))

    def test_cost_scope_list_fields_reject_scalars_and_nulls(self) -> None:
        for field in ("projections", "bitwidths", "measurements"):
            with self.subTest(field=field):
                manifest = base_manifest()
                manifest["costs"][0]["scope"][field] = 4 if field == "bitwidths" else None
                with self.assertRaises(planner.PlanError) as caught:
                    self.plan_for(manifest)
                self.assertIn(
                    f"costs[0].scope.{field} must be a non-empty list", str(caught.exception)
                )

    def test_measurement_labels_do_not_cross_non_matching_bitwidths(self) -> None:
        manifest = base_manifest()
        manifest["costs"][0]["scope"]["bitwidths"] = [8]
        manifest["costs"][1]["scope"]["bitwidths"] = [4]
        _, _, resolution, summary = self.plan_for(manifest)
        self.assertFalse(summary["gpu_ready"])
        self.assertTrue(
            any(
                "covers no planned measurement-compatible candidate" in issue
                for issue in resolution.issues
            )
        )
        self.assertEqual(summary["counts"]["candidate_records"], 90)


class HostSafetyTests(unittest.TestCase):
    """The seam must stay usable on a host without Torch, CUDA, or a network."""

    FORBIDDEN_IMPORTS = (
        "torch",
        "transformers",
        "numpy",
        "requests",
        "urllib",
        "socket",
        "http",
        "subprocess",
        "accelerate",
        "datasets",
    )

    def test_source_has_no_forbidden_imports(self) -> None:
        source = PLANNER_PATH.read_text(encoding="utf-8")
        for module in self.FORBIDDEN_IMPORTS:
            pattern = rf"^\s*(?:import|from)\s+{re.escape(module)}\b"
            self.assertIsNone(
                re.search(pattern, source, re.MULTILINE),
                f"{PLANNER_PATH.name} must not import {module}",
            )

    def test_importing_the_planner_does_not_pull_in_torch_or_network_modules(self) -> None:
        probe = (
            "import sys, qwen4exp_weight_store_plan; "
            "watched = {'torch', 'requests', 'urllib.request', 'http.client', 'socket', "
            "'subprocess', 'transformers', 'numpy', 'accelerate', 'datasets'}; "
            "print(sorted(m for m in sys.modules if m in watched))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[]")

    def test_plan_is_declared_dry_run(self) -> None:
        source = PLANNER_PATH.read_text(encoding="utf-8")
        self.assertIn('"dry_run": True', source)
        self.assertIn('"executes_jobs": False', source)


class CommandLineTests(ManifestCase):
    def run_cli(self, argv: list[str]) -> tuple[int, dict]:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = planner.main(argv)
        return code, json.loads(buffer.getvalue())

    def test_plan_reports_exact_counts(self) -> None:
        path = self.write_manifest(base_manifest())
        code, payload = self.run_cli(["plan", "--manifest", str(path)])
        self.assertEqual(code, 0)
        self.assertTrue(payload["gpu_ready"])
        self.assertTrue(payload["dry_run"])
        self.assertFalse(payload["executes_jobs"])
        self.assertEqual(payload["counts"]["jobs"], 90)
        self.assertEqual(payload["counts"]["weightstore_directories"], 45)
        self.assertEqual(payload["counts"]["artifact_files"], 180)

    def test_check_passes_only_with_exact_costs(self) -> None:
        path = self.write_manifest(base_manifest())
        code, payload = self.run_cli(["check", "--manifest", str(path)])
        self.assertEqual(code, 0)
        self.assertTrue(payload["passed"])

    def test_check_fails_closed_on_nominal_costs(self) -> None:
        manifest = base_manifest()
        manifest["costs"][0]["kind"] = "nominal"
        path = self.write_manifest(manifest)
        code, payload = self.run_cli(["check", "--manifest", str(path)])
        self.assertEqual(code, 1)
        self.assertFalse(payload["passed"])
        self.assertFalse(payload["gpu_ready"])
        self.assertGreater(payload["blocking_issue_count"], 0)

    def test_invalid_manifest_exits_one_with_an_error(self) -> None:
        path = self.write_manifest({"format": planner.FORMAT})
        code, payload = self.run_cli(["plan", "--manifest", str(path)])
        self.assertEqual(code, 1)
        self.assertFalse(payload["gpu_ready"])
        self.assertIn("error", payload)

    def test_execute_is_rejected(self) -> None:
        path = self.write_manifest(base_manifest())
        code, payload = self.run_cli(["plan", "--manifest", str(path), "--execute"])
        self.assertEqual(code, 2)
        self.assertFalse(payload["gpu_ready"])
        self.assertIn("dry-run only", payload["error"])

    def test_emit_jobs_writes_one_row_per_candidate(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        jobs_path = Path(directory.name) / "jobs.jsonl"
        path = self.write_manifest(base_manifest())
        code, payload = self.run_cli(
            ["plan", "--manifest", str(path), "--emit-jobs", str(jobs_path)]
        )
        self.assertEqual(code, 0)
        rows = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), payload["counts"]["candidate_records"])
        self.assertEqual(len(rows), 90)
        first = rows[0]
        self.assertEqual(first["weightstore_dir"], "model.layers.0.mlp.experts.0.gate_proj")
        self.assertEqual(first["weight_artifact"], "model.layers.0.mlp.experts.0.gate_proj/4.pth")
        self.assertEqual(
            first["qparams_artifact"], "model.layers.0.mlp.experts.0.gate_proj/4_qparams.pt"
        )
        self.assertEqual(first["measurement"], "W4A16")
        self.assertEqual(first["bytes"], 4096)
        self.assertEqual(first["cost_kind"], "exact")


class PerLayerExpertTests(ManifestCase):
    """Heterogeneous per-layer expert counts must fail closed, not narrow."""

    def heterogeneous_manifest(self) -> dict:
        manifest = base_manifest()
        manifest["topology"]["experts_per_layer"] = {"0": 5, "1": 3, "2": 5}
        manifest["topology"]["active_per_layer"] = {"0": 2, "1": 2, "2": 2}
        manifest["costs"] = [
            {
                "scope": {
                    "layers": "all",
                    "experts": "all",
                    "projections": ["gate_proj", "up_proj", "down_proj"],
                    "bitwidths": [4, 8],
                    "measurements": ["W4A16", "W4A4"],
                },
                "kind": "exact",
                "bytes": 4096,
                "source": "gguf tensor bytes",
            }
        ]
        return manifest

    def test_explicit_cost_scope_experts_cover_the_plan(self) -> None:
        # Regression: an explicit expert list in a cost scope must resolve to
        # the same candidates as the equivalent "all" scope.
        manifest = base_manifest()
        manifest["costs"] = [
            {
                "scope": {
                    "layers": "all",
                    "experts": [0, 1, 2, 3, 4],
                    "projections": ["gate_proj", "up_proj", "down_proj"],
                    "bitwidths": [4, 8],
                    "measurements": ["W4A16", "W4A4"],
                },
                "kind": "exact",
                "bytes": 4096,
                "source": "gguf tensor bytes",
            }
        ]
        _, jobs, resolution, summary = self.plan_for(manifest)
        self.assertEqual(len(jobs), 90)
        self.assertEqual(resolution.issue_count, 0)
        self.assertTrue(summary["gpu_ready"])

    def test_cost_scope_expert_outside_every_layer_is_rejected(self) -> None:
        manifest = base_manifest()
        manifest["costs"][0]["scope"]["experts"] = [0, 9]
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("do not exist in every selected layer: [9]", str(caught.exception))

    def test_selected_expert_above_the_expert_count_is_rejected(self) -> None:
        manifest = base_manifest()
        manifest["selection"] = {"layers": "all", "experts": [5]}
        with self.assertRaises(planner.PlanError) as caught:
            self.plan_for(manifest)
        self.assertIn("selection.experts[0]", str(caught.exception))

    def test_per_layer_counts_honored_when_selecting_all_experts(self) -> None:
        _, jobs, resolution, summary = self.plan_for(self.heterogeneous_manifest())
        self.assertEqual(summary["counts"]["experts"], 13)
        self.assertEqual(summary["counts"]["jobs"], 13 * 3 * 2)
        self.assertEqual(summary["counts"]["weightstore_directories"], 13 * 3)
        self.assertEqual(resolution.issue_count, 0)
        self.assertEqual(summary["topology"]["active_per_layer"], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
