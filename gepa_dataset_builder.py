#!/usr/bin/env python3
"""Build GEPA training examples from the mined FPGA optimization runs.

This script reads the mined checkpoints under the mining_runs tree, walks each
benchmark from the deepest mined level back to level 0, and creates:

1. One JSONL dataset with one example per mined node.
2. One text document per benchmark and per mining level containing the relevant
   GEPA features.

The script intentionally avoids modifying anything on the remote SSH server: it
only reads the data and writes output into the local working directory.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import litellm
except ImportError:
    litellm = None

DEFAULT_MINING_ROOT = Path("~/novicad/mining_experiments/fpl26_optimization_contest/mining_runs").expanduser()
DEFAULT_OUTPUT_ROOT = Path("gepa_training_output")
DEFAULT_CONFIG_PATH = Path("gepa_dataset_config.json")

STRATEGY_NAMES = [
    "vv-place-explore",
    "vv-place-aggexp",
    "vv-route-explore",
    "vv-route-aggexp",
    "vv-physopt-explore",
    "vv-physopt-aggexp",
    "vv-physopt-altretime",
    "suggest-pblock",
]


def iter_benchmark_dirs(mining_root: Path) -> Iterable[Path]:
    if not mining_root.exists():
        raise FileNotFoundError(f"Mining root does not exist: {mining_root}")

    for child in sorted(mining_root.iterdir()):
        if not child.is_dir():
            continue
        run_json = child / "run.json"
        if run_json.exists():
            yield child


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def parse_event_strategy_map(events_path: Path) -> Dict[str, str]:
    """Return {edge_name: unit_name} for all job_claimed or unit-bearing events."""
    mapping: Dict[str, str] = {}
    if not events_path.exists():
        return mapping

    for line in events_path.read_text(errors="ignore").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        edge = record.get("edge")
        unit = record.get("unit")
        if edge and unit:
            mapping[edge] = unit

    return mapping


def find_max_depth(run_dir: Path) -> int:
    max_depth = -1
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.startswith("depth_"):
            try:
                depth = int(child.name.split("_", 1)[1])
                max_depth = max(max_depth, depth)
            except ValueError:
                continue
    return max_depth


def find_node_files_for_depth(run_dir: Path, depth: int) -> List[Path]:
    depth_dir = run_dir / f"depth_{depth}"
    if not depth_dir.exists():
        return []
    return sorted(depth_dir.glob("*.node.json"))


def get_metric(node: Dict[str, Any], *paths: str) -> Any:
    current: Any = node
    for key in paths:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def normalize_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_notes(benchmark_path: Path, node_name: str, run_dir: Path) -> str:
    """Best-effort note lookup. If no LLM-generated text exists, return empty string.

    The mining tree does not appear to maintain a dedicated LLM notes file for every
    node; if it is present it will be one of a small set of likely filenames.
    """
    candidates: List[Path] = []

    # Search around the node's reports directory.
    for report_dir in sorted((run_dir / "reports").glob("**/*")):
        if report_dir.is_dir() and node_name in report_dir.name:
            candidates.extend(report_dir.glob("*.txt"))
            candidates.extend(report_dir.glob("*.md"))
            candidates.extend(report_dir.glob("*.json"))

    # Search the benchmark root for obvious note artifacts.
    for pattern in ("*notes*", "*llm*", "*analysis*", "*summary*", "*_report*"):
        candidates.extend(sorted(run_dir.glob(pattern)))

    # Limit to textual files and give precedence to LLM-like names.
    text_files = []
    for path in candidates:
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".json"}:
            text_files.append(path)

    # Deduplicate while preserving order.
    seen = set()
    ordered: List[Path] = []
    for candidate in text_files:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)

    if not ordered:
        return ""

    # Read the first small textual artifact; in the absence of LLM notes, an empty
    # notes field is still valid for later training.
    for path in ordered[:5]:
        try:
            text = path.read_text(errors="ignore").strip()
            if text:
                return text[:4000]
        except OSError:
            continue

    return ""


def infer_critical_path_stats(node: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Pull critical path spread from the node JSON when present.

    The mined node JSON stores the timing check summary and some DCP metadata, but not
    all critical-path spread metrics. When the run has already computed them, they are
    embedded under node['critical_path_analysis']; otherwise we return None values.
    """
    candidates = [
        node.get("critical_path_analysis"),
        node.get("critical_path_spread"),
        node.get("spread_analysis"),
    ]

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        avg = normalize_float(candidate.get("avg_critical_path_distance_tiles") or candidate.get("avg_distance") or candidate.get("avg_max_distance"))
        maxd = normalize_float(candidate.get("max_critical_path_distance_tiles") or candidate.get("max_distance") or candidate.get("max_distance_found"))
        paths = candidate.get("paths_analyzed")
        if avg is not None or maxd is not None or paths is not None:
            return {
                "avg_critical_path_distance_tiles": avg,
                "max_critical_path_distance_tiles": maxd,
                "paths_analyzed": int(paths) if paths is not None else None,
            }

    return {
        "avg_critical_path_distance_tiles": None,
        "max_critical_path_distance_tiles": None,
        "paths_analyzed": None,
    }


def infer_num_high_fanout_nets(node: Dict[str, Any]) -> Optional[int]:
    for key in ("num_high_fanout_nets", "critical_high_fanout_nets", "num_critical_high_fanout_nets"):
        value = node.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    high_fanout = node.get("checks", {}).get("high_fanout_nets")
    if isinstance(high_fanout, list):
        return len(high_fanout)

    if isinstance(high_fanout, dict):
        return len(high_fanout.get("nets", []))

    return None


def choose_best_node(nodes: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best = None
    best_wns: Optional[float] = None

    for node in nodes:
        checks = node.get("checks", {})
        wns = normalize_float(checks.get("wns") or checks.get("summary", {}).get("wns"))
        if wns is None:
            continue
        if best is None or wns > best_wns:
            best = node
            best_wns = wns

    return best


def build_example_for_node(
    benchmark: str,
    benchmark_dir: Path,
    depth: int,
    node: Dict[str, Any],
    final_best_strategy: str,
    event_map: Dict[str, str],
    final_depth: int,
    notes_generator: Optional[OpenRouterNotesGenerator] = None,
) -> Dict[str, Any]:
    checks = node.get("checks", {})
    summary = checks.get("summary", {})

    wns = normalize_float(checks.get("wns") or summary.get("wns"))
    clock_period = normalize_float(checks.get("contest_period") or checks.get("period") or checks.get("clock_period"))

    critical = infer_critical_path_stats(node)
    num_high_fanout = infer_num_high_fanout_nets(node)

    if num_high_fanout is None:
        num_high_fanout = 0

    node_name = str(node.get("name") or "unknown_node")
    strategy = event_map.get(node_name, "unknown")

    notes = infer_notes(benchmark_dir, node_name, benchmark_dir)

    levels_left = max(final_depth - depth, 0)

    example = {
        "benchmark": benchmark,
        "mining_level": depth,
        "levels_left_until_final_level": levels_left,
        "worst_negative_slack": wns,
        "clock_period": clock_period,
        "avg_critical_path_distance_tiles": critical["avg_critical_path_distance_tiles"],
        "max_critical_path_distance_tiles": critical["max_critical_path_distance_tiles"],
        "paths_analyzed": critical["paths_analyzed"],
        "num_high_fanout_nets": num_high_fanout,
        "best_wns_strategy_by_final_level": final_best_strategy,
        "strategy_for_this_dcp": strategy,
        "node_name": node_name,
        "notes": notes,
    }

    if notes_generator and not notes.strip():
        llm_notes = notes_generator.generate(example)
        example["notes"] = llm_notes

    return example


def write_benchmark_text_doc(output_dir: Path, benchmark: str, depth: int, example: Dict[str, Any]) -> None:
    safe_node_name = str(example.get("node_name") or "unknown_node").replace("/", "_")
    file_path = output_dir / benchmark / f"depth_{depth}" / f"{safe_node_name}.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    text = "\n".join(
        [
            f"benchmark: {example['benchmark']}",
            f"mining_level: {example['mining_level']}",
            f"levels_left_until_final_level: {example['levels_left_until_final_level']}",
            f"worst_negative_slack: {example['worst_negative_slack']}",
            f"clock_period: {example['clock_period']}",
            f"avg_critical_path_distance_tiles: {example['avg_critical_path_distance_tiles']}",
            f"max_critical_path_distance_tiles: {example['max_critical_path_distance_tiles']}",
            f"paths_analyzed: {example['paths_analyzed']}",
            f"num_high_fanout_nets: {example['num_high_fanout_nets']}",
            f"best_wns_strategy_by_final_level: {example['best_wns_strategy_by_final_level']}",
            f"strategy_for_this_dcp: {example['strategy_for_this_dcp']}",
            "notes:",
            str(example.get("notes") or ""),
        ]
    )
    file_path.write_text(text + "\n")


def build_dataset(
    mining_root: Path,
    output_root: Path,
    notes_generator: Optional[OpenRouterNotesGenerator] = None,
) -> List[Dict[str, Any]]:
    all_examples: List[Dict[str, Any]] = []

    for benchmark_dir in iter_benchmark_dirs(mining_root):
        benchmark = benchmark_dir.name
        run_json = benchmark_dir / "run.json"
        if not run_json.exists():
            continue

        event_map = parse_event_strategy_map(benchmark_dir / "events.jsonl")
        final_depth = find_max_depth(benchmark_dir)

        final_nodes = [
            node
            for node_path in find_node_files_for_depth(benchmark_dir, final_depth)
            for node in [read_json(node_path)]
            if node is not None
        ]

        final_best = choose_best_node(final_nodes)

        final_best_strategy = "unknown"
        if final_best is not None:
            final_best_name = str(final_best.get("name") or "unknown")
            final_best_strategy = event_map.get(final_best_name, "unknown")

        for depth in range(final_depth, -1, -1):
            for node_path in find_node_files_for_depth(benchmark_dir, depth):
                node = read_json(node_path)
                if node is None:
                    continue

                example = build_example_for_node(
                    benchmark=benchmark,
                    benchmark_dir=benchmark_dir,
                    depth=depth,
                    node=node,
                    final_best_strategy=final_best_strategy,
                    event_map=event_map,
                    final_depth=final_depth,
                    notes_generator=notes_generator,
                )
                all_examples.append(example)

                write_benchmark_text_doc(output_root, benchmark, depth, example)

    return all_examples


def write_jsonl(output_root: Path, examples: Iterable[Dict[str, Any]]) -> None:
    out_path = output_root / "gepa_examples.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GEPA training examples from mined DCPs.")
    parser.add_argument("--mining-root", type=Path, default=DEFAULT_MINING_ROOT, help="Root directory containing mining_runs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory to write JSONL and text docs into.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Optional JSON config for OpenRouter note generation.")
    parser.add_argument("--generate-notes", action="store_true", help="Generate notes for examples that have no existing notes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.config)

    notes_generator = None
    if args.generate_notes:
        api_key = str(config.get("openrouter_api_key") or "").strip()
        if not api_key:
            raise ValueError("Set openrouter_api_key in gepa_dataset_config.json or OPENROUTER_API_KEY")

        notes_generator = OpenRouterNotesGenerator(
            api_key=api_key,
            model=str(config.get("notes_model", "openrouter/openai/gpt-4.1-mini")),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 300)),
            timeout_seconds=float(config.get("timeout_seconds", 120)),
        )

    examples = build_dataset(args.mining_root, args.output_root, notes_generator)
    write_jsonl(args.output_root, examples)
    print(f"Generated {len(examples)} GEPA examples in {args.output_root}")


class OpenRouterNotesGenerator:
    """Generate concise FPGA-routing notes for a mined checkpoint via OpenRouter."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        timeout_seconds: float = 120,
    ) -> None:
        if litellm is None:
            raise ImportError("LiteLLM is required. Install it with: pip install litellm")

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def generate(self, example: Dict[str, Any]) -> str:
        prompt = """You are an FPGA timing-analysis assistant.
Summarize the checkpoint state below in 2-4 concise sentences.
Explain only observations relevant to choosing the next Vivado optimization action.
Do not invent metrics or strategy results. End with one practical routing recommendation.

STATE:
""" + json.dumps(example, indent=2, ensure_ascii=False)

        try:
            response = litellm.completion(
                model=self.model,
                api_key=self.api_key,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout_seconds,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"LLM note unavailable: {exc}"


def load_dataset_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    api_key = str(config.get("openrouter_api_key") or "").strip()
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    config["openrouter_api_key"] = api_key
    return config


if __name__ == "__main__":
    main()
