import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ABLATION_ROOT = Path("outputs/ablations")

VARIANTS = [
    {
        "name": "A_full_system",
        "description": (
            "Full system: deterministic hint + JSON LLM repair + "
            "format retry + restricted clean semantic-improvement acceptance"
        ),
        "env": {
            "ENABLE_DETERMINISTIC_HINT": "1",
            "ENABLE_FORMAT_RETRY": "1",
            "ENABLE_CLEAN_SEMANTIC_IMPROVEMENT_ACCEPT": "1",
            "ENABLE_LLM_REPAIR": "1",
            "ENABLE_GRAPH_GUARD": "1",
        },
    },
    {
        "name": "B_no_format_retry",
        "description": "Disable generic format retry",
        "env": {
            "ENABLE_DETERMINISTIC_HINT": "1",
            "ENABLE_FORMAT_RETRY": "0",
            "ENABLE_CLEAN_SEMANTIC_IMPROVEMENT_ACCEPT": "1",
            "ENABLE_LLM_REPAIR": "1",
            "ENABLE_GRAPH_GUARD": "1",
        },
    },
    {
        "name": "C_no_clean_semantic_improvement",
        "description": "Disable restricted clean semantic-improvement acceptance",
        "env": {
            "ENABLE_DETERMINISTIC_HINT": "1",
            "ENABLE_FORMAT_RETRY": "1",
            "ENABLE_CLEAN_SEMANTIC_IMPROVEMENT_ACCEPT": "0",
            "ENABLE_LLM_REPAIR": "1",
            "ENABLE_GRAPH_GUARD": "1",
        },
    },
    {
        "name": "D_no_deterministic_hint",
        "description": "Disable deterministic hint, keep LLM repair and guards",
        "env": {
            "ENABLE_DETERMINISTIC_HINT": "0",
            "ENABLE_FORMAT_RETRY": "1",
            "ENABLE_CLEAN_SEMANTIC_IMPROVEMENT_ACCEPT": "1",
            "ENABLE_LLM_REPAIR": "1",
            "ENABLE_GRAPH_GUARD": "1",
        },
    },
    {
        "name": "E_no_llm_repair",
        "description": "Disable LLM repair; diagnostic-only baseline",
        "env": {
            "ENABLE_DETERMINISTIC_HINT": "1",
            "ENABLE_FORMAT_RETRY": "1",
            "ENABLE_CLEAN_SEMANTIC_IMPROVEMENT_ACCEPT": "1",
            "ENABLE_LLM_REPAIR": "0",
            "ENABLE_GRAPH_GUARD": "1",
        },
    },
]


def run_command(command: List[str], env: Dict[str, str]) -> None:
    print("\n" + "=" * 100)
    print("Running:", " ".join(command))
    print("=" * 100)

    subprocess.run(
        command,
        check=True,
        env=env,
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_markdown_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "variant",
        "initial_accuracy",
        "final_accuracy",
        "absolute_improvement",
        "fixed_errors",
        "broke_correct",
        "harm_rate",
        "accepted_repair_precision",
        "error_repair_rate",
        "num_repairs_accepted",
        "format_retry_used",
        "llm_format_failures",
        "initially_wrong_candidate_correct_count",
        "initially_wrong_candidate_correct_but_rejected",
        "initially_wrong_candidate_missing_or_unparseable",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_metrics(
    variant_name: str,
    description: str,
    final_report: Dict[str, Any],
    analysis_summary: Dict[str, Any],
) -> Dict[str, Any]:
    analysis = analysis_summary.get("analysis", {})

    return {
        "variant": variant_name,
        "description": description,

        "total": final_report.get("total"),
        "initial_correct": final_report.get("initial_correct"),
        "final_correct": final_report.get("final_correct"),
        "initial_accuracy": final_report.get("initial_accuracy"),
        "final_accuracy": final_report.get("final_accuracy"),
        "absolute_improvement": final_report.get("absolute_improvement"),

        "initial_wrong": final_report.get("initial_wrong"),
        "fixed_errors": final_report.get("fixed_errors"),
        "broke_correct": final_report.get("broke_correct"),
        "failed_to_fix": final_report.get("failed_to_fix"),

        "num_repair_candidates": final_report.get("num_repair_candidates"),
        "num_repairs_accepted": final_report.get("num_repairs_accepted"),
        "repair_acceptance_rate": final_report.get("repair_acceptance_rate"),
        "accepted_repair_precision": final_report.get("accepted_repair_precision"),
        "error_repair_rate": final_report.get("error_repair_rate"),
        "harm_rate": final_report.get("harm_rate"),

        "format_retry_enabled": final_report.get("format_retry_enabled"),
        "format_retry_used": final_report.get("format_retry_used"),
        "deterministic_hint_enabled": final_report.get("deterministic_hint_enabled"),
        "clean_semantic_improvement_accept_enabled": final_report.get(
            "clean_semantic_improvement_accept_enabled"
        ),

        "llm_format_failures": analysis.get("llm_format_failures"),
        "initially_wrong_candidate_correct_count": analysis.get(
            "initially_wrong_candidate_correct_count"
        ),
        "initially_wrong_candidate_correct_but_rejected": analysis.get(
            "initially_wrong_candidate_correct_but_rejected"
        ),
        "initially_wrong_candidate_wrong_count": analysis.get(
            "initially_wrong_candidate_wrong_count"
        ),
        "initially_wrong_candidate_missing_or_unparseable": analysis.get(
            "initially_wrong_candidate_missing_or_unparseable"
        ),
        "initially_wrong_format_failed": analysis.get(
            "initially_wrong_format_failed"
        ),
    }


def main() -> None:
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    base_env = os.environ.copy()
    python_executable = sys.executable

    for variant in VARIANTS:
        name = variant["name"]
        description = variant["description"]

        output_dir = ABLATION_ROOT / name
        output_dir.mkdir(parents=True, exist_ok=True)

        env = base_env.copy()
        env.update(variant["env"])
        env["GUARDED_OUTPUT_DIR"] = str(output_dir)

        print("\n" + "#" * 100)
        print(f"Variant: {name}")
        print(description)
        print(f"Output dir: {output_dir}")
        print("#" * 100)

        run_command([python_executable, "run_pipeline_guarded_from_cache.py"], env)
        run_command([python_executable, "analyze_guarded_results.py"], env)

        final_report = load_json(output_dir / "final_report.json")
        analysis_summary = load_json(output_dir / "analysis" / "analysis_summary.json")

        row = compact_metrics(
            variant_name=name,
            description=description,
            final_report=final_report,
            analysis_summary=analysis_summary,
        )

        all_rows.append(row)

        save_json(output_dir / "compact_metrics.json", row)

    save_json(ABLATION_ROOT / "ablation_summary.json", all_rows)
    save_markdown_table(ABLATION_ROOT / "ablation_summary.md", all_rows)

    print("\nAblation suite finished.")
    print(f"Saved JSON summary to: {ABLATION_ROOT / 'ablation_summary.json'}")
    print(f"Saved Markdown summary to: {ABLATION_ROOT / 'ablation_summary.md'}")

    print("\nCompact results:")
    for row in all_rows:
        print(
            f"{row['variant']}: "
            f"initial={row.get('initial_accuracy')}, "
            f"final={row.get('final_accuracy')}, "
            f"fixed={row.get('fixed_errors')}, "
            f"broke={row.get('broke_correct')}, "
            f"harm={row.get('harm_rate')}"
        )


if __name__ == "__main__":
    main()
