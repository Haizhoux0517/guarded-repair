import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


ABLATION_ROOT = Path("outputs/ablations")

VARIANTS = [
    "A_full_system",
    "B_no_format_retry",
    "C_no_clean_semantic_improvement",
    "D_no_deterministic_hint",
]

OUTPUT_JSON = ABLATION_ROOT / "case_comparison.json"
OUTPUT_MD = ABLATION_ROOT / "case_comparison.md"
OUTPUT_TABLE_JSONL = ABLATION_ROOT / "case_status_table.jsonl"


CASE_FILES = {
    "fixed": "fixed_cases.jsonl",
    "broken": "broken_cases.jsonl",
    "remaining": "remaining_errors.jsonl",
    "correct_but_rejected": "correct_but_rejected.jsonl",
    "initially_wrong_correct_but_rejected": "initially_wrong_candidate_correct_but_rejected.jsonl",
    "initially_wrong_graph_guard_rejected": "initially_wrong_graph_guard_rejected.jsonl",
    "initially_wrong_format_failed": "initially_wrong_format_failed.jsonl",
    "initially_wrong_candidate_wrong": "initially_wrong_candidate_wrong.jsonl",
    "initially_wrong_no_candidate": "initially_wrong_no_candidate.jsonl",
}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}, line {line_idx}: {e}") from e

    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def case_id(row: Dict[str, Any]) -> str:
    return str(row.get("id", "")).strip()


def ids_from_rows(rows: List[Dict[str, Any]]) -> Set[str]:
    return {case_id(row) for row in rows if case_id(row)}


def index_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        cid = case_id(row)
        if cid and cid not in indexed:
            indexed[cid] = row

    return indexed


def compact_case(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row:
        return {}

    return {
        "id": row.get("id"),
        "problem": row.get("problem"),
        "gold_answer": row.get("gold_answer"),
        "initial_answer": row.get("initial_answer"),
        "candidate_answer": row.get("candidate_answer"),
        "final_answer": row.get("final_answer"),
        "initial_correct": row.get("initial_correct"),
        "candidate_correct": row.get("candidate_correct"),
        "final_correct": row.get("final_correct"),
        "repair_accepted": row.get("repair_accepted"),
        "repair_source": row.get("repair_source"),
        "initial_meta_error_type": row.get("initial_meta_error_type"),
        "candidate_meta_error_type": row.get("candidate_meta_error_type"),
        "initial_graph_error_type": row.get("initial_graph_error_type"),
        "candidate_graph_error_type": row.get("candidate_graph_error_type"),
        "acceptance_reason": row.get("acceptance_reason"),
        "graph_guard_reason": row.get("graph_guard_reason"),
        "rejection_class": row.get("rejection_class"),
    }


def collect_variant(variant: str) -> Dict[str, Any]:
    variant_dir = ABLATION_ROOT / variant
    analysis_dir = variant_dir / "analysis"

    final_report = load_json(variant_dir / "final_report.json")
    analysis_summary = load_json(analysis_dir / "analysis_summary.json")

    case_rows: Dict[str, List[Dict[str, Any]]] = {}
    case_ids: Dict[str, Set[str]] = {}
    all_case_index: Dict[str, Dict[str, Any]] = {}

    for key, filename in CASE_FILES.items():
        rows = load_jsonl(analysis_dir / filename)
        case_rows[key] = rows
        case_ids[key] = ids_from_rows(rows)

        for cid, row in index_rows(rows).items():
            if cid not in all_case_index:
                all_case_index[cid] = row

    return {
        "variant": variant,
        "variant_dir": str(variant_dir),
        "analysis_dir": str(analysis_dir),
        "final_report": final_report,
        "analysis_summary": analysis_summary,
        "case_rows": case_rows,
        "case_ids": case_ids,
        "case_index": all_case_index,
    }


def sorted_ids(ids: Set[str]) -> List[str]:
    def sort_key(x: str) -> Tuple[int, Any]:
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    return sorted(ids, key=sort_key)


def status_for_case(variant_data: Dict[str, Any], cid: str) -> str:
    ids = variant_data["case_ids"]

    if cid in ids.get("fixed", set()):
        return "fixed"

    if cid in ids.get("broken", set()):
        return "broken"

    if cid in ids.get("initially_wrong_correct_but_rejected", set()):
        return "candidate_correct_rejected"

    if cid in ids.get("initially_wrong_graph_guard_rejected", set()):
        return "graph_guard_rejected"

    if cid in ids.get("initially_wrong_format_failed", set()):
        return "format_failed"

    if cid in ids.get("initially_wrong_candidate_wrong", set()):
        return "candidate_wrong"

    if cid in ids.get("initially_wrong_no_candidate", set()):
        return "no_candidate"

    if cid in ids.get("remaining", set()):
        return "remaining"

    return "not_listed"


def build_case_status_table(all_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_ids: Set[str] = set()

    for variant_data in all_data.values():
        for ids in variant_data["case_ids"].values():
            all_ids.update(ids)

    table: List[Dict[str, Any]] = []

    for cid in sorted_ids(all_ids):
        row: Dict[str, Any] = {"id": cid}

        representative_case = None

        for variant in VARIANTS:
            variant_data = all_data[variant]
            status = status_for_case(variant_data, cid)
            row[variant] = status

            if representative_case is None:
                representative_case = variant_data["case_index"].get(cid)

        if representative_case:
            compact = compact_case(representative_case)
            row.update(
                {
                    "problem": compact.get("problem"),
                    "gold_answer": compact.get("gold_answer"),
                    "initial_answer": compact.get("initial_answer"),
                    "initial_meta_error_type": compact.get("initial_meta_error_type"),
                    "initial_graph_error_type": compact.get("initial_graph_error_type"),
                }
            )

        table.append(row)

    return table


def pairwise_fixed_comparison(all_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    fixed_sets = {
        variant: all_data[variant]["case_ids"].get("fixed", set())
        for variant in VARIANTS
    }

    comparison: Dict[str, Any] = {}

    baseline = "A_full_system"
    baseline_fixed = fixed_sets[baseline]

    for variant in VARIANTS:
        current_fixed = fixed_sets[variant]

        comparison[variant] = {
            "fixed_ids": sorted_ids(current_fixed),
            "num_fixed": len(current_fixed),
            "only_in_this_variant_vs_A": sorted_ids(current_fixed - baseline_fixed),
            "missing_from_this_variant_vs_A": sorted_ids(baseline_fixed - current_fixed),
            "overlap_with_A": sorted_ids(current_fixed & baseline_fixed),
        }

    all_fixed_intersection = set.intersection(*fixed_sets.values()) if fixed_sets else set()
    all_fixed_union = set.union(*fixed_sets.values()) if fixed_sets else set()

    comparison["global"] = {
        "fixed_by_all_variants": sorted_ids(all_fixed_intersection),
        "fixed_by_any_variant": sorted_ids(all_fixed_union),
        "num_fixed_by_all_variants": len(all_fixed_intersection),
        "num_fixed_by_any_variant": len(all_fixed_union),
    }

    for variant in VARIANTS:
        other_union = set()

        for other_variant in VARIANTS:
            if other_variant == variant:
                continue
            other_union.update(fixed_sets[other_variant])

        comparison[variant]["only_in_this_variant_global"] = sorted_ids(
            fixed_sets[variant] - other_union
        )

    return comparison


def build_summary(all_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    variant_summary: Dict[str, Any] = {}

    for variant, data in all_data.items():
        final_report = data["final_report"]
        analysis = data["analysis_summary"].get("analysis", {})

        variant_summary[variant] = {
            "final_accuracy": final_report.get("final_accuracy"),
            "absolute_improvement": final_report.get("absolute_improvement"),
            "fixed_errors": final_report.get("fixed_errors"),
            "broke_correct": final_report.get("broke_correct"),
            "harm_rate": final_report.get("harm_rate"),
            "accepted_repair_precision": final_report.get("accepted_repair_precision"),
            "num_repairs_accepted": final_report.get("num_repairs_accepted"),
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
            "initially_wrong_graph_guard_rejected": analysis.get(
                "initially_wrong_graph_guard_rejected"
            ),
            "initially_wrong_format_failed": analysis.get(
                "initially_wrong_format_failed"
            ),
        }

    return variant_summary


def markdown_list(ids: List[str]) -> str:
    if not ids:
        return "-"

    return ", ".join(ids)


def save_markdown_report(
    path: Path,
    summary: Dict[str, Any],
    fixed_comparison: Dict[str, Any],
    case_table: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []

    lines.append("# Ablation Case Comparison")
    lines.append("")

    lines.append("## Variant Summary")
    lines.append("")
    lines.append(
        "| Variant | Final Acc | Fixed | Broke | Harm | Accepted Precision | Format Retry | Hint | Clean Semantic Accept | LLM Format Failures |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|---:|")

    for variant in VARIANTS:
        s = summary[variant]
        lines.append(
            "| {variant} | {final_accuracy} | {fixed_errors} | {broke_correct} | {harm_rate} | {accepted_precision} | {format_retry} | {hint} | {clean_semantic} | {format_failures} |".format(
                variant=variant,
                final_accuracy=s.get("final_accuracy"),
                fixed_errors=s.get("fixed_errors"),
                broke_correct=s.get("broke_correct"),
                harm_rate=s.get("harm_rate"),
                accepted_precision=s.get("accepted_repair_precision"),
                format_retry=s.get("format_retry_enabled"),
                hint=s.get("deterministic_hint_enabled"),
                clean_semantic=s.get("clean_semantic_improvement_accept_enabled"),
                format_failures=s.get("llm_format_failures"),
            )
        )

    lines.append("")
    lines.append("## Fixed Case Overlap")
    lines.append("")

    global_cmp = fixed_comparison["global"]

    lines.append(f"- Fixed by all variants: `{markdown_list(global_cmp['fixed_by_all_variants'])}`")
    lines.append(f"- Fixed by any variant: `{markdown_list(global_cmp['fixed_by_any_variant'])}`")
    lines.append("")

    lines.append("| Variant | Num Fixed | Fixed IDs | Only vs A | Missing vs A | Only Global |")
    lines.append("|---|---:|---|---|---|---|")

    for variant in VARIANTS:
        cmp_data = fixed_comparison[variant]
        lines.append(
            "| {variant} | {num_fixed} | {fixed_ids} | {only_a} | {missing_a} | {only_global} |".format(
                variant=variant,
                num_fixed=cmp_data["num_fixed"],
                fixed_ids=markdown_list(cmp_data["fixed_ids"]),
                only_a=markdown_list(cmp_data["only_in_this_variant_vs_A"]),
                missing_a=markdown_list(cmp_data["missing_from_this_variant_vs_A"]),
                only_global=markdown_list(cmp_data["only_in_this_variant_global"]),
            )
        )

    lines.append("")
    lines.append("## Case Status Table")
    lines.append("")
    lines.append("| ID | A | B | C | D | Gold | Initial | Initial Graph | Initial Meta |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for row in case_table:
        lines.append(
            "| {id} | {a} | {b} | {c} | {d} | {gold} | {initial} | {graph} | {meta} |".format(
                id=row.get("id", ""),
                a=row.get("A_full_system", ""),
                b=row.get("B_no_format_retry", ""),
                c=row.get("C_no_clean_semantic_improvement", ""),
                d=row.get("D_no_deterministic_hint", ""),
                gold=row.get("gold_answer", ""),
                initial=row.get("initial_answer", ""),
                graph=row.get("initial_graph_error_type", ""),
                meta=row.get("initial_meta_error_type", ""),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_data: Dict[str, Dict[str, Any]] = {}

    for variant in VARIANTS:
        variant_dir = ABLATION_ROOT / variant

        if not variant_dir.exists():
            raise FileNotFoundError(
                f"Missing ablation directory: {variant_dir}\n"
                "Please run run_ablation_suite.py first."
            )

        all_data[variant] = collect_variant(variant)

    summary = build_summary(all_data)
    fixed_comparison = pairwise_fixed_comparison(all_data)
    case_table = build_case_status_table(all_data)

    output = {
        "ablation_root": str(ABLATION_ROOT),
        "variants": VARIANTS,
        "summary": summary,
        "fixed_comparison": fixed_comparison,
        "case_status_table_path": str(OUTPUT_TABLE_JSONL),
    }

    save_json(OUTPUT_JSON, output)
    save_jsonl(OUTPUT_TABLE_JSONL, case_table)
    save_markdown_report(
        path=OUTPUT_MD,
        summary=summary,
        fixed_comparison=fixed_comparison,
        case_table=case_table,
    )

    print("Ablation case comparison finished.")
    print(f"Saved JSON summary to: {OUTPUT_JSON}")
    print(f"Saved Markdown report to: {OUTPUT_MD}")
    print(f"Saved case status table to: {OUTPUT_TABLE_JSONL}")

    print("\nFixed case overlap:")
    print(json.dumps(fixed_comparison["global"], ensure_ascii=False, indent=2))

    print("\nVariant fixed IDs:")
    for variant in VARIANTS:
        cmp_data = fixed_comparison[variant]
        print(f"{variant}: {cmp_data['fixed_ids']}")


if __name__ == "__main__":
    main()