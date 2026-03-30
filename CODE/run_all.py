"""
Single entry point: run every analysis script in order and verify outputs.

Usage:
    python analysis/run_all.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_REPORTS = ROOT.parent / "OUTPUTS" / "reports"
OUTPUT_FIGURES = ROOT.parent / "OUTPUTS" / "figures"
OUTPUT_TABLES = ROOT.parent / "OUTPUTS" / "tables"

# Execution order — each entry is (module_name, expected_outputs)
PIPELINE = [
    ("descriptive_tables", [
        OUTPUT_TABLES / "data_coverage_table.tex",
        OUTPUT_TABLES / "descriptive_statistics_table.tex",
        OUTPUT_TABLES / "h1_descriptive_statistics.tex",
        OUTPUT_TABLES / "h2_spread_statistics.tex",
        OUTPUT_TABLES / "macro_summary_table.tex",
    ]),
    ("h1_ecb_rate_hike_impact", [
        OUTPUT_REPORTS / "h1_regression_results.txt",
        OUTPUT_REPORTS / "h1_placebo_results.txt",
        OUTPUT_FIGURES / "h1_placebo_tests.png",
    ]),
    ("h1_hac_vif_appendix", [
        OUTPUT_REPORTS / "h1_hac_results.txt",
    ]),
    ("h1_event_study", [
        OUTPUT_REPORTS / "h1_event_study_results.txt",
        OUTPUT_FIGURES / "h1_event_study_window5.png",
        OUTPUT_FIGURES / "h1_event_study_window3.png",
    ]),
    ("h1b_ecb_feb2023_impact", [
        OUTPUT_REPORTS / "h1b_regression_results.txt",
        OUTPUT_FIGURES / "h1b_robustness_checks.png",
    ]),
    ("h2_euro_adoption_impact", [
        OUTPUT_REPORTS / "h2_regression_results.txt",
        OUTPUT_REPORTS / "h2_placebo_results.txt",
        OUTPUT_FIGURES / "h2_placebo_tests.png",
        OUTPUT_FIGURES / "h2_spread_convergence_did.png",
        OUTPUT_FIGURES / "h2_spread_timeseries.png",
        OUTPUT_FIGURES / "h2_robustness_checks.png",
    ]),
    ("h2_hac_vif_appendix", [
        OUTPUT_REPORTS / "h2_hac_results.txt",
    ]),
    ("h2_event_study", [
        OUTPUT_REPORTS / "h2_event_study_results.txt",
        OUTPUT_FIGURES / "h2_event_study_window5.png",
        OUTPUT_FIGURES / "h2_event_study_window3.png",
    ]),
    ("f_test_comparison", [
        OUTPUT_REPORTS / "f_test_comparison_results.txt",
    ]),
]


def main() -> None:
    # Ensure the analysis package is importable
    sys.path.insert(0, str(ROOT.parent))
    sys.path.insert(0, str(ROOT))

    passed, failed = 0, 0
    t0 = time.time()

    for module_name, expected_files in PIPELINE:
        print(f"\n{'=' * 60}")
        print(f"  Running {module_name}")
        print(f"{'=' * 60}")

        try:
            mod = __import__(module_name)
            mod.run()
        except Exception as exc:
            print(f"  [FAIL] {module_name}: {exc}")
            failed += 1
            continue

        # Verify outputs
        missing = [f for f in expected_files if not f.exists()]
        if missing:
            for f in missing:
                print(f"  [MISSING] {f.name}")
            failed += 1
        else:
            print(f"  [OK] {len(expected_files)} output(s) verified")
            passed += 1

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  DONE: {passed} passed, {failed} failed  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
