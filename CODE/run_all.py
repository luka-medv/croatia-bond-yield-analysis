"""
Single entry point: run the lean paper replication pipeline and verify outputs.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_REPORTS = ROOT.parent / "OUTPUTS" / "reports"
OUTPUT_FIGURES = ROOT.parent / "OUTPUTS" / "figures"
DATA_DIR = ROOT.parent / "DATA"

PIPELINE = [
    ("descriptive_exports", [
        DATA_DIR / "descriptive_stats_macro.csv",
        DATA_DIR / "descriptive_stats_h1_yields.csv",
        DATA_DIR / "descriptive_stats_h2_spreads.csv",
    ]),
    ("h1_ecb_rate_hike_impact", [
        OUTPUT_REPORTS / "h1_regression_results.txt",
        OUTPUT_REPORTS / "h1_placebo_results.txt",
    ]),
    ("h1_hac_vif_appendix", [
        OUTPUT_REPORTS / "h1_hac_results.txt",
    ]),
    ("h1_event_study", [
        OUTPUT_REPORTS / "h1_event_study_results.txt",
    ]),
    ("h1b_ecb_feb2023_impact", [
        OUTPUT_REPORTS / "h1b_regression_results.txt",
    ]),
    ("h2_euro_adoption_impact", [
        OUTPUT_REPORTS / "h2_regression_results.txt",
        OUTPUT_REPORTS / "h2_placebo_results.txt",
    ]),
    ("h2_hac_vif_appendix", [
        OUTPUT_REPORTS / "h2_hac_results.txt",
    ]),
    ("h2_event_study", [
        OUTPUT_REPORTS / "h2_event_study_results.txt",
    ]),
    ("f_test_comparison", [
        OUTPUT_REPORTS / "f_test_comparison_results.txt",
    ]),
    ("paper_figures", [
        OUTPUT_FIGURES / "figure_3_1_all_countries_yields.png",
        OUTPUT_FIGURES / "figure_3_3_spreads_vs_germany.png",
        OUTPUT_FIGURES / "figure_3_4_croatia_euro_timeline.png",
        OUTPUT_FIGURES / "figure_4_1_h1_did_visual.png",
        OUTPUT_FIGURES / "figure_4_5_h1b_did_visual.png",
        OUTPUT_FIGURES / "figure_4_7_h2_spread_convergence.png",
    ]),
]


def main() -> None:
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

        missing = [path for path in expected_files if not path.exists()]
        if missing:
            for path in missing:
                print(f"  [MISSING] {path.name}")
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
