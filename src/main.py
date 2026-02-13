from __future__ import annotations

import argparse
from pathlib import Path

from edge_al_pipeline.config import load_experiment_config
from edge_al_pipeline.experiments.bootstrap import initialize_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bootstrap, curriculum phase experiments, and gate evaluation flows."
        )
    )
    parser.add_argument(
        "--mode",
        choices=[
            "bootstrap",
            "phase1",
            "phase1b",
            "phase2",
            "phase3_setup",
            "phase3",
            "phase3_compare",
            "gate_b",
            "gate_c",
            "gate_a",
        ],
        default="bootstrap",
        help="Execution mode. Use 'bootstrap' to scaffold artifacts only.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cdgp_week1_toy_cifar10_domain_guided.json"),
        help="Path to a JSON experiment config.",
    )
    parser.add_argument(
        "--pretrain-config",
        type=Path,
        default=Path("configs/legacy/phase2_plantvillage.json"),
        help="Pretrain config path used by Gate B transfer comparison.",
    )
    parser.add_argument(
        "--budget-ratio",
        type=float,
        default=0.10,
        help="Target labeled budget ratio used by Gate A.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="Minimum required entropy-minus-random improvement for Gate A.",
    )
    parser.add_argument(
        "--gate-c-min-improvement",
        type=float,
        default=0.0,
        help="Minimum required uncertainty-minus-random map50 improvement for Gate C.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.mode == "bootstrap":
        result = initialize_run(config, config_source=args.config)
        print(f"Initialized run directory: {result.run_dir}")
        print(
            "Split sizes: "
            f"L={result.labeled_count}, "
            f"U={result.unlabeled_count}, "
            f"V={result.val_count}, "
            f"T={result.test_count}"
        )
        print("Created artifacts: splits.json, metrics.csv, profile.csv, checkpoints/")
        return 0

    if args.mode == "phase1":
        from edge_al_pipeline.experiments.phase1_fashion_mnist import (
            run_phase1_fashion_mnist,
        )

        summary = run_phase1_fashion_mnist(config, config_source=args.config)
        print(f"Completed Phase 1 run for {len(summary.results)} seed(s).")
        for result in summary.results:
            print(
                f"Seed {result.seed}: run_dir={result.run_dir}, "
                f"L={result.final_labeled_count}, U={result.final_unlabeled_count}"
            )
        return 0

    if args.mode == "phase1b":
        from edge_al_pipeline.experiments.phase1b_cifar10 import run_phase1b_cifar10

        summary = run_phase1b_cifar10(config, config_source=args.config)
        print(f"Completed Phase 1b run for {len(summary.results)} seed(s).")
        for result in summary.results:
            print(
                f"Seed {result.seed}: run_dir={result.run_dir}, "
                f"L={result.final_labeled_count}, U={result.final_unlabeled_count}"
            )
        return 0

    if args.mode == "phase2":
        from edge_al_pipeline.experiments.phase2_agri_classification import (
            run_phase2_agri_classification,
        )

        summary = run_phase2_agri_classification(config, config_source=args.config)
        print(f"Completed Phase 2 run for {len(summary.results)} seed(s).")
        for result in summary.results:
            print(
                f"Seed {result.seed}: run_dir={result.run_dir}, "
                f"L={result.final_labeled_count}, U={result.final_unlabeled_count}, "
                f"backbone={result.backbone_path}"
            )
        return 0

    if args.mode == "phase3_setup":
        from edge_al_pipeline.experiments.phase3_wgisd_setup import (
            setup_phase3_wgisd,
        )

        summary = setup_phase3_wgisd(config, config_source=args.config)
        print(f"Initialized Phase 3 setup for {len(summary.results)} seed(s).")
        for result in summary.results:
            print(
                f"Seed {result.seed}: run_dir={result.run_dir}, "
                f"setup_manifest={result.setup_manifest_path}"
            )
        return 0

    if args.mode == "phase3":
        from edge_al_pipeline.experiments.phase3_wgisd_detection import (
            run_phase3_wgisd_detection,
        )

        summary = run_phase3_wgisd_detection(config, config_source=args.config)
        print(f"Completed Phase 3 run for {len(summary.results)} seed(s).")
        for result in summary.results:
            print(
                f"Seed {result.seed}: run_dir={result.run_dir}, "
                f"L={result.final_labeled_count}, U={result.final_unlabeled_count}, "
                f"uncertainty_summary={result.uncertainty_summary_path}"
            )
        return 0

    if args.mode == "phase3_compare":
        from edge_al_pipeline.experiments.phase3_uncertainty_comparison import (
            run_phase3_uncertainty_comparison,
        )

        report = run_phase3_uncertainty_comparison(config, config_source=args.config)
        print("Completed Phase 3 uncertainty comparison.")
        print(f"Report CSV: {report.report_csv_path}")
        print(f"Report JSON: {report.report_json_path}")
        print(f"Report Markdown: {report.report_markdown_path}")
        if report.paired_rows:
            mean_delta = sum(
                float(row["delta_map50_proxy_test"]) for row in report.paired_rows
            ) / len(report.paired_rows)
            print(f"Mean delta map50_proxy_test (loc-cls): {mean_delta:.6f}")
        return 0

    if args.mode == "gate_b":
        from edge_al_pipeline.evaluation.gate_b import run_gate_b_transfer

        pretrain_config = load_experiment_config(args.pretrain_config)
        report = run_gate_b_transfer(
            pretrain_config=pretrain_config,
            transfer_config=config,
            pretrain_config_source=args.pretrain_config,
            transfer_config_source=args.config,
        )
        print(f"Gate B status: {'PASS' if report.passed else 'FAIL'}")
        print(f"Reason: {report.reason}")
        print(
            "Mean delta final test accuracy (agri-imagenet): "
            f"{report.mean_delta_final_test_accuracy:.6f}"
        )
        print(
            "Mean delta AUC test accuracy (agri-imagenet): "
            f"{report.mean_delta_auc_test_accuracy:.6f}"
        )
        print(f"Report CSV: {report.report_csv_path}")
        print(f"Report JSON: {report.report_json_path}")
        print(f"Report Markdown: {report.report_markdown_path}")
        return 0

    if args.mode == "gate_c":
        from edge_al_pipeline.evaluation.gate_c import run_gate_c_field_validation

        report = run_gate_c_field_validation(
            config=config,
            config_source=args.config,
            minimum_required_improvement=args.gate_c_min_improvement,
        )
        print(f"Gate C status: {'PASS' if report.passed else 'FAIL'}")
        print(f"Reason: {report.reason}")
        print(
            "Mean delta final map50_proxy_test (uncertainty-random): "
            f"{report.mean_delta_final_map50_proxy_test:.6f}"
        )
        print(
            "Mean delta AUC map50_proxy_test (uncertainty-random): "
            f"{report.mean_delta_auc_map50_proxy_test:.6f}"
        )
        print(
            "Mean delta map50_proxy_test (localization-classification): "
            f"{report.mean_delta_localization_vs_classification_map50_proxy_test:.6f}"
        )
        print(
            "Mean delta map50_proxy_test (int8-fp32): "
            f"{report.mean_delta_int8_minus_fp32_map50_proxy_test:.6f}"
        )
        print(
            "Mean delta uncertainty_combined (int8-fp32): "
            f"{report.mean_delta_int8_minus_fp32_uncertainty_combined:.6f}"
        )
        print(f"Report CSV: {report.report_csv_path}")
        print(f"Report JSON: {report.report_json_path}")
        print(f"Report Markdown: {report.report_markdown_path}")
        return 0

    from edge_al_pipeline.evaluation.gate_a import run_gate_a

    report = run_gate_a(
        config,
        config_source=args.config,
        budget_ratio=args.budget_ratio,
        minimum_required_improvement=args.min_improvement,
    )
    print(f"Gate A status: {'PASS' if report.passed else 'FAIL'}")
    print(f"Reason: {report.reason}")
    print(
        "Paired improvement (entropy-random): "
        f"{report.paired_improvement.mean_improvement:.6f} "
        f"(95% CI: {report.paired_improvement.ci95_low:.6f}, "
        f"{report.paired_improvement.ci95_high:.6f})"
    )
    print(f"Report JSON: {report.report_json_path}")
    print(f"Report Markdown: {report.report_markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
