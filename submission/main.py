import argparse
import os
import pandas as pd

from submission.predictor import ImmuneStatePredictor
from submission.utils import (
    validate_dirs_and_files,
    get_dataset_pairs,
    concatenate_output_files,
)

def run_once(train_dir, test_dirs_root, out_dir, n_jobs, device):
    # Validate high-level structure (recursive-friendly)
    validate_dirs_and_files(train_dir, [test_dirs_root], out_dir)

    # Pair train_dataset_i with all matching test_dataset_i*
    pairs = get_dataset_pairs(train_dir, test_dirs_root)
    if not pairs:
        raise RuntimeError(
            f"No (train, test) pairs found. "
            f"train_dir={train_dir}, test_dirs_root={test_dirs_root}"
        )

    # Train once per train dataset, rank sequences, then predict for each matched test dataset
    for train_path, test_paths in pairs:
        train_name = os.path.basename(train_path)
        print(f"\n=== Training on {train_name} ===")

        train_out = os.path.join(out_dir, train_name)
        os.makedirs(train_out, exist_ok=True)

        predictor = ImmuneStatePredictor(n_jobs=n_jobs, device=device)
        predictor.train(train_path, train_out)

        # Rank label-associated sequences for this train dataset
        seq_df = predictor.rank_sequences(train_path, train_out, topk=50000)
        # Write a competition-shaped file for sequences
        seq_save = os.path.join(out_dir, f"{train_name}_important_sequences.tsv")
        seq_submit = pd.DataFrame({
            "ID": ["-999.0"] * len(seq_df),
            "dataset": [train_name] * len(seq_df),
            "label_positive_probability": ["-999.0"] * len(seq_df),
            "junction_aa": seq_df["junction_aa"].values,
            "v_call": seq_df["v_call"].values,
            "j_call": seq_df["j_call"].values,
        })
        seq_submit.to_csv(seq_save, sep="\t", index=False)
        print(f"Wrote sequences: {seq_save}")

        # Predictions for each corresponding test dataset
        for test_path in test_paths:
            test_name = os.path.basename(test_path)
            print(f"--- Predicting on {test_name} (paired with {train_name}) ---")
            test_out = os.path.join(out_dir, test_name)
            os.makedirs(test_out, exist_ok=True)

            pred_df = predictor.predict(test_path, test_out)  # columns: repertoire_id, label_positive_probability
            pred_submit = pd.DataFrame({
                "ID": pred_df["repertoire_id"].values,
                "dataset": [test_name] * len(pred_df),  # <-- use subfolder name, not root
                "label_positive_probability": pred_df["label_positive_probability"].values,
                "junction_aa": ["-999.0"] * len(pred_df),
                "v_call": ["-999.0"] * len(pred_df),
                "j_call": ["-999.0"] * len(pred_df),
            })

            pred_save = os.path.join(out_dir, f"{test_name}_test_predictions.tsv")
            pred_submit.to_csv(pred_save, sep="\t", index=False)
            print(f"Wrote predictions: {pred_save}")

    # Build a single submissions.csv in out_dir
    concatenate_output_files(out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, type=str)
    parser.add_argument("--test_dir", type=str, help="Root of test datasets (singular).")
    parser.add_argument("--test_dirs", type=str, help="Root of test datasets (plural variant).")
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # support either --test_dir or --test_dirs flag variants
    test_dirs_root = args.test_dirs or args.test_dir
    if not test_dirs_root:
        raise ValueError("Provide either --test_dir or --test_dirs path to the root of test datasets.")

    run_once(
        train_dir=args.train_dir,
        test_dirs_root=test_dirs_root,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        device=args.device,
    )

if __name__ == "__main__":
    main()

