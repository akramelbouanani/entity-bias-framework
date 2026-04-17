"""
Verify that all experiment output files were generated correctly by checking
their existence and ensuring that the number of valid (non-NaN) rows exceeds
a specified threshold. This serves as a consistency check to confirm that
each experimental configuration completed successfully and produced
sufficiently large result files.
"""
import os
import pandas as pd
import itertools
import argparse
from utils import *

def verify_row_counts(tasks, few_shot_list, numerical_list, languages, models, default_threshold=800000):
    ok = []
    ko = []

    print("Starting row-count verification")

    for task in tasks:
        entities = get_entities(task)

        # Set specific threshold for 'countries', otherwise use default
        if entities == 'countries':
            current_threshold = 200000
        else:
            current_threshold = default_threshold

        for model, lang, few_shot, numerical_labels in itertools.product(models, languages, few_shot_list, numerical_list):
            if few_shot and numerical_labels:
                continue

            path = get_output_path(entities, task, lang, model, few_shot, numerical_labels)

            if not os.path.exists(path):
                ko.append((task, model, lang, few_shot, numerical_labels, "missing file"))
                continue

            try:
                df = pd.read_csv(path).dropna()
                row_count = len(df)

                if row_count > current_threshold:
                    ok.append((task, model, lang, few_shot, numerical_labels, row_count))
                else:
                    # Include the specific threshold used in the failure message
                    ko.append((task, model, lang, few_shot, numerical_labels, f"{row_count} (threshold: {current_threshold})"))
            except Exception as e:
                ko.append((task, model, lang, few_shot, numerical_labels, f"error reading file: {e}"))

    print("\n=== OK (rows > threshold) ===")
    for entry in ok:
        task, model, lang, fs, num, count = entry
        print(f"OK: {task}, {lang}, model {model}, few_shot={fs}, numerical={num} → {count} rows")

    print("\n=== KO (rows ≤ threshold or missing) ===")
    for entry in ko:
        task, model, lang, fs, num, reason = entry
        print(f"KO: {task}, {lang}, model {model}, few_shot={fs}, numerical={num} → {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--few_shot', nargs='+', required=True)
    parser.add_argument('--numerical', nargs='+', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--language', nargs='+', required=True)
    parser.add_argument('--threshold', type=int, default=800000)

    args = parser.parse_args()

    few_shot_list = [x.lower() == "true" for x in args.few_shot]
    numerical_list = [x.lower() == "true" for x in args.numerical]

    verify_row_counts(args.tasks, few_shot_list, numerical_list, args.language, args.models, args.threshold)