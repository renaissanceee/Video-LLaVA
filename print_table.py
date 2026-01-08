import json
import re
import argparse
import os

def calculate_metrics(file_path):
    method = file_path.split('/')[-1].split('_results')[0]
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = 0
    yes_count = 0
    sum_score = 0

    for key, value in data.items():
        evaluation = value[0]
        pred_label = evaluation.get("pred", "no").lower()
        score = evaluation.get("score", 0)

        if pred_label == "yes":
            yes_count += 1

        sum_score += score
        total_samples += 1

    accuracy = (yes_count / total_samples) * 100 if total_samples > 0 else 0
    avg_score = sum_score / total_samples if total_samples > 0 else 0
    print(f"{method} & {accuracy:.2f} & {avg_score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default="predicts_prune/tgif") # "predicts_prune/tgif/sparsevlm_results.json"

    args = parser.parse_args()
    print(f"   Method   &  Acc  &  Score")
    if os.path.isdir(args.pred):
        json_files = [
            os.path.join(args.pred, f)
            for f in os.listdir(args.pred)
            if f.endswith("_results.json")
        ]
    elif args.pred.endswith(".json"):
        json_files = [args.pred]
    else:
        raise ValueError("pred_path must .json or a dir")
    for json_file in json_files:
        calculate_metrics(json_file)
