import json
import random


def balance_dataset(input_file, output_file, target_normal_ratio=0.40, max_repeat=5):
  
    normal_records = []
    abnormal_records = []

    with open(input_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            output = json.loads(record['output'])
            if len(output['abnormal_findings']) == 0:
                normal_records.append(record)
            else:
                abnormal_records.append(record)

    total_abnormal = len(abnormal_records)
    target_normal = int(total_abnormal * target_normal_ratio / (1 - target_normal_ratio))

    print(f"Abnormal records: {total_abnormal}")
    print(f"Normal records available (unique): {len(normal_records)}")
    print(f"Target normal records (ideal): {target_normal}")
    print(f"Max repeat per record: {max_repeat}")

    max_achievable = len(normal_records) * max_repeat

    if target_normal <= len(normal_records):
        # Enough unique normal records, no duplication needed
        balanced_normal = random.sample(normal_records, target_normal)
        print("No duplication needed, enough unique normal records.")
    elif target_normal <= max_achievable:
        # Reachable within the max_repeat cap
        repeats_needed = -(-target_normal // len(normal_records))  # ceil
        balanced_normal = (normal_records * repeats_needed)[:target_normal]
        print(f"Records repeated {repeats_needed}x to hit target (within cap).")
    else:
        # NOT reachable within the cap -- stop here instead of over-duplicating
        balanced_normal = normal_records * max_repeat
        achieved_ratio = len(balanced_normal) / (len(balanced_normal) + total_abnormal)
        print(f"WARNING: target ratio (%{target_normal_ratio*100:.0f}) not reachable with max_repeat={max_repeat}.")
        print(f"         Achieved ratio: %{achieved_ratio*100:.1f} (target: %{target_normal_ratio*100:.0f})")
        print(f"         Consider generating more normal patients in Synthea instead.")

    balanced = balanced_normal + abnormal_records
    random.shuffle(balanced)

    with open(output_file, 'w') as f:
        for record in balanced:
            f.write(json.dumps(record) + '\n')

    # How many rows are actual repeats (not unique)
    unique_normal_used = len(set(json.dumps(r) for r in balanced_normal))
    duplicate_count = len(balanced_normal) - unique_normal_used

    print(f"\nBalanced dataset saved: {len(balanced)} records")
    print(f"Ratio: {100*len(balanced_normal)/len(balanced):.1f}% normal")
    print(f"{duplicate_count} of those are repeated rows "
          f"(expected due to max_repeat={max_repeat} cap, not runaway duplication).")


if __name__ == "__main__":
    # Run this on train.jsonl ONLY -- val/test should reflect the real distribution, not an artificially balanced one.
    balance_dataset('data/processed/train.jsonl', 'data/processed/train_balanced.jsonl')