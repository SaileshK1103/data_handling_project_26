import os
import json
import random
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from range_checker import RangeChecker
from step7_hybrid_checker import HybridRangeChecker


CODE_TO_PARAM = {
    "2339-0": "Glucose",
    "2345-7": "Glucose",
    "6299-2": "Urea Nitrogen",
    "3094-0": "Urea Nitrogen",
    "2947-0": "Sodium",
    "2951-2": "Sodium",
    "6298-4": "Potassium",
    "2823-3": "Potassium",
    "2069-3": "Chloride",
    "2075-0": "Chloride",
    "49765-1": "Calcium",
    "17861-6": "Calcium",
    "20565-8": "Carbon Dioxide",
    "2028-9": "Carbon Dioxide",
    "718-7": "Hemoglobin",
    "4544-3": "Hematocrit",
    "6690-2": "Leukocytes",
    "777-3": "Platelets",
    "789-8": "Erythrocytes",
    "787-2": "MCV",
    "4548-4": "HbA1c",
    "1742-6": "ALT",
    "2276-4": "Ferritin",
    "3016-3": "TSH",
}


def build_datasets():

    obs_path = "data/raw/observations.csv"
    pat_path = "data/raw/patients.csv"

    if not os.path.exists(obs_path):
        print(f"ERROR: {obs_path} not found")
        return
    if not os.path.exists(pat_path):
        print(f"ERROR: {pat_path} not found")
        return

    checker = HybridRangeChecker() 

    print("Loading data...")
    df_obs = pd.read_csv(obs_path)
    df_pat = pd.read_csv(pat_path)

    print(f"  Observations: {len(df_obs):,} records")
    print(f"  Patients: {len(df_pat):,} records")

    df_obs = df_obs[df_obs['CODE'].isin(CODE_TO_PARAM.keys())].copy()
    df_obs['parameter'] = df_obs['CODE'].map(CODE_TO_PARAM)
    df_obs['VALUE_NUM'] = pd.to_numeric(df_obs['VALUE'], errors='coerce')
    df_obs = df_obs.dropna(subset=['VALUE_NUM', 'ENCOUNTER'])

    print(f"  Filtered observations: {len(df_obs):,} records")

    pat_map = df_pat.set_index('Id')[['GENDER', 'BIRTHDATE']].to_dict('index')
    print(f"  Patient map created: {len(pat_map):,} entries")

    encounters = df_obs.groupby('ENCOUNTER')
    print(f"  Encounters: {len(encounters):,}")

    samples = []
    skipped_impossible_hba1c = 0
    skipped_empty = 0
    skipped_misaligned = 0 

    for enc_id, group in encounters:
        patient_id = group['PATIENT'].iloc[0]
        pat_info = pat_map.get(
            patient_id,
            {'GENDER': 'all', 'BIRTHDATE': '1980-01-01'}
        )
        gender = pat_info.get('GENDER', 'all')

        value_counts_per_param = group.groupby('parameter')['VALUE_NUM'].nunique()
        if (value_counts_per_param > 1).any():
            skipped_misaligned += 1
            continue

        group = group.drop_duplicates(subset=['parameter', 'VALUE_NUM'])

        abnormal_findings = []
        lab_panel = {}

        for _, row in group.iterrows():
            param = row['parameter']
            val = float(row['VALUE_NUM'])
            unit = row['UNITS']

            if param == "HbA1c" and val < 3.5:
                skipped_impossible_hba1c += 1
                continue

            lab_panel[param] = f"{val} {unit}"

            status = checker.evaluate(param, val, gender=gender)
            if status != "Normal":
                abnormal_findings.append({
                    "parameter": param,
                    "value": val,
                    "unit": unit,
                    "flag": status
                })

        if not lab_panel:
            skipped_empty += 1
            continue

        input_text = (
            f"Patient Demographics: Gender={gender}. "
            f"Lab Results: "
        ) + ", ".join([f"{k}: {v}" for k, v in lab_panel.items()])

        target_output = {
            "abnormal_findings": abnormal_findings,
            "interpretation_summary": (
                f"Identified {len(abnormal_findings)} abnormal parameter(s)."
                if abnormal_findings
                else "All tested parameters within normal reference bounds."
            )
        }

        samples.append({
            "patient_id": patient_id, 
            "input": input_text,
            "output": json.dumps(target_output)
        })

    print(f"\nData filtering:")
    print(f"  Samples created: {len(samples):,}")
    print(f"  Skipped (misaligned encounter - multiple visits merged): {skipped_misaligned:,}")
    print(f"  Implausible HbA1c values dropped (value-level, not record-level): {skipped_impossible_hba1c:,}")
    print(f"  Skipped (empty panel): {skipped_empty:,}")

    # Group samples by patient
    from collections import defaultdict
    patient_samples = defaultdict(list)
    for s in samples:
        patient_samples[s['patient_id']].append(s)
    
    random.seed(42)
    patient_list = list(patient_samples.items())
    random.shuffle(patient_list)
    patient_list.sort(key=lambda x: len(x[1]), reverse=True)
    
    n_total = len(samples)
    target_train = n_total * 0.8
    target_val = n_total * 0.1
    target_test = n_total * 0.1

    splits = {"train": [], "val": [], "test": []}
    
    for pid, p_samples in patient_list:
        deficits = [
            ("train", target_train - len(splits["train"])),
            ("val", target_val - len(splits["val"])),
            ("test", target_test - len(splits["test"]))
        ]
        # Sort by largest deficit
        deficits.sort(key=lambda x: x[1], reverse=True)
        best_bucket = deficits[0][0]
        splits[best_bucket].extend(p_samples)

    os.makedirs("data/processed", exist_ok=True)
    print(f"\nSaving datasets to data/processed/")

    for name, data in splits.items():
        out_path = f"data/processed/{name}.jsonl"
        with open(out_path, "w") as f:
            for item in data:
                clean_item = {"input": item["input"], "output": item["output"]}
                f.write(json.dumps(clean_item) + "\n")
        print(f"  Saved {len(data):,} samples -> {out_path}")

    train_p = set(s['patient_id'] for s in splits['train'])
    val_p = set(s['patient_id'] for s in splits['val'])
    test_p = set(s['patient_id'] for s in splits['test'])
    assert not (train_p & val_p), "LEAKAGE: train/val share patients!"
    assert not (train_p & test_p), "LEAKAGE: train/test share patients!"
    assert not (val_p & test_p), "LEAKAGE: val/test share patients!"
    print("Patient-level split verified: no patient overlap.")

    n_total = len(samples)

    print(f"\n" + "=" * 80)
    print(f"SUMMARY")
    print(f"=" * 80)
    print(f"Total records: {n_total}")
    print(f"  Train: {len(splits['train'])} ({100*len(splits['train'])/n_total:.1f}%)")
    print(f"  Val:   {len(splits['val'])} ({100*len(splits['val'])/n_total:.1f}%)")
    print(f"  Test:  {len(splits['test'])} ({100*len(splits['test'])/n_total:.1f}%)")
    print(f"=" * 80)

    checker.print_audit_report()


if __name__ == "__main__":
    build_datasets()