import os
import json
import random
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from range_checker import RangeChecker
# STEP 7: swap RangeChecker -> HybridRangeChecker to get a free LabQAR audit
# trail alongside every evaluation, with ZERO change to the resulting
# dataset -- reference_ranges.csv is still the sole source of the flags
# that get written out. See step7_hybrid_checker.py for why this is a
# secondary/audit layer rather than a replacement.
from step7_hybrid_checker import HybridRangeChecker


# Mapping Synthea descriptions to standard parameter keys
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
    """
    Build training datasets from Synthea observations.

    Process:
    1. Load raw Synthea data (observations.csv, patients.csv)
    2. Filter to target LOINC codes
    3. Group by encounter (single lab panel)
    4. Evaluate each parameter against reference ranges
    5. Create input/output pairs for fine-tuning
    6. Split into train/val/test (80/10/10), patient-grouped
    7. Save as JSONL files
    """

    obs_path = "data/raw/observations.csv"
    pat_path = "data/raw/patients.csv"

    if not os.path.exists(obs_path):
        print(f"ERROR: {obs_path} not found")
        return
    if not os.path.exists(pat_path):
        print(f"ERROR: {pat_path} not found")
        return

    checker = HybridRangeChecker()  # same .evaluate(param, val, gender=gender) call site as before

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
    skipped_misaligned = 0  # encounters that actually merge multiple real visits

    for enc_id, group in encounters:
        patient_id = group['PATIENT'].iloc[0]
        pat_info = pat_map.get(
            patient_id,
            {'GENDER': 'all', 'BIRTHDATE': '1980-01-01'}
        )
        gender = pat_info.get('GENDER', 'all')

        # If a parameter has multiple DIFFERENT values in this encounter,
        # this group is actually multiple real visits merged together.
        # Picking "first" or "last" value would still misalign input vs
        # output, so the only safe fix is to skip the whole encounter.
        value_counts_per_param = group.groupby('parameter')['VALUE_NUM'].nunique()
        if (value_counts_per_param > 1).any():
            skipped_misaligned += 1
            continue

        # Each parameter now has exactly one value, so duplicate LOINC
        # codes (e.g. "Glucose in Blood" + "in Serum/Plasma") are safe
        # to drop since they'll share the same value.
        group = group.drop_duplicates(subset=['parameter', 'VALUE_NUM'])

        abnormal_findings = []
        lab_panel = {}

        for _, row in group.iterrows():
            param = row['parameter']
            val = float(row['VALUE_NUM'])
            unit = row['UNITS']

            # HbA1c < 3.5% is physiologically impossible (known Synthea
            # diabetes-module glitch, confirmed via investigate_hba1c.py).
            # Skip only this VALUE, not the whole record.
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

        # Skip only if NOTHING is left after the HbA1c filter (rare).
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
            "patient_id": patient_id,  # needed for patient-level split below
            "input": input_text,
            "output": json.dumps(target_output)
        })

    print(f"\nData filtering:")
    print(f"  Samples created: {len(samples):,}")
    print(f"  Skipped (misaligned encounter - multiple visits merged): {skipped_misaligned:,}")
    print(f"  Implausible HbA1c values dropped (value-level, not record-level): {skipped_impossible_hba1c:,}")
    print(f"  Skipped (empty panel): {skipped_empty:,}")

    # --- Patient-level split, NOT row-level random.shuffle. ---
    # Group samples by patient
    from collections import defaultdict
    patient_samples = defaultdict(list)
    for s in samples:
        patient_samples[s['patient_id']].append(s)
    
    # Sort patients by number of encounters (descending) for better greedy packing
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
        # Calculate how far each bucket is from its target
        deficits = [
            ("train", target_train - len(splits["train"])),
            ("val", target_val - len(splits["val"])),
            ("test", target_test - len(splits["test"]))
        ]
        # Sort by largest deficit
        deficits.sort(key=lambda x: x[1], reverse=True)
        # Assign to the bucket with the largest deficit
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

    # Fail loudly if leakage ever creeps back in, instead of failing silently.
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

    # STEP 7: free correctness signal on reference_ranges.csv -- doesn't
    # change anything written to disk above, just reports where LabQAR
    # would have flagged something differently.
    checker.print_audit_report()


if __name__ == "__main__":
    build_datasets()