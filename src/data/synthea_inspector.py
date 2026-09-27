import os
import sys
import pandas as pd


def run_inspection():
    obs_path = "data/raw/observations.csv"
    pat_path = "data/raw/patients.csv"

    if not os.path.exists(obs_path) or not os.path.exists(pat_path):
        print("Error: observations.csv or patients.csv missing in data/raw/")
        sys.exit(1)

    print("Loading Synthea observations...")
    df_obs = pd.read_csv(obs_path)
    df_pat = pd.read_csv(pat_path)

    total_patients = df_pat['Id'].nunique()
    print(f"Total synthetic patients: {total_patients:,}")
    print(f"Total observation rows: {len(df_obs):,}\n")

    summary = (
        df_obs.groupby(['DESCRIPTION', 'CODE', 'UNITS'])
        .agg(
            patient_count=('PATIENT', 'nunique'),
            record_count=('VALUE', 'count')
        )
        .reset_index()
    )
    summary['patient_coverage_%'] = (summary['patient_count'] / total_patients * 100).round(2)

    from dataset_builder import CODE_TO_PARAM
    relevant = summary[summary['CODE'].isin(CODE_TO_PARAM.keys())].copy()
    relevant = relevant.sort_values(by='patient_count', ascending=False)

    relevant['status'] = relevant['patient_coverage_%'].apply(
        lambda x: 'Well-Covered' if x >= 30.0 else 'Sparse'
    )

    os.makedirs("data/processed", exist_ok=True)
    report_file = "data/processed/parameter_scope_report.csv"
    relevant.to_csv(report_file, index=False)

    print("--- Parameter Scope & Coverage Analysis ---")
    cols = ['DESCRIPTION', 'CODE', 'patient_count', 'patient_coverage_%', 'status']
    print(relevant[cols].to_string(index=False))
    print(f"\nReport generated at: {report_file}")

if __name__ == "__main__":
    run_inspection()