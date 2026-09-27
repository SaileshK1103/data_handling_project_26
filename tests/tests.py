import json
from collections import Counter

print("="*80)
print("POST-FIX VALIDATION")
print("="*80)

total = 0
duplicates = 0
violations = 0
problematic_hba1c = 0
normal = 0
abnormal = 0

for name in ['train', 'val', 'test']:
    print(f"\n{name.upper()}:")
    dataset_dup = 0
    dataset_viol = 0
    dataset_total = 0
    
    with open(f'data/processed/{name}.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            output = json.loads(record['output'])
            findings = output['abnormal_findings']
            total += 1
            dataset_total += 1
            
            # Check duplicates
            keys = [f"{f['parameter']}:{f['flag']}" for f in findings]
            if len(keys) != len(set(keys)):
                duplicates += 1
                dataset_dup += 1
            
            # Check violations
            for finding in findings:
                if finding['parameter'] == 'HbA1c':
                    if finding['value'] < 3.5:
                        problematic_hba1c += 1
            
            # Check class balance
            if len(findings) == 0:
                normal += 1
            else:
                abnormal += 1
    
    print(f"  Records: {dataset_total}")
    print(f"  Duplicates: {dataset_dup}")
    if dataset_dup == 0:
        print(f"    [OK] No duplicate findings detected")

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"  Total records: {total}")
print(f"  Duplicate findings: {duplicates} ({'FIXED [OK]' if duplicates == 0 else f'{100*duplicates/total:.1f}% - FAILED [FAIL]'})")
print(f"  Problematic HbA1c: {problematic_hba1c} ({'FIXED [OK]' if problematic_hba1c == 0 else f'{100*problematic_hba1c/total:.1f}% - NEEDS FILTERING'})")
print(f"  Class balance: {100*normal/total:.1f}% normal, {100*abnormal/total:.1f}% abnormal")
print(f"    ({'GOOD [OK]' if 0.3 <= normal/total <= 0.5 else f'IMBALANCED - needs resampling'})")
