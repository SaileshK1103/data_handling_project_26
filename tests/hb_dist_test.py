import json
import re
from collections import Counter

hba1c_stats = {
    "impossible (< 3.5%)": 0,
    "normal (4.0 - 5.6%)": 0,
    "abnormal/high (> 5.6%)": 0,
    "other_ranges": 0,
    "total_found": 0
}

for name in ["train", "val", "test"]:
    file_path = f"data/processed/{name}.jsonl"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                input_text = record["input"]
                
                match = re.search(r"HbA1c:\s*([\d\.]+)", input_text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
                    hba1c_stats["total_found"] += 1
                    
                    if val < 3.5:
                        hba1c_stats["impossible (< 3.5%)"] += 1
                    elif 4.0 <= val <= 5.6:
                        hba1c_stats["normal (4.0 - 5.6%)"] += 1
                    elif val > 5.6:
                        hba1c_stats["abnormal/high (> 5.6%)"] += 1
                    else:
                        hba1c_stats["other_ranges"] += 1 # Others (e.g. 3.5 to 4.0)
                        
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, skipping.")

print("\n--- Real HbA1c Distribution Report ---")
for category, count in hba1c_stats.items():
    print(f"  {category}: {count}")