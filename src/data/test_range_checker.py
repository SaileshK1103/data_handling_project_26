from range_checker import RangeChecker

# Test the gender mapping fix
checker = RangeChecker()

test_cases = [
    ("Hemoglobin", 12.5, "F", "Normal"),      # Female normal: 12.0-15.5
    ("Hemoglobin", 12.5, "M", "Low"),      # Male: 12.5 is low (13.8-17.2)
    ("Hemoglobin", 14.5, "F", "Normal"),   # Female normal
    ("Hemoglobin", 14.5, "M", "Normal"),   # Male normal
    ("Erythrocytes", 4.5, "F", "Normal"),  # Female: 3.9-5.0
    ("Erythrocytes", 4.5, "M", "Normal"),  # Male: 4.3-5.9
]

print("Testing gender-specific range mapping:")
for param, value, gender, expected in test_cases:
    result = checker.evaluate(param, value, gender=gender)
    status = "[OK]" if result == expected else "[FAIL]"
    print(f"{status} {param}={value} (gender={gender}): got {result}, expected {expected}")
