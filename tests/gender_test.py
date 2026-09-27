"""
Test script to verify gender normalization bug is fixed.

The bug was:
  F        -> Female   ✓
  Female   -> all      ✗ (should be Female)
  female   -> all      ✗ (should be Female)
  M        -> Male     ✓
  Male     -> all      ✗ (should be Male)
  male     -> all      ✗ (should be Male)

Root cause:
  The _normalize_gender() method was checking lowercase versions
  against the original gender strings instead of normalized ones.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'data'))
from range_checker import RangeChecker


# Test the fix
if __name__ == "__main__":
    checker = RangeChecker()

    print("="*80)
    print("TESTING GENDER NORMALIZATION FIX")
    print("="*80)

    print("\n1. Gender Normalization Test:")
    print("-" * 80)
    
    test_genders = ["F", "Female", "female", "M", "Male", "male", "all"]
    expected = {
        "F": "Female",
        "Female": "Female",
        "female": "Female",
        "M": "Male",
        "Male": "Male",
        "male": "Male",
        "all": "all"
    }
    
    all_pass = True
    for gender in test_genders:
        result = checker._normalize_gender(gender)
        exp = expected[gender]
        passed = result == exp
        all_pass = all_pass and passed
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {gender:8} -> {result:8} (expected: {exp:8})")

    print("\n" + "="*80)
    if all_pass:
        print("[OK] GENDER NORMALIZATION FIX SUCCESSFUL")
    else:
        print("[FAIL] GENDER NORMALIZATION STILL HAS ISSUES")
    print("="*80)

    print("\n2. Range Evaluation Test (with fixed gender handling):")
    print("-" * 80)
    
    range_tests = [
        ("Glucose", 168.0, "all", "High"),
        ("Hemoglobin", 11.1, "F", "Low"),
        ("Hemoglobin", 14.5, "M", "Normal"),
        ("Hemoglobin", 14.5, "Female", "Normal"),  # Female range: 12.0-15.5
        ("Hemoglobin", 17.0, "Female", "High"),     # Female range: 12.0-15.5
        ("Hemoglobin", 17.0, "Male", "Normal"),     # Male range: 13.8-17.2
        ("Hemoglobin", 17.0, "female", "High"),     # Lowercase female
        ("Hemoglobin", 17.0, "male", "Normal"),     # Lowercase male
    ]

    range_pass = True
    for param, value, gender, expected in range_tests:
        try:
            result = checker.evaluate(param, value, gender=gender)
            passed = result == expected
            range_pass = range_pass and passed
            status = "[OK]" if passed else "[FAIL]"
            print(
                f"{status} {param:12} = {value:5.1f} "
                f"({gender:8}) -> {result:8} "
                f"(expected: {expected:8})"
            )
        except Exception as e:
            print(f"[FAIL] {param:12} = {value:5.1f} ({gender:8}) -> ERROR: {e}")
            range_pass = False

    print("\n" + "="*80)
    if range_pass:
        print("[OK] RANGE EVALUATION WITH FIXED GENDER HANDLING WORKS")
    else:
        print("[FAIL] RANGE EVALUATION STILL HAS ISSUES")
    print("="*80)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if all_pass and range_pass:
        print("[OK] ALL TESTS PASSED - FIX IS WORKING!")
        print("\nYou can now:")
        print("  1. Replace range_checker.py with range_checker_FIXED.py")
        print("  2. Replace dataset_builder.py with dataset_builder_FIXED.py")
        print("  3. Run: python dataset_builder.py")
        print("  4. Verify dataset quality improved")
    else:
        print("[FAIL] SOME TESTS FAILED - CHECK REFERENCE RANGES CSV")
    print("="*80)