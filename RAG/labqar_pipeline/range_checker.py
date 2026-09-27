import pandas as pd
import os


class RangeChecker:

    def __init__(self, ref_file="config/reference_ranges.csv"):
        if not os.path.exists(ref_file):
            raise FileNotFoundError(
                f"Reference range config not found: {ref_file}"
            )

        self.df_ref = pd.read_csv(ref_file)

    def _normalize_gender(self, gender: str) -> str:
        """
        Normalize gender input to standard format: Male, Female, or all
        
        Handles:
        - Single letters: M, F, m, f
        - Full names: Male, Female, male, female
        - Default: all
        """
        if not gender:
            return "all"

        # First strip whitespace and convert to lowercase for comparison
        gender_clean = str(gender).strip().lower()

        # Check for M/F codes first
        if gender_clean == "m":
            return "Male"
        elif gender_clean == "f":
            return "Female"
        # Check for full names
        elif gender_clean == "male":
            return "Male"
        elif gender_clean == "female":
            return "Female"
        # Default to all if unrecognized
        else:
            return "all"

    def evaluate(
        self,
        parameter: str,
        value: float,
        gender: str = "all",
        age: int = 40
    ) -> str:
        """
        Evaluates a lab measurement and returns 'Low', 'Normal', or 'High'.
        
        Args:
            parameter: Lab test name (e.g., 'Glucose', 'Hemoglobin')
            value: Numeric lab value
            gender: Patient gender (M, F, Male, Female, or 'all')
            age: Patient age (for future age-group support)
        
        Returns:
            'Low', 'Normal', or 'High'
        """

        # Normalize parameter name - case insensitive, strip whitespace
        param_clean = parameter.strip().lower()

        # Find all matching parameters (case-insensitive)
        matches = self.df_ref[
            self.df_ref["parameter"].str.strip().str.lower() == param_clean
        ]

        if matches.empty:
            print(
                f"WARNING: Parameter '{parameter}' "
                f"not found in reference ranges"
            )
            return "Normal"

        # Normalize gender to standard format
        gender_normalized = self._normalize_gender(gender)

        # Get the appropriate row based on gender
        row = None

        if gender_normalized != "all":
            # Try to find gender-specific range
            gender_matches = matches[
                matches["gender"]
                .astype(str)
                .str.strip()
                .str.lower()
                == gender_normalized.lower()
            ]

            if not gender_matches.empty:
                row = gender_matches.iloc[0]
            else:
                # If no gender-specific match, fall back to 'all'
                all_matches = matches[
                    matches["gender"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    == "all"
                ]

                if not all_matches.empty:
                    row = all_matches.iloc[0]
                else:
                    # Last resort: use first available row
                    row = matches.iloc[0]
        else:
            # gender="all" → directly use 'all' range
            all_matches = matches[
                matches["gender"]
                .astype(str)
                .str.strip()
                .str.lower()
                == "all"
            ]

            if not all_matches.empty:
                row = all_matches.iloc[0]
            else:
                # Last resort: use first available row
                row = matches.iloc[0]

        # Extract and validate bounds
        low = float(row["lower_bound"])
        high = float(row["upper_bound"])

        if low > high:
            print(
                f"ERROR: Invalid range for {parameter}: "
                f"{low}-{high}"
            )
            return "Normal"

        # Evaluate against bounds
        if value < low:
            return "Low"
        elif value > high:
            return "High"
        else:
            return "Normal"


if __name__ == "__main__":

    checker = RangeChecker()

    print("="*80)
    print("TESTING GENDER NORMALIZATION")
    print("="*80)
    
    print("\nGender normalization test:")
    for gender in ["F", "Female", "female", "M", "Male", "male", "all"]:
        result = checker._normalize_gender(gender)
        status = '[OK]' if result in ['Male', 'Female', 'all'] else '[FAIL]'
        print(f"  {gender:8} -> {result:8} {status}")

    print("\n" + "="*80)
    print("TESTING RANGE EVALUATION")
    print("="*80)

    tests = [
        ("Glucose", 168.0, "all", "High"),
        ("Hemoglobin", 11.1, "F", "Low"),
        ("Hemoglobin", 14.5, "M", "Normal"),
        ("Erythrocytes", 4.5, "F", "Normal"),
        ("HbA1c", 6.0, "all", "High"),
        # Gender-specific edge cases
        ("Erythrocytes", 5.4, "Female", "High"),  # Female max is 5.0
        ("Erythrocytes", 5.4, "Male", "Normal"),  # Male range is 4.3-5.9
        ("Hemoglobin", 17.0, "Female", "High"),   # Female max is 15.5
        ("Hemoglobin", 17.0, "Male", "Normal"),   # Male range is 13.8-17.2
        ("Hemoglobin", 12.5, "M", "Low"),
        ("Hemoglobin", 12.5, "F", "Normal"),
    ]

    print("\nRange evaluation tests:")
    all_pass = True
    for param, value, gender, expected in tests:
        result = checker.evaluate(param, value, gender=gender)
        passed = result == expected
        all_pass = all_pass and passed
        status = "[OK]" if passed else "[FAIL]"
        print(
            f"  {status} {param:15} = {value:6.1f} "
            f"({gender:8}) -> {result:8} "
            f"[expected: {expected}]"
        )

    print("\n" + "="*80)
    if all_pass:
        print("[OK] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED - CHECK REFERENCE RANGES CSV")
    print("="*80)