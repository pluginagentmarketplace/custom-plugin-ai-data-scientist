#!/usr/bin/env python3
"""
Data Quality Check Framework
Comprehensive data validation for ETL pipelines
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class CheckStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: Optional[Dict[str, Any]] = None


class DataQualityChecker:
    """Data quality validation framework."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results: List[CheckResult] = []

    def null_check(self, columns: List[str], threshold: float = 0.0) -> CheckResult:
        """Check for null values in specified columns."""
        null_counts = {}
        failed = False

        for col in columns:
            if col not in self.df.columns:
                return CheckResult(
                    name="null_check",
                    status=CheckStatus.FAILED,
                    message=f"Column '{col}' not found",
                    details={"missing_column": col}
                )

            null_pct = self.df[col].isnull().mean()
            null_counts[col] = null_pct

            if null_pct > threshold:
                failed = True

        result = CheckResult(
            name="null_check",
            status=CheckStatus.FAILED if failed else CheckStatus.PASSED,
            message=f"Null check {'failed' if failed else 'passed'} for columns: {columns}",
            details={"null_percentages": null_counts, "threshold": threshold}
        )
        self.results.append(result)
        return result

    def uniqueness_check(self, columns: List[str]) -> CheckResult:
        """Check if columns contain unique values."""
        duplicates = self.df.duplicated(subset=columns, keep=False).sum()

        result = CheckResult(
            name="uniqueness_check",
            status=CheckStatus.FAILED if duplicates > 0 else CheckStatus.PASSED,
            message=f"Found {duplicates} duplicate rows" if duplicates > 0 else "All rows unique",
            details={
                "columns": columns,
                "duplicate_count": duplicates,
                "total_rows": len(self.df)
            }
        )
        self.results.append(result)
        return result

    def range_check(self, column: str, min_val: Optional[float] = None,
                   max_val: Optional[float] = None) -> CheckResult:
        """Check if values fall within expected range."""
        if column not in self.df.columns:
            return CheckResult(
                name="range_check",
                status=CheckStatus.FAILED,
                message=f"Column '{column}' not found"
            )

        violations = 0
        if min_val is not None:
            violations += (self.df[column] < min_val).sum()
        if max_val is not None:
            violations += (self.df[column] > max_val).sum()

        result = CheckResult(
            name="range_check",
            status=CheckStatus.FAILED if violations > 0 else CheckStatus.PASSED,
            message=f"{violations} values outside range [{min_val}, {max_val}]",
            details={
                "column": column,
                "min_expected": min_val,
                "max_expected": max_val,
                "actual_min": self.df[column].min(),
                "actual_max": self.df[column].max(),
                "violations": violations
            }
        )
        self.results.append(result)
        return result

    def schema_check(self, expected_schema: Dict[str, str]) -> CheckResult:
        """Validate DataFrame schema matches expected."""
        missing_cols = []
        type_mismatches = []

        for col, expected_type in expected_schema.items():
            if col not in self.df.columns:
                missing_cols.append(col)
            else:
                actual_type = str(self.df[col].dtype)
                if not self._type_compatible(actual_type, expected_type):
                    type_mismatches.append({
                        "column": col,
                        "expected": expected_type,
                        "actual": actual_type
                    })

        failed = len(missing_cols) > 0 or len(type_mismatches) > 0

        result = CheckResult(
            name="schema_check",
            status=CheckStatus.FAILED if failed else CheckStatus.PASSED,
            message=f"Schema {'invalid' if failed else 'valid'}",
            details={
                "missing_columns": missing_cols,
                "type_mismatches": type_mismatches,
                "extra_columns": [c for c in self.df.columns if c not in expected_schema]
            }
        )
        self.results.append(result)
        return result

    def freshness_check(self, date_column: str, max_age_hours: int = 24) -> CheckResult:
        """Check if data is fresh (within expected time range)."""
        if date_column not in self.df.columns:
            return CheckResult(
                name="freshness_check",
                status=CheckStatus.FAILED,
                message=f"Column '{date_column}' not found"
            )

        max_date = pd.to_datetime(self.df[date_column]).max()
        age_hours = (datetime.now() - max_date).total_seconds() / 3600

        result = CheckResult(
            name="freshness_check",
            status=CheckStatus.FAILED if age_hours > max_age_hours else CheckStatus.PASSED,
            message=f"Data is {age_hours:.1f} hours old (max allowed: {max_age_hours})",
            details={
                "latest_record": str(max_date),
                "age_hours": age_hours,
                "max_allowed_hours": max_age_hours
            }
        )
        self.results.append(result)
        return result

    def row_count_check(self, min_rows: int = 0, max_rows: Optional[int] = None) -> CheckResult:
        """Check if row count is within expected range."""
        count = len(self.df)
        failed = count < min_rows or (max_rows is not None and count > max_rows)

        result = CheckResult(
            name="row_count_check",
            status=CheckStatus.FAILED if failed else CheckStatus.PASSED,
            message=f"Row count: {count} (expected: {min_rows}-{max_rows or 'inf'})",
            details={
                "row_count": count,
                "min_expected": min_rows,
                "max_expected": max_rows
            }
        )
        self.results.append(result)
        return result

    def _type_compatible(self, actual: str, expected: str) -> bool:
        """Check if types are compatible."""
        type_groups = {
            'int': ['int64', 'int32', 'int16', 'int8', 'Int64', 'integer'],
            'float': ['float64', 'float32', 'float16', 'float'],
            'string': ['object', 'string', 'str'],
            'datetime': ['datetime64[ns]', 'datetime', 'timestamp'],
            'bool': ['bool', 'boolean']
        }

        for group, types in type_groups.items():
            if expected.lower() in [t.lower() for t in types]:
                return actual.lower() in [t.lower() for t in types]

        return actual.lower() == expected.lower()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all check results."""
        passed = sum(1 for r in self.results if r.status == CheckStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == CheckStatus.FAILED)
        warnings = sum(1 for r in self.results if r.status == CheckStatus.WARNING)

        return {
            "total_checks": len(self.results),
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "overall_status": "PASSED" if failed == 0 else "FAILED",
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message
                }
                for r in self.results
            ]
        }


def main():
    """Demo data quality checks."""
    # Create sample data
    df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'email': ['a@test.com', 'b@test.com', None, 'd@test.com', 'e@test.com'],
        'age': [25, 30, -5, 45, 150],
        'created_at': pd.date_range('2024-01-01', periods=5)
    })

    print("Data Quality Check Demo")
    print("=" * 50)

    checker = DataQualityChecker(df)

    # Run checks
    checker.null_check(['user_id', 'email'], threshold=0.0)
    checker.uniqueness_check(['user_id'])
    checker.range_check('age', min_val=0, max_val=120)
    checker.row_count_check(min_rows=1, max_rows=1000)
    checker.schema_check({
        'user_id': 'int',
        'email': 'string',
        'age': 'int',
        'created_at': 'datetime'
    })

    # Print summary
    summary = checker.get_summary()
    print(f"\nOverall Status: {summary['overall_status']}")
    print(f"Passed: {summary['passed']}/{summary['total_checks']}")

    for result in summary['results']:
        status_icon = "✓" if result['status'] == 'passed' else "✗"
        print(f"  {status_icon} {result['name']}: {result['message']}")


if __name__ == '__main__':
    main()
