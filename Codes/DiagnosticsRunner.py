import pytest
import sys
import os
from pathlib import Path


def run_tests(verbose: bool = False) -> int:
    """
    Runs all tests with proper coverage reporting.
    """
    project_root = Path(__file__).parent.parent

    # 🔥 ВАЖНО: перейти в корень проекта
    os.chdir(project_root)

    test_path = project_root / "Codes" / "Testers"

    args = [
        str(test_path),
        "--cov=Codes/Databases",
        "--cov-report=term-missing",
        "--tb=short",
        "--disable-warnings",
    ]

    if verbose:
        args.append("-v")

    print("=== Running diagnostics ===\n")

    result = pytest.main(args)

    print("\n===========================")
    if result == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ TESTS FAILED (exit code {result})")

    return result


if __name__ == "__main__":
    sys.exit(run_tests())