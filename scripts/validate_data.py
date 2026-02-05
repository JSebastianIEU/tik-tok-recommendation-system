import sys
from pathlib import Path

from src.common.validation import validate_file
from src.common.constants import MOCK_DATA_PATH

def main() -> int:
    jsonl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MOCK_DATA_PATH
    count, errors = validate_file(jsonl_path)
    print(f"Validated {count} records from {jsonl_path}")
    if errors:
        print("Errors:")
        for err in errors:
            print(f"- {err}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# TODO: add optional schema auto-fix suggestions and output as JSON.
