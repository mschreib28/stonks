#!/usr/bin/env python3
from pathlib import Path

EXPECTED_FILE = Path("data/files_list.txt")          # <-- your S3 listing text file
LOCAL_ROOT    = Path("polygon_day_aggs_2025") # <-- your download folder

def parse_expected_paths(text: str) -> set[str]:
    expected = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # expected format:
        # 2025-05-07 20:30:14     212396 us_stocks_sip/day_aggs_v1/2025/01/2025-01-02.csv.gz
        parts = line.split()
        if len(parts) < 4:
            continue

        key = parts[-1]  # last token is the S3 key path
        # normalize to just the filename (date.csv.gz) to keep it simple
        expected.add(Path(key).name)
    return expected

def local_filenames(root: Path) -> set[str]:
    return {p.name for p in root.rglob("*.csv.gz")}

def main():
    expected_text = EXPECTED_FILE.read_text(encoding="utf-8", errors="replace")
    expected_names = parse_expected_paths(expected_text)
    local_names = local_filenames(LOCAL_ROOT)

    missing = sorted(expected_names - local_names)
    extra   = sorted(local_names - expected_names)

    print(f"Expected files: {len(expected_names)}")
    print(f"Local files:    {len(local_names)}")
    print(f"Missing files:  {len(missing)}")
    print(f"Extra files:    {len(extra)}\n")

    if missing:
        print("=== MISSING ===")
        for name in missing:
            print(name)

    if extra:
        print("\n=== EXTRA (local but not in expected list) ===")
        for name in extra:
            print(name)

if __name__ == "__main__":
    main()

