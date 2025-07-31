import sys
import json
import lzma
import argparse
from typing import TypedDict
from collections import Counter, defaultdict


class Canonical(TypedDict):
    sample: int
    assignment: list[int]


def process_plan(
    plan: Canonical,
    districts: Counter[frozenset[int] | int],
    plans: Counter[frozenset[frozenset[int] | int]],
    use_hash: bool,
) -> None:
    ds: dict[int, set[int]] = defaultdict(set)
    for i, district in enumerate(plan["assignment"]):
        ds[district].add(i)
    all_districts: set[frozenset[int] | int] = set()
    for district, nodes in ds.items():
        fs = frozenset(nodes)
        if use_hash:
            x = hash(fs)
        else:
            x = fs
        districts[x] += 1
        all_districts.add(x)
    whole: frozenset[frozenset[int] | int] = frozenset(all_districts)
    plans[whole] += 1


def main():
    args = parse_args()
    seen_districts = Counter[frozenset[int] | int]()
    seen_plans = Counter[frozenset[frozenset[int] | int]]()
    ensemble_file = lzma.open(args.file, "rt")
    metadata = json.loads(ensemble_file.readline())
    num_precincts: int = 0
    for line in sys.stdin:
        data: Canonical = json.loads(line)
        num_precincts = len(data["assignment"])
        process_plan(data, seen_districts, seen_plans, args.hash)

    plan_frequencies: Counter[int] = Counter()
    for count in seen_plans.values():
        plan_frequencies[count] += 1
    district_frequencies: Counter[int] = Counter()
    for count in seen_districts.values():
        district_frequencies[count] += 1

    d = {
        "file": args.file,
        "metadata": metadata,
        "num_precincts": num_precincts,
        "plan_frequencies": dict(plan_frequencies),
        "district_frequencies": dict(district_frequencies),
        "districts": {"total": seen_districts.total(), "unique": len(seen_districts)},
        "plans": {"total": seen_plans.total(), "unique": len(seen_plans)},
    }
    print(json.dumps(d))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count unique districting plans and districts."
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Use hash-based comparison instead of set-based.",
    )
    parser.add_argument("--file", type=str, help="File of the ensemble.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
