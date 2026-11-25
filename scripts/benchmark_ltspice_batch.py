"""Time the LTspice batch runner on the same 35 nominal-corner decks.

Use the project environment with psutil installed and ltspice_path configured.
Run from the project root and pass --output with the desired JSON filename.
"""

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import platform
import re
from statistics import mean, median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_dataset import HEALTHY_CORE_PATH, OPERATIONS, create_fault_cell, get_default_fault_specs
from src.simulation_runner import _load_config, run_batch, run_simulation
from src.testbench_composer import compose_testbench
from src.utils import load_cell_core


CASES = (
    (0, "none", 0.0),
    (1, "M5", 5000.0),
    (2, "Q_Qb", 5000.0),
    (3, "storage_pair", 15.0),
    (4, "access_pair", 15.0),
)
WORKER_ORDERS = ((1, 2, 4, 8), (8, 4, 2, 1), (4, 1, 8, 2))
ELAPSED = re.compile(r"Total elapsed time:\s*(\d+(?:\.\d+)?)\s+seconds", re.IGNORECASE)


def compose_decks(directory):
    healthy = load_cell_core(str(HEALTHY_CORE_PATH))
    specs = {
        (spec["fault_class"], spec["target"], spec["severity_value"]): spec
        for spec in get_default_fault_specs()
    }
    paths = []
    for fault_class, target, severity in CASES:
        case = f"f{fault_class}_{target}"
        core = directory / f"{case}.net"
        create_fault_cell(specs[fault_class, target, severity], healthy, str(core))
        for op in OPERATIONS:
            path = directory / f"{case}__{op}.net"
            compose_testbench(str(core), op, str(path), vdd=1.0, temperature_c=27.0)
            paths.append(str(path))
    return paths


def time_batch(decks, directory, workers):
    # Exclude terminal rendering from the timed batch.
    with redirect_stdout(io.StringIO()):
        start = perf_counter()
        results = run_batch(
            decks, str(directory / "raw"), str(directory / "logs"),
            str(directory / "flags.csv"), max_workers=workers,
        )
        elapsed = perf_counter() - start

    records = []
    for row in results.itertuples(index=False):
        log = Path(row.log_path)
        if not log.is_file() or not Path(row.raw_path).is_file():
            raise RuntimeError(f"Missing LTspice output for {Path(row.net_path).name}: {row.warning}")
        time_match = ELAPSED.search(log.read_text(encoding="utf-8", errors="replace"))
        if time_match is None:
            raise RuntimeError(f"Missing elapsed time in {log.name}")
        records.append({
            "deck": Path(row.net_path).name,
            "operation": Path(row.net_path).stem.split("__")[1],
            "simulator_seconds": float(time_match[1]),
            "converged": bool(row.converged),
            "warning": row.warning if isinstance(row.warning, str) else None,
        })
    flagged = sum(not record["converged"] for record in records)
    print(f"{workers} workers, {len(records)} jobs, {flagged} flagged, {elapsed:.3f} s", flush=True)
    return {
        "workers": workers,
        "wall_seconds": elapsed,
        "jobs_completed": len(records),
        "flagged_jobs": flagged,
        "jobs": records,
    }


def machine():
    processor = platform.processor()
    if os.name == "nt":
        import winreg

        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as cpu:
            processor = winreg.QueryValueEx(cpu, "ProcessorNameString")[0].strip()
    return {
        "processor": processor,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "usable_memory_bytes": psutil.virtual_memory().total,
        "os": f"{platform.system()} {platform.version()}",
    }


def summarize(batches):
    times = {}
    for batch in batches:
        times.setdefault(batch["workers"], []).append(batch["wall_seconds"])
    serial_median = median(times[1])
    return [
        {
            "workers": w,
            "median_batch_seconds": median(times[w]),
            "min_batch_seconds": min(times[w]),
            "max_batch_seconds": max(times[w]),
            "speedup": serial_median / median(times[w]),
            "efficiency": serial_median / (w * median(times[w])),
        }
        for w in sorted(times)
    ]


def summarize_serial_operations(batches):
    serial_jobs = [job for batch in batches if batch["workers"] == 1 for job in batch["jobs"]]
    job_means = {}
    for label, operations in (("transient", OPERATIONS[:5]), ("snm_dc", OPERATIONS[5:])):
        values = [job["simulator_seconds"] for job in serial_jobs if job["operation"] in operations]
        job_means[label] = {"jobs": len(values), "mean_simulator_seconds": mean(values)}
    return job_means


def save_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON file for raw and summarized results")
    args = parser.parse_args()
    executable = _load_config().get("ltspice_path")
    if not executable or not Path(executable).is_file():
        parser.error("Set ltspice_path in the project config.yaml to an existing executable.")

    report = {
        "machine": machine(),
        "corner": {"process": "TT", "vdd_v": 1.0, "temperature_c": 27},
        "cases": CASES,
        "operations": OPERATIONS,
        "complete": False,
        "batches": [],
    }
    with TemporaryDirectory(prefix="sram-ltspice-timing-") as temporary:
        work = Path(temporary)
        decks = compose_decks(work)
        warmup = run_simulation(decks[0], str(work / "warmup_raw"), str(work / "warmup_logs"))
        if not warmup["converged"] or not warmup["ltspice_version"]:
            raise RuntimeError(f"LTspice warm-up failed: {warmup['warning']}")
        report["ltspice_version"] = warmup["ltspice_version"]
        print(f"Warm-up complete using {warmup['ltspice_version']}.", flush=True)
        for trial, worker_order in enumerate(WORKER_ORDERS, start=1):
            for workers in worker_order:
                batch = time_batch(decks, work / f"trial_{trial}_workers_{workers}", workers)
                report["batches"].append({"trial": trial, **batch})
                save_report(args.output, report)

    report["summary"] = summarize(report["batches"])
    report["serial_operation_timings"] = summarize_serial_operations(report["batches"])
    report["complete"] = True
    save_report(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["serial_operation_timings"], indent=2))
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
