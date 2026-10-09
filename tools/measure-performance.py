#!/usr/bin/env python3
"""Run the opt-in Release XCTest benchmark and export comparable local reports."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST = "QueryableTests/PerformanceMeasurementTests/testModelPerformance"


def command(args, *, log=None, environment=None, check=True):
    if log is None:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    with log.open("w") as stream:
        result = subprocess.run(args, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if check and result.returncode:
        lines = log.read_text(errors="replace").splitlines()
        diagnostics = [line for line in lines if "error:" in line]
        tail = "\n".join(line[:300] for line in (diagnostics[:15] + lines[-10:]))
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}\n{tail}")
    return result.returncode


def provenance():
    git = ["git", "-c", "core.fsmonitor=false"]
    names = subprocess.check_output(git + ["ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(set(names) - {b""}):
        path = ROOT / os.fsdecode(name)
        if path.suffix not in {".swift", ".pbxproj", ".metal", ".plist", ".sh", ".py", ".xcscheme"} or not path.is_file():
            continue
        data = path.read_bytes()
        digest.update(str(len(name)).encode() + b":" + name + str(len(data)).encode() + b":" + data)
    return {
        "commit": command(git + ["rev-parse", "HEAD"]),
        "workingTreeDirty": bool(command(git + ["status", "--porcelain"])),
        "sourceSHA256": digest.hexdigest(),
        "xcode": command(["xcodebuild", "-version"]),
        "buildConfiguration": "Release",
    }


def required_metrics(config):
    names = {"image_model_load", "text_model_load", "image_first_batch", "text_first_query", "image_batch", "image_batch_1", "text_query"}
    if config["includeSearch"]:
        for size in config["indexSizes"]:
            names.update(f"{stage}_{size}" for stage in ("index_build", "search_first", "search", "text_search"))
    return names


def validate_report(report, config):
    if report.get("schemaVersion") != 1 or report.get("status") != "complete":
        raise ValueError(f"Incomplete benchmark: {report.get('failure', 'no completed report')}")
    if report.get("configuration") != config:
        raise ValueError("The test did not execute the requested workload")
    metrics = report.get("metrics", {})
    if set(metrics) != required_metrics(config):
        raise ValueError("The benchmark is missing required measurements")
    for name, metric in metrics.items():
        samples = metric.get("samplesMilliseconds", [])
        repeated = (name in {"image_batch", "image_batch_1", "text_query"}
                    or name.startswith("text_search_")
                    or name.startswith("search_") and not name.startswith("search_first_"))
        expected = config["samples"] if repeated else 1
        if len(samples) != expected or not all(isinstance(x, (float, int)) and math.isfinite(x) and x > 0 for x in samples):
            raise ValueError(f"Invalid/missing timing samples for {name}")
        if not math.isfinite(metric.get("workItemsPerSecond", math.nan)) or metric["workItemsPerSecond"] <= 0:
            raise ValueError(f"Invalid throughput for {name}")
    artifacts = report.get("artifacts", [])
    if len(artifacts) != 4 or any(len(item.get("sha256", "")) != 64 or item.get("bytes", 0) <= 0 for item in artifacts):
        raise ValueError("Missing model/tokenizer fingerprints")
    memory = report.get("memory", {})
    if memory.get("successfulSamples", 0) <= 0 or not memory.get("sampledPeakBytes"):
        raise ValueError("Process memory could not be measured")
    if not report.get("actualImageComputeUnits") or not report.get("actualTextComputeUnits"):
        raise ValueError("Effective compute settings are missing")


def comparison_key(report, host):
    keys = ("workloadVersion", "configuration", "modelID", "modelRevision", "modelContract", "embeddingDimension",
            "hardware", "os", "platform", "simulatorModel", "gpuName", "physicalMemoryBytes",
            "actualImageComputeUnits", "actualTextComputeUnits", "artifacts")
    return {**{key: report.get(key) for key in keys}, "xcode": host["xcode"],
            "buildConfiguration": host["buildConfiguration"], "buildSettings": host.get("buildSettings", []),
            "destination": host.get("destination"), "swiftOptimization": host.get("swiftOptimization")}


def aggregate(reports, host):
    config = reports[0]["configuration"]
    key = comparison_key(reports[0], host)
    for report in reports:
        validate_report(report, config)
        if comparison_key(report, host) != key:
            raise ValueError("Runs used different devices, models, or workloads")
    metrics = {}
    for name in sorted(reports[0]["metrics"]):
        runs = [report["metrics"][name] for report in reports]
        pooled = sorted(sample for metric in runs for sample in metric["samplesMilliseconds"])
        medians = [statistics.median(metric["samplesMilliseconds"]) for metric in runs]
        metrics[name] = {
            "medianMilliseconds": statistics.median(medians),
            "p95Milliseconds": pooled[math.ceil(len(pooled) * 0.95) - 1],
            "runMediansMilliseconds": medians,
            "workItemsPerSecond": statistics.median(metric["workItemsPerSecond"] for metric in runs),
        }
    peaks = [report["memory"]["sampledPeakBytes"] for report in reports]
    thermal = sorted({state for report in reports for state in report["memory"]["thermalStates"]})
    return {"schemaVersion": 1, "comparisonKey": key, "provenance": host,
            "runs": len(reports), "metrics": metrics, "thermalStates": thermal,
            "medianSampledPeakBytes": statistics.median(peaks)}


def compare(baseline, candidate, limit):
    if baseline.get("schemaVersion") != 1 or candidate.get("schemaVersion") != 1:
        raise ValueError("Unsupported report schema")
    if baseline.get("comparisonKey") != candidate.get("comparisonKey"):
        raise ValueError("Cannot compare different hardware, OS, Xcode, model files, compute settings, or workloads")
    if set(baseline["metrics"]) != set(candidate["metrics"]) or not baseline["metrics"]:
        raise ValueError("Baseline and candidate measurements differ or are empty")
    if any(state in {"serious", "critical"} for report in (baseline, candidate) for state in report["thermalStates"]):
        raise ValueError("A run experienced thermal pressure; repeat after the device cools")
    changes = {}
    for name, metric in candidate["metrics"].items():
        old = baseline["metrics"][name]["medianMilliseconds"]
        new = metric["medianMilliseconds"]
        if not all(math.isfinite(x) and x > 0 for x in (old, new)):
            raise ValueError(f"Invalid baseline/candidate timing: {name}")
        changes[name] = 100 * (new / old - 1)
    return {"maxRegressionPercent": limit, "changesPercent": changes,
            "regressions": {name: value for name, value in changes.items() if value > limit}}


def write_summary(path, result, comparison=None):
    key = result["comparisonKey"]
    lines = ["# Queryable performance measurements", "",
             f"Model: **{key['modelID']}** · Destination: **{key['platform']} / {key['hardware']}**",
             f"Runs: {result['runs']} · Thermal states: {', '.join(result['thermalStates'])}", "",
             "Scope: **" + ("models and GPU search" if key['configuration']['includeSearch'] else "models only; GPU search excluded") + "**.", "",
             "Median is the middle of the per-run medians. P95 is the pooled 95th percentile; 95% of samples finished within it.", "",
             "| Operation | Median (ms) | P95 (ms) | Work items/second | Run medians (ms) |",
             "|---|---:|---:|---:|---|"]
    for name, metric in result["metrics"].items():
        medians = ", ".join(f"{value:.2f}" for value in metric["runMediansMilliseconds"])
        lines.append(f"| {metric_label(name, key['configuration'])} | {metric['medianMilliseconds']:.2f} | {metric['p95Milliseconds']:.2f} | {metric['workItemsPerSecond']:.2f} | {medians} |")
    lines += ["", f"Median sampled peak process footprint: **{result['medianSampledPeakBytes'] / 1024**2:.1f} MiB**.", "",
              "Model loads and first-use operations have one sample per process. Disk/compiler caches are not cleared.",
              "Generated images/vectors measure component speed. Photo fetching, index disk I/O, UI rendering, search quality, and total accelerator memory are outside this measurement."]
    if key["platform"] == "simulator":
        lines += ["", "Simulator results validate the runner and describe the host Mac; use physical hardware for phone performance decisions."]
    if comparison:
        lines += ["", f"Regression threshold: {comparison['maxRegressionPercent']}% slower than baseline.", "",
                  "| Operation | Change in median time |", "|---|---:|"]
        for name, change in comparison["changesPercent"].items():
            lines.append(f"| {name} | {change:+.1f}% |")
        lines += ["", "Threshold exceeded: " + (", ".join(comparison["regressions"]) or "none") + "."]
    path.write_text("\n".join(lines) + "\n")


def metric_label(name, config):
    labels = {"image_model_load": "Load image model", "text_model_load": "Load text model and word files",
              "image_first_batch": f"First image batch ({config['batchSize']} images)",
              "text_first_query": "First text prediction", "image_batch": f"Repeated image batch ({config['batchSize']} images)",
              "image_batch_1": "Encode one image through batch API", "text_query": "Repeated text prediction"}
    if name in labels:
        return labels[name]
    stage, count = name.rsplit("_", 1)
    stages = {"index_build": "Prepare GPU index", "search_first": "First search and sorting",
              "search": "Repeated search and sorting", "text_search": "Text prediction + search and sorting"}
    return f"{stages.get(stage, stage)} ({int(count):,} entries)"


def load_aggregate(path):
    return json.loads((path / "aggregate.json" if path.is_dir() else path).read_text())


def finish(args, output, reports, host):
    result = aggregate(reports, host)
    (output / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
    write_summary(output / "summary.md", result)
    comparison = compare(load_aggregate(args.baseline), result, args.max_regression_percent) if args.baseline else None
    if comparison:
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    write_summary(output / "summary.md", result, comparison)
    print(f"Report: {output / 'summary.md'}", flush=True)
    return 1 if comparison and comparison["regressions"] else 0


def native_reports(args, output, config, host):
    package = output / "native-package"
    sources = package / "Sources/Queryable"
    tests = package / "Tests/NativePerformanceTests"
    sources.mkdir(parents=True)
    tests.mkdir(parents=True)
    app = ROOT / "Queryable/Queryable"
    shared = ["Model/Embedding.swift", "Model/GPUSimilaritySearch.swift", "CLIP/ImgEncoder.swift", "CLIP/TextEncoder.swift",
              "CLIP/Tokenizer/BPETokenizer.swift", "CLIP/Tokenizer/BPETokenizer+Reading.swift"]
    for name in shared:
        shutil.copy2(app / name, sources / Path(name).name)
    shutil.copy2(ROOT / "tools/performance-native/PlatformImage.swift", sources)
    shutil.copy2(ROOT / "tools/performance-native/Package.swift", package)
    shutil.copy2(ROOT / "Queryable/QueryableTests/PerformanceMeasurementTests.swift", tests)
    host.update(destination="native-mac", swiftOptimization="-O", swift=command(["swift", "--version"]))
    (output / "run-config.json").write_text(json.dumps({"configuration": config, "provenance": host}, indent=2) + "\n")
    common = ["--package-path", str(package), "--scratch-path", str(args.derived_data.resolve() / "native"),
              "--configuration", "release", "-Xswiftc", "-enable-testing"]
    environment = os.environ.copy()
    environment["QUERYABLE_BENCHMARK"] = "1"
    environment["QUERYABLE_BENCHMARK_CONFIG"] = json.dumps(config)
    environment["QUERYABLE_BENCHMARK_RESOURCES"] = str((args.resources or app / "CoreMLModels").resolve())
    print(f"Building native Mac Release benchmark; log: {output / 'build.log'}", flush=True)
    command(["swift", "build", "--build-tests", *common], log=output / "build.log", environment=environment)
    if provenance()["sourceSHA256"] != host["sourceSHA256"]:
        raise RuntimeError("Source changed during the build; repeat for trustworthy provenance")
    reports = []
    for number in range(1, args.runs + 1):
        path = output / f"run-{number}.json"
        environment["QUERYABLE_BENCHMARK_REPORT_PATH"] = str(path)
        print(f"Measuring native run {number}/{args.runs}; log: {output / f'run-{number}.log'}", flush=True)
        command(["swift", "test", "--skip-build", "--filter", "PerformanceMeasurementTests.testModelPerformance", *common],
                log=output / f"run-{number}.log", environment=environment)
        if not path.exists():
            raise RuntimeError("The native test produced no report")
        report = json.loads(path.read_text())
        validate_report(report, config)
        reports.append(report)
    return reports


def run(args):
    config = {"modelID": args.model, "computeUnits": args.compute_units, "samples": args.samples,
              "warmups": args.warmups, "batchSize": args.batch_size, "indexSizes": args.index_sizes,
              "topK": args.top_k, "includeSearch": not args.model_only}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    host = provenance()
    host["destination"] = args.destination
    host["buildSettings"] = args.build_setting
    if args.native_mac:
        return finish(args, output, native_reports(args, output, config, host), host)
    common = ["xcodebuild", "-project", str(ROOT / "Queryable/Queryable.xcodeproj"), "-scheme", "QueryablePerformance",
              "-configuration", "Release", "-destination", args.destination, "-derivedDataPath", str(args.derived_data.resolve()),
              "ENABLE_TESTABILITY=YES"]
    if "iOS Simulator" in args.destination:
        common.append("CODE_SIGNING_ALLOWED=NO")
    common += args.build_setting
    # The old test target has a different hard-coded team from the app. Resolve
    # the app's existing signing team and apply it to both, without editing Xcode.
    with (output / "build-settings.log").open("w") as stream:
        settings = json.loads(subprocess.check_output(common + ["-showBuildSettings", "-json"], cwd=ROOT, stderr=stream, text=True))
    app = next(item["buildSettings"] for item in settings if item["target"] == "Queryable")
    host["swiftOptimization"] = app.get("SWIFT_OPTIMIZATION_LEVEL")
    if host["swiftOptimization"] not in {"-O", "-Osize"}:
        raise ValueError("The app's Release configuration is not optimized")
    if not any(setting.startswith("DEVELOPMENT_TEAM=") for setting in args.build_setting) and app.get("DEVELOPMENT_TEAM"):
        inherited_team = f"DEVELOPMENT_TEAM={app['DEVELOPMENT_TEAM']}"
        common.append(inherited_team)
        host["buildSettings"] = [*args.build_setting, inherited_team]
    (output / "run-config.json").write_text(json.dumps({"configuration": config, "provenance": host}, indent=2) + "\n")
    common += ["-enableCodeCoverage", "NO", "-parallel-testing-enabled", "NO", "-enableAddressSanitizer", "NO",
               "-enableThreadSanitizer", "NO", "-enableUndefinedBehaviorSanitizer", "NO"]
    environment = os.environ.copy()
    environment["TEST_RUNNER_QUERYABLE_BENCHMARK"] = "1"
    environment["TEST_RUNNER_QUERYABLE_BENCHMARK_CONFIG"] = json.dumps(config)
    print(f"Building Release benchmark; log: {output / 'build.log'}", flush=True)
    command(common + ["build-for-testing"], log=output / "build.log", environment=environment)
    if provenance()["sourceSHA256"] != host["sourceSHA256"]:
        raise RuntimeError("Source changed during the build; repeat for trustworthy provenance")
    reports = []
    for number in range(1, args.runs + 1):
        result_bundle = output / f"run-{number}.xcresult"
        print(f"Measuring run {number}/{args.runs}; log: {output / f'run-{number}.log'}", flush=True)
        status = command(common + ["test-without-building", f"-only-testing:{TEST}", "-resultBundlePath", str(result_bundle), "-collect-test-diagnostics", "never"],
                         log=output / f"run-{number}.log", environment=environment, check=False)
        attachments = output / f"attachments-{number}"
        command(["xcrun", "xcresulttool", "export", "attachments", "--path", str(result_bundle), "--output-path", str(attachments)],
                log=output / f"export-{number}.log")
        candidates = []
        for path in attachments.rglob("*"):
            if not path.is_file() or path.name == "manifest.json":
                continue
            try:
                candidate = json.loads(path.read_text())
            except (ValueError, UnicodeError):
                continue
            if isinstance(candidate, dict) and "workloadVersion" in candidate and "metrics" in candidate:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one benchmark report, found {len(candidates)}; see {output / f'run-{number}.log'}")
        report = candidates[0]
        (output / f"run-{number}.json").write_text(json.dumps(report, indent=2) + "\n")
        validate_report(report, config)
        if status:
            raise RuntimeError(f"XCTest failed ({status}); see {output / f'run-{number}.log'}")
        reports.append(report)
    return finish(args, output, reports, host)


def bounded(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number
    return parse


def index_sizes(value):
    try:
        sizes = [bounded(1, 100_000)(part) for part in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("use comma-separated index sizes from 1 to 100000") from error
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("index sizes must be unique")
    return sorted(sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    measure = commands.add_parser("run", help="build, measure, and export reports")
    destination = measure.add_mutually_exclusive_group(required=True)
    destination.add_argument("--destination", help="Xcode destination, e.g. platform=iOS,id=PHONE_UDID")
    destination.add_argument("--native-mac", action="store_true", help="measure the actual components natively on Mac without a signed app host")
    measure.add_argument("--resources", type=Path, help="native Mac model directory; defaults to Queryable/Queryable/CoreMLModels")
    measure.add_argument("--output", type=Path, default=ROOT / "performance-results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    measure.add_argument("--derived-data", type=Path, default=ROOT / "build/PerformanceDerivedData")
    measure.add_argument("--model", choices=["mobileclip2-s4", "mobileclip-s2"], default="mobileclip2-s4")
    measure.add_argument("--compute-units", choices=["all", "cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine"], default="all")
    measure.add_argument("--runs", type=bounded(1, 20), default=3)
    measure.add_argument("--samples", type=bounded(3, 1000), default=20)
    measure.add_argument("--warmups", type=bounded(1, 100), default=3)
    measure.add_argument("--batch-size", type=bounded(1, 128), default=32)
    measure.add_argument("--index-sizes", type=index_sizes, default=[1000, 10000, 25000])
    measure.add_argument("--top-k", type=bounded(1, 1000), default=120)
    measure.add_argument("--model-only", action="store_true", help="measure models only; required on Simulator")
    measure.add_argument("--build-setting", action="append", default=[], metavar="KEY=VALUE", help="optional Xcode signing setting")
    measure.add_argument("--baseline", type=Path, help="previous aggregate.json or result directory")
    comparison = commands.add_parser("compare", help="compare two existing reports without rebuilding")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    for subparser in (measure, comparison):
        subparser.add_argument("--max-regression-percent", type=float, default=15.0, help="comparison threshold; default 15%%")
    args = parser.parse_args()
    if not math.isfinite(args.max_regression_percent) or args.max_regression_percent < 0:
        parser.error("regression percentage must be finite and nonnegative")
    if args.action == "run" and any("=" not in setting or setting.startswith("-") for setting in args.build_setting):
        parser.error("build settings must use KEY=VALUE")
    signing_keys = {"DEVELOPMENT_TEAM", "CODE_SIGN_IDENTITY", "PROVISIONING_PROFILE_SPECIFIER",
                    "PRODUCT_BUNDLE_IDENTIFIER", "CODE_SIGNING_ALLOWED", "CODE_SIGNING_REQUIRED"}
    if args.action == "run" and any(setting.split("=", 1)[0] not in signing_keys for setting in args.build_setting):
        parser.error("only signing settings are accepted; performance builds always use the project's Release optimization")
    if args.action == "run" and "iOS Simulator" in (args.destination or "") and not args.model_only:
        parser.error("Simulator requires --model-only; run GPU search measurements on your phone or Mac")
    if args.action == "run" and args.native_mac and args.build_setting:
        parser.error("native Mac runs do not use Xcode signing settings")
    if args.action == "run" and args.resources and not args.native_mac:
        parser.error("--resources applies only to native Mac runs")
    try:
        if args.action == "run":
            return run(args)
        result = compare(load_aggregate(args.baseline), load_aggregate(args.candidate), args.max_regression_percent)
        print(json.dumps(result, indent=2))
        return 1 if result["regressions"] else 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Performance measurement failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
