import copy
import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "measure-performance.py"
SPEC = importlib.util.spec_from_file_location("measure_performance", MODULE_PATH)
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)

CONFIG = {"modelID": "mobileclip2-s4", "computeUnits": "all", "samples": 3,
          "warmups": 1, "batchSize": 2, "indexSizes": [100], "topK": 120, "includeSearch": True}
HOST = {"xcode": "test Xcode", "buildConfiguration": "Release"}


def report(scale=1):
    metrics = {}
    for name in perf.required_metrics(CONFIG):
        repeated = name in {"image_batch", "image_batch_1", "text_query", "search_100", "text_search_100"}
        metrics[name] = {"samplesMilliseconds": [scale, 2 * scale, 3 * scale] if repeated else [scale],
                         "workItemsPerSecond": 100 / scale}
    return {"schemaVersion": 1, "status": "complete", "configuration": copy.deepcopy(CONFIG),
            "modelID": "mobileclip2-s4", "workloadVersion": "test-v1",
            "hardware": "test-device", "platform": "iOS-device", "metrics": metrics,
            "actualImageComputeUnits": "all", "actualTextComputeUnits": "all",
            "artifacts": [{"name": str(i), "sha256": "a" * 64, "bytes": 100} for i in range(4)],
            "memory": {"successfulSamples": 10, "sampledPeakBytes": 1024, "thermalStates": ["nominal"]}}


class PerformanceReportTests(unittest.TestCase):
    def test_median_across_runs_is_not_distorted_by_one_slow_run(self):
        result = perf.aggregate([report(1), report(2), report(100)], HOST)
        self.assertEqual(result["metrics"]["image_batch"]["medianMilliseconds"], 4)
        self.assertEqual(result["metrics"]["image_batch"]["p95Milliseconds"], 300)

    def test_failed_or_missing_work_never_passes_validation(self):
        for mutate in [lambda r: r.update(status="failed"),
                       lambda r: r["metrics"].pop("text_search_100"),
                       lambda r: r["metrics"]["image_batch"].update(samplesMilliseconds=[1]),
                       lambda r: r["metrics"]["image_batch"].update(samplesMilliseconds=[1, float("nan"), 3]),
                       lambda r: r["memory"].update(successfulSamples=0),
                       lambda r: r.update(artifacts=[]),
                       lambda r: r.update(actualTextComputeUnits=None)]:
            candidate = report()
            mutate(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                perf.validate_report(candidate, CONFIG)

    def test_aggregation_rejects_mixed_hardware(self):
        other = report()
        other["hardware"] = "other-device"
        with self.assertRaisesRegex(ValueError, "different devices"):
            perf.aggregate([report(), other], HOST)

    def test_model_only_reports_do_not_claim_search_measurements(self):
        candidate = report()
        config = copy.deepcopy(CONFIG)
        config["includeSearch"] = False
        candidate["configuration"] = config
        expected = perf.required_metrics(config)
        candidate["metrics"] = {key: value for key, value in candidate["metrics"].items() if key in expected}
        perf.validate_report(candidate, config)
        self.assertEqual(len(candidate["metrics"]), 7)

    def test_regression_threshold_and_improvement(self):
        baseline = perf.aggregate([report()], HOST)
        for scale, expected in [(1.14, False), (1.16, True), (0.5, False)]:
            with self.subTest(scale=scale):
                candidate = perf.aggregate([report(scale)], HOST)
                result = perf.compare(baseline, candidate, 15)
                self.assertEqual(bool(result["regressions"]), expected)

    def test_mismatched_or_hot_reports_are_rejected(self):
        baseline = perf.aggregate([report()], HOST)
        for mutate in [lambda r: r["comparisonKey"].update(hardware="other-device"),
                       lambda r: r["comparisonKey"].update(actualTextComputeUnits="cpuOnly"),
                       lambda r: r["comparisonKey"].update(artifacts=[]),
                       lambda r: r.update(thermalStates=["serious"]),
                       lambda r: r.update(metrics={})]:
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                perf.compare(baseline, candidate, 15)


if __name__ == "__main__":
    unittest.main()
