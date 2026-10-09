"""Exercise clean-build packaging without requiring large model downloads."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "bundle-coreml-models.sh"
MODELS = ["ImageEncoder_mobileCLIP2_s4.mlmodelc", "TextEncoder_mobileCLIP2_s4.mlmodelc"]


class ModelPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.resources = self.project / "Queryable/CoreMLModels"
        self.resources.mkdir(parents=True)
        self.cache = self.root / "models"
        self.destination = self.root / "build/Queryable.app/CoreMLModels"
        for name in ["vocab.json", "merges.txt"]:
            (self.resources / name).write_text("tokenizer fixture")
        for name in MODELS:
            self.write_model(self.cache, name, "cached model")

    def write_model(self, parent, name, content):
        model = parent / name
        (model / "weights").mkdir(parents=True, exist_ok=True)
        for file in ["coremldata.bin", "model.mil", "weights/weight.bin"]:
            (model / file).write_text(content)

    def package(self):
        environment = dict(
            os.environ,
            SRCROOT=str(self.project),
            TARGET_BUILD_DIR=str(self.root / "build"),
            UNLOCALIZED_RESOURCES_FOLDER_PATH="Queryable.app",
        )
        return subprocess.run(["/bin/sh", str(SCRIPT)], env=environment, capture_output=True, text=True)

    def test_clean_output_uses_repository_models(self):
        result = self.package()
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in MODELS:
            self.assertEqual((self.destination / name / "weights/weight.bin").read_text(), "cached model")
        self.assertTrue((self.destination / "vocab.json").is_file())
        self.assertTrue((self.destination / "merges.txt").is_file())

    def test_complete_local_model_overrides_cache_and_updates_existing_output(self):
        self.assertEqual(self.package().returncode, 0)
        self.write_model(self.resources, MODELS[0], "local model")
        self.assertEqual(self.package().returncode, 0)
        self.assertEqual((self.destination / MODELS[0] / "weights/weight.bin").read_text(), "local model")

    def test_incomplete_local_model_uses_complete_cache(self):
        self.write_model(self.resources, MODELS[0], "incomplete model")
        (self.resources / MODELS[0] / "weights/weight.bin").unlink()
        self.assertEqual(self.package().returncode, 0)
        self.assertEqual((self.destination / MODELS[0] / "weights/weight.bin").read_text(), "cached model")

    def test_missing_model_fails_before_creating_app_resources(self):
        (self.cache / MODELS[1] / "weights/weight.bin").unlink()
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing or incomplete " + MODELS[1], result.stderr)
        self.assertFalse(self.destination.exists())

    def test_missing_tokenizer_fails_before_creating_app_resources(self):
        (self.resources / "vocab.json").unlink()
        result = self.package()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing tokenizer asset", result.stdout)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
