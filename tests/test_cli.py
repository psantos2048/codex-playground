import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "evm.py"

# A fresh process exercises startup as well as main(). Block imports to reproduce
# missing dependencies, and stub capture metadata that real files cannot reliably
# provide across codecs. The processing smoke test below uses real OpenCV I/O.
RUNNER = """
import importlib.abc
import json
import runpy
import sys
import types

script, options = sys.argv[1], json.loads(sys.argv[2])
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in options.get('blocked', []):
            raise ModuleNotFoundError('Simulated missing ' + fullname, name=fullname)
sys.meta_path.insert(0, BlockImports())

if options.get('metadata'):
    class Capture:
        def isOpened(self): return True
        def get(self, key):
            if key == 0:
                return 0.0 if options['metadata'] == 'invalid_fps' else 30.0
            return 32 if key in (1, 2) else 0
        def read(self): return False, None
        def release(self): pass
    sys.modules['cv2'] = types.SimpleNamespace(
        VideoCapture=lambda _path: Capture(), CAP_PROP_FPS=0,
        CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2, CAP_PROP_FOURCC=3,
    )

sys.argv = [script, *sys.argv[3:]]
runpy.run_path(script, run_name='__main__')
"""


def run_cli(*args, **options):
    return subprocess.run(
        [sys.executable, "-c", RUNNER, str(SCRIPT), json.dumps(options), *map(str, args)],
        capture_output=True, text=True, timeout=30,
    )


class CLITests(unittest.TestCase):
    def assert_clean_error(self, result, message):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Error: ", result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_invalid_path_reports_clean_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_cli(root / "missing.mp4", root / "output.mp4")
            self.assert_clean_error(result, "Failed to open input video")
            self.assertFalse((root / "output.mp4").exists())

    def test_invalid_video_metadata_reports_clean_error(self):
        for metadata, message in (("invalid_fps", "FPS"), ("no_frames", "no frames")):
            with self.subTest(metadata=metadata):
                result = run_cli("input.mp4", "output.mp4", metadata=metadata)
                self.assert_clean_error(result, message)

    def test_missing_runtime_dependencies_report_clean_errors(self):
        for module, message in (("numpy", "NumPy"), ("cv2", "OpenCV")):
            with self.subTest(module=module):
                result = run_cli("input.mp4", "output.mp4", blocked=[module])
                self.assert_clean_error(result, message)

    def test_help_works_without_runtime_dependencies(self):
        result = run_cli("--help", blocked=["numpy", "cv2"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-frames", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_invalid_max_frames_reports_clean_error(self):
        for limit in (-1, 0, 1):
            with self.subTest(limit=limit):
                result = run_cli(
                    "input.mp4", "output.mp4", "--max-frames", limit,
                    metadata="no_frames",
                )
                self.assert_clean_error(result, "--max-frames")

    def test_noninteger_max_frames_reports_clean_error(self):
        for limit in ("abc", "1.5", ""):
            with self.subTest(limit=limit):
                result = run_cli("input.mp4", "output.mp4", "--max-frames", limit)
                self.assert_clean_error(result, "--max-frames")


class VideoSmokeTests(unittest.TestCase):
    def test_real_mp4_processing_honors_frame_limit(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            output = Path(directory) / "output.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
            self.assertTrue(writer.isOpened(), "MP4 writer is unavailable")
            try:
                for index in range(60):
                    value = int(100 + 5 * np.sin(2 * np.pi * index / 30))
                    writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
            finally:
                writer.release()

            result = run_cli(
                source, output, "--low", "0.8", "--high", "1.2", "--alpha", "2",
                "--pyramid-level", "1", "--max-frames", "30",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened(), "Output video cannot be decoded")
            count = 0
            try:
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30.0, places=2)
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    self.assertEqual(frame.shape, (32, 32, 3))
                    count += 1
            finally:
                capture.release()
            self.assertEqual(count, 30)


if __name__ == "__main__":
    unittest.main()
