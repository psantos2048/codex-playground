import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import evm


class EVMTests(unittest.TestCase):
    def test_validate_frame_count_rejects_unresolvable_band(self):
        # 10 frames at 10 FPS => 1 Hz frequency resolution.
        # 0.2-0.4 Hz contains no FFT bins.
        with self.assertRaises(ValueError):
            evm.validate_frame_count(n_frames=10, fps=10.0, low_hz=0.2, high_hz=0.4)

    def test_temporal_bandpass_preserves_in_band_frequency(self):
        fps = 30.0
        n = 120
        t = np.arange(n) / fps

        signal = np.sin(2 * np.pi * 1.0 * t) + 0.3 * np.sin(2 * np.pi * 4.0 * t)
        video = signal[:, None, None, None].astype(np.float32)

        filtered = evm.temporal_ideal_bandpass(video, fps=fps, low=0.8, high=1.2)
        out = filtered[:, 0, 0, 0]

        corr_1hz = np.corrcoef(out, np.sin(2 * np.pi * 1.0 * t))[0, 1]
        corr_4hz = np.corrcoef(out, np.sin(2 * np.pi * 4.0 * t))[0, 1]

        self.assertGreater(corr_1hz, 0.9)
        self.assertLess(abs(corr_4hz), 0.2)

    def test_magnify_video_keeps_shape_and_range(self):
        rng = np.random.default_rng(0)
        video = rng.random((16, 32, 32, 3), dtype=np.float32)
        config = evm.EVMConfig(
            low_hz=0.5,
            high_hz=1.0,
            alpha=10.0,
            pyramid_level=1,
            chrom_attenuation=0.1,
        )

        out = evm.magnify_video(video, fps=8.0, config=config)

        self.assertEqual(out.shape, video.shape)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_capture_is_released_when_fps_is_invalid(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 0.0
        backend = SimpleNamespace(VideoCapture=lambda _path: capture, CAP_PROP_FPS=0)
        with patch.object(evm, "cv2_module", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "FPS"):
                evm.read_video_frames(Path("input.mp4"))
        capture.release.assert_called_once_with()

    def test_interactive_tuner_explains_unavailable_gui(self):
        backend = Mock()
        backend.namedWindow.side_effect = RuntimeError("GUI unavailable")
        config = evm.EVMConfig(0.8, 1.2, 2.0, 1, 0.1)
        with patch.object(evm, "cv2_module", return_value=backend):
            with self.assertRaisesRegex(RuntimeError, "GUI-enabled OpenCV"):
                evm.run_interactive_tuner(config, fps=30.0)


if __name__ == "__main__":
    unittest.main()
