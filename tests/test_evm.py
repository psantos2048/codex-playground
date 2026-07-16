import unittest

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


if __name__ == "__main__":
    unittest.main()
