#!/usr/bin/env python3
"""Eulerian Video Magnification for amplifying selected temporal frequencies.

This script reads an input video, applies a temporal band-pass filter at each
pixel of a low-resolution spatial representation, amplifies the filtered signal,
and reconstructs an output video.

You can configure frequency bounds and amplification with CLI options, or launch
an interactive slider UI to tune those values before processing.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_CV2 = None


def cv2_module():
    global _CV2
    if _CV2 is None:
        try:
            _CV2 = importlib.import_module("cv2")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "OpenCV (cv2) is required. Install with `pip install opencv-python` "
                "and ensure system GUI/OpenGL libs are available."
            ) from exc
    return _CV2


@dataclass
class EVMConfig:
    low_hz: float
    high_hz: float
    alpha: float
    pyramid_level: int
    chrom_attenuation: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Amplify chosen temporal frequencies in a video using EVM."
    )
    parser.add_argument("input", type=Path, help="Path to input video")
    parser.add_argument("output", type=Path, help="Path to output video")
    parser.add_argument(
        "--low",
        type=float,
        default=0.8,
        help="Low cutoff frequency in Hz (default: 0.8)",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=1.0,
        help="High cutoff frequency in Hz (default: 1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=20.0,
        help="Amplification factor (default: 20.0)",
    )
    parser.add_argument(
        "--pyramid-level",
        type=int,
        default=4,
        help="How aggressively to downsample spatially (default: 4)",
    )
    parser.add_argument(
        "--chrom-attenuation",
        type=float,
        default=0.1,
        help=(
            "Attenuation factor for chrominance-like channels in YIQ space "
            "(default: 0.1)"
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open slider UI to set low/high frequency and alpha before processing",
    )
    return parser.parse_args()


def validate_config(config: EVMConfig, fps: float) -> None:
    nyquist = fps / 2.0
    if config.low_hz < 0:
        raise ValueError("--low must be non-negative")
    if config.high_hz <= config.low_hz:
        raise ValueError("--high must be greater than --low")
    if config.high_hz >= nyquist:
        raise ValueError(
            f"--high must be < Nyquist ({nyquist:.3f}Hz) for this video's FPS ({fps:.3f})"
        )
    if config.alpha < 0:
        raise ValueError("--alpha must be non-negative")
    if config.pyramid_level < 0:
        raise ValueError("--pyramid-level must be >= 0")
    if config.chrom_attenuation < 0:
        raise ValueError("--chrom-attenuation must be non-negative")


def read_video_frames(path: Path) -> tuple[np.ndarray, float, tuple[int, int], str]:
    cv2 = cv2_module()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Could not determine FPS from the input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc_val >> 8 * i) & 0xFF) for i in range(4)])

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.float32) / 255.0)

    cap.release()

    if not frames:
        raise RuntimeError("Input video contains no frames")

    return np.stack(frames, axis=0), fps, (width, height), codec


def validate_frame_count(n_frames: int, fps: float, low_hz: float, high_hz: float) -> None:
    if n_frames < 2:
        raise ValueError("Need at least 2 frames for temporal frequency analysis")

    freqs = np.fft.fftfreq(n_frames, d=1.0 / fps)
    positive_bins = np.abs(freqs)
    in_band = (positive_bins >= low_hz) & (positive_bins <= high_hz)
    if not np.any(in_band):
        min_resolvable = fps / n_frames
        raise ValueError(
            "No FFT bins fall inside the requested passband. "
            f"Try more frames or adjust --low/--high (frequency resolution is ~{min_resolvable:.4f}Hz)."
        )


def gaussian_downsample(video: np.ndarray, levels: int) -> np.ndarray:
    cv2 = cv2_module()
    reduced = video
    for _ in range(levels):
        reduced_frames = [cv2.pyrDown(frame) for frame in reduced]
        reduced = np.stack(reduced_frames, axis=0)
    return reduced


def gaussian_upsample(video: np.ndarray, levels: int, target_shape: tuple[int, int]) -> np.ndarray:
    cv2 = cv2_module()
    expanded = video
    for _ in range(levels):
        expanded_frames = [cv2.pyrUp(frame) for frame in expanded]
        expanded = np.stack(expanded_frames, axis=0)

    target_h, target_w = target_shape
    resized = np.stack(
        [cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for frame in expanded],
        axis=0,
    )
    return resized


def rgb_to_yiq(video: np.ndarray) -> np.ndarray:
    m = np.array(
        [
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312],
        ],
        dtype=np.float32,
    )
    return np.tensordot(video, m.T, axes=1)


def yiq_to_rgb(video: np.ndarray) -> np.ndarray:
    m_inv = np.array(
        [
            [1.0, 0.956, 0.621],
            [1.0, -0.272, -0.647],
            [1.0, -1.106, 1.703],
        ],
        dtype=np.float32,
    )
    return np.tensordot(video, m_inv.T, axes=1)


def temporal_ideal_bandpass(video: np.ndarray, fps: float, low: float, high: float) -> np.ndarray:
    n_frames = video.shape[0]
    freqs = np.fft.fftfreq(n_frames, d=1.0 / fps)
    fft_video = np.fft.fft(video, axis=0)

    passband = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    fft_video[~passband] = 0

    filtered = np.fft.ifft(fft_video, axis=0).real.astype(np.float32)
    return filtered


def magnify_video(video: np.ndarray, fps: float, config: EVMConfig) -> np.ndarray:
    validate_frame_count(video.shape[0], fps, config.low_hz, config.high_hz)
    reduced = gaussian_downsample(video, config.pyramid_level)

    yiq = rgb_to_yiq(reduced)
    filtered = temporal_ideal_bandpass(yiq, fps, config.low_hz, config.high_hz)

    filtered[..., 1:] *= config.chrom_attenuation

    # Amplify only the band-passed temporal signal and add it back to
    # the full-resolution source to preserve image detail.
    amplified_signal = config.alpha * filtered
    amplified_rgb = yiq_to_rgb(amplified_signal)

    upsampled_signal = gaussian_upsample(
        amplified_rgb, config.pyramid_level, target_shape=video.shape[1:3]
    )
    combined = video + upsampled_signal

    return np.clip(combined, 0.0, 1.0)


def write_video(frames_rgb: np.ndarray, path: Path, fps: float, size: tuple[int, int], codec_hint: str) -> None:
    cv2 = cv2_module()
    if codec_hint.strip("\x00"):
        fourcc = cv2.VideoWriter_fourcc(*codec_hint)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        # Fallback codec for broader compatibility.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {path}")

    for frame_rgb in frames_rgb:
        frame_bgr = cv2.cvtColor((frame_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()


def run_interactive_tuner(defaults: EVMConfig, fps: float) -> EVMConfig:
    cv2 = cv2_module()
    window = "EVM Tuning"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    max_hz_slider = max(1, int((fps / 2.0) * 100))

    cv2.createTrackbar("low x100", window, int(defaults.low_hz * 100), max_hz_slider, lambda _v: None)
    cv2.createTrackbar("high x100", window, int(defaults.high_hz * 100), max_hz_slider, lambda _v: None)
    cv2.createTrackbar("alpha", window, int(defaults.alpha), 200, lambda _v: None)
    cv2.createTrackbar(
        "chrom x100",
        window,
        int(defaults.chrom_attenuation * 100),
        100,
        lambda _v: None,
    )

    print("Interactive tuning window opened.")
    print("Adjust sliders, then press ENTER in the window to start processing.")
    print("Press ESC to cancel.")

    while True:
        low = cv2.getTrackbarPos("low x100", window) / 100.0
        high = cv2.getTrackbarPos("high x100", window) / 100.0
        alpha = float(cv2.getTrackbarPos("alpha", window))
        chrom = cv2.getTrackbarPos("chrom x100", window) / 100.0

        preview = np.zeros((130, 540, 3), dtype=np.uint8)
        lines = [
            f"low={low:.2f}Hz high={high:.2f}Hz",
            f"alpha={alpha:.1f} chrom_attenuation={chrom:.2f}",
            "ENTER = process, ESC = cancel",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                preview,
                text,
                (10, 35 + 35 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window, preview)
        key = cv2.waitKey(50) & 0xFF
        if key in (10, 13):
            cv2.destroyWindow(window)
            return EVMConfig(
                low_hz=low,
                high_hz=high,
                alpha=alpha,
                pyramid_level=defaults.pyramid_level,
                chrom_attenuation=chrom,
            )
        if key == 27:
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("User canceled interactive tuning")


def main() -> int:
    args = parse_args()

    video, fps, size, codec_hint = read_video_frames(args.input)

    config = EVMConfig(
        low_hz=args.low,
        high_hz=args.high,
        alpha=args.alpha,
        pyramid_level=args.pyramid_level,
        chrom_attenuation=args.chrom_attenuation,
    )

    try:
        if args.interactive:
            config = run_interactive_tuner(config, fps)

        validate_config(config, fps)
        processed = magnify_video(video, fps, config)
        write_video(processed, args.output, fps, size, codec_hint)
    except KeyboardInterrupt as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Done. Wrote amplified video to '{args.output}'. "
        f"Settings: low={config.low_hz:.2f}Hz high={config.high_hz:.2f}Hz "
        f"alpha={config.alpha:.2f} pyramid_level={config.pyramid_level} "
        f"chrom_attenuation={config.chrom_attenuation:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
