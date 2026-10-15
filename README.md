# codex-playground

## Eulerian Video Magnification script

This repository now includes `evm.py`, a Python program that takes an input video,
amplifies a selected temporal frequency band, and writes an output video.

### Install dependencies

```bash
pip install -r requirements.txt
```

If you want the `--interactive` slider window, replace the headless package with
GUI-enabled OpenCV in your Python environment. Install only one OpenCV variant:

```bash
python -m pip uninstall opencv-python-headless
python -m pip install "numpy>=1.24" "opencv-python>=4.8"
```

### Basic usage

```bash
python evm.py input.mp4 output.mp4 --low 0.8 --high 1.2 --alpha 30
```

### Interactive slider tuning

Use OpenCV sliders to choose frequency band and amplification before processing:

```bash
python evm.py input.mp4 output.mp4 --interactive
```

In the slider window:
- Press **Enter** to start processing with the selected values.
- Press **Esc** to cancel.

### Main options

- `--low`: lower cutoff frequency in Hz.
- `--high`: upper cutoff frequency in Hz.
- `--alpha`: amplification amount.
- `--pyramid-level`: number of spatial downsampling steps.
- `--chrom-attenuation`: dampening for chrominance channels to reduce color artifacts.
- `--max-frames`: process only the first N frames (at least 2) for quick experiments before running the full video.


### Tuning tips

- For subtle skin color changes such as pulse, start with a narrow band around the expected heart-rate range, for example `--low 0.8 --high 1.5 --alpha 20`.
- For small repetitive motion, use a band matching the motion frequency and increase `--alpha` gradually to avoid halos or clipping.
- Use `--max-frames` with `--interactive` to run short experiments quickly before processing a long source video.
- Short clips have coarser frequency resolution (`FPS / frame count`). Increase
  `--max-frames` if no FFT bins fall inside your selected band.

### Troubleshooting

- Invalid input videos, missing dependencies, and invalid frame limits report
  `Error: ...` on stderr and exit with status 1. `--help` works without NumPy or OpenCV.
- If NumPy fails to import under Cygwin, try reinstalling it with
  `python -m pip install --force-reinstall numpy` or use a native Windows/Linux
  Python distribution.
- If interactive sliders are unavailable, use the GUI-enabled OpenCV installation
  above and make sure your operating system provides the required GUI libraries.

### Run tests

```bash
python -m unittest discover -v
```

The suite covers frequency filtering, output shape and range, CLI error handling,
dependency-free help, capture cleanup, and a real synthetic-MP4 processing run
that verifies the `--max-frames` limit.
