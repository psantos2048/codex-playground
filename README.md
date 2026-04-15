# codex-playground

## Eulerian Video Magnification script

This repository now includes `evm.py`, a Python program that takes an input video,
amplifies a selected temporal frequency band, and writes an output video.

### Install dependencies

```bash
pip install numpy opencv-python-headless
```

If you want the `--interactive` slider window, install GUI-enabled OpenCV instead:

```bash
pip install opencv-python
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
