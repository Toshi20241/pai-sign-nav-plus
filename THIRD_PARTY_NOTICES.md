# Dependencies and references

This project includes original procedural sign images, numeric MLP parameters and simulation code. It does not include real photographs, real maps, external API credentials, or font binaries.

External libraries installed by the user:

- NumPy (BSD-3-Clause): https://numpy.org/
- Pillow (HPND): https://python-pillow.org/
- scikit-learn (BSD-3-Clause): https://scikit-learn.org/
- threadpoolctl (BSD-3-Clause): https://github.com/joblib/threadpoolctl
- ReportLab (BSD-style): https://www.reportlab.com/
- pytest (MIT): https://docs.pytest.org/
- FFmpeg, separately installed (build-specific LGPL/GPL terms): https://ffmpeg.org/legal.html

The recorded video and PDF use locally installed Japanese fonts. Font software is not redistributed. To reproduce the media, use a Japanese font installed on your system and observe its license.

The classifier implements the standard ReLU MLP forward pass; training uses scikit-learn. No claim of a newly invented learning algorithm, calibrated safety probability, real-robot validation, or VLA/RL controller is made.
