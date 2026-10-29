"""Synthetic arrow images and a NumPy-only three-class neural classifier.

The camera input in this educational simulation is a procedurally rendered sign
patch. It is not a photograph, object detector, or a calibrated camera model.
No target label is accepted by SignClassifier.predict().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASS_NAMES = ["LEFT", "STRAIGHT", "RIGHT"]
IMAGE_SIZE = 96
FEATURE_SIZE = 24
ROOT = Path(__file__).resolve().parent


def render_sign(label: int, rng: np.random.Generator,
                profile: str = "standard") -> Image.Image:
    """Render one independently varied 96 x 96 RGB sign patch.

    ``standard`` is the common training/validation/test distribution; ``stress``
    changes size, rotation, blur, contrast and partial occlusion jointly.
    ``canonical`` supplies centered reference images for an illustration.
    """
    if label not in range(len(CLASS_NAMES)):
        raise ValueError("label must be 0, 1, or 2")
    if profile not in {"standard", "stress", "canonical"}:
        raise ValueError("unknown rendering profile")
    canonical = profile == "canonical"
    stress = profile == "stress"
    if canonical:
        scale, shift_x, shift_y, angle = 1.0, 0.0, 0.0, 0.0
        bg, fg, blur, noise_sd = 242, 35, 0.0, 0.0
    else:
        scale = float(rng.uniform(0.52, 0.95) if stress else rng.uniform(0.78, 1.12))
        delta = 11 if stress else 5
        shift_x, shift_y = rng.uniform(-delta, delta, size=2)
        angle = float(rng.uniform(-38, 38) if stress else rng.uniform(-13, 13))
        bg = int(rng.integers(155, 246) if stress else rng.integers(220, 253))
        fg = int(rng.integers(50, 125) if stress else rng.integers(15, 72))
        blur = float(rng.uniform(0.5, 2.8) if stress else rng.uniform(0.0, 0.65))
        noise_sd = float(rng.uniform(7.0, 20.0) if stress else rng.uniform(0.0, 6.0))

    # One hand-authored generic upward arrow, rotated by the requested class.
    vertices = np.array([[-10, 27], [10, 27], [10, -1], [24, -1],
                         [0, -29], [-24, -1], [-10, -1]], dtype=float)
    # Image y increases downwards: negative rotation sends an upward tip left.
    class_angle = [-90.0, 0.0, 90.0][label]
    theta = np.deg2rad(class_angle + angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    vertices = (vertices @ rotation.T) * scale + [48 + shift_x, 48 + shift_y]
    # Draw at double resolution to avoid class-dependent raster edge artifacts.
    up = 2
    tile = Image.new("RGB", (IMAGE_SIZE * up, IMAGE_SIZE * up), (bg, bg, min(255, bg + 2)))
    draw = ImageDraw.Draw(tile)
    border = max(0, bg - 22)
    draw.rounded_rectangle((5 * up, 5 * up, 91 * up, 91 * up),
                           radius=9 * up, outline=(border, border, border), width=up)
    draw.polygon([tuple(v * up) for v in vertices], fill=(fg, min(255, fg + 7), min(255, fg + 18)))
    tile = tile.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    if stress and rng.random() < 0.45:
        # A neutral strip occludes a random portion of the sign; it never carries
        # a class label or a code that the classifier could read.
        d = ImageDraw.Draw(tile)
        y = int(rng.integers(31, 66))
        width = int(rng.integers(5, 13))
        d.rectangle((15, y, 81, y + width), fill=(bg, bg, min(255, bg + 2)))
    if blur:
        tile = tile.filter(ImageFilter.GaussianBlur(blur))
    if noise_sd:
        pixels = np.asarray(tile, dtype=np.float32)
        noise = rng.normal(0.0, noise_sd, pixels.shape[:2])[..., None]
        tile = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
    return tile


def preprocess(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return the 576 float32 inputs used during both training and inference."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    image = image.convert("L").resize((FEATURE_SIZE, FEATURE_SIZE), Image.Resampling.BILINEAR)
    return (1.0 - np.asarray(image, dtype=np.float32) / 255.0).reshape(-1)


class SignClassifier:
    """MLP inference from numeric NPZ weights; loading does not execute pickle."""

    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path) if model_path is not None else ROOT / "models" / "robust_mlp.npz"
        with np.load(path, allow_pickle=False) as archive:
            self.weights = [archive[f"weight_{i}"].copy()
                            for i in range(int(archive["n_layers"]))]
            self.biases = [archive[f"bias_{i}"].copy()
                           for i in range(int(archive["n_layers"]))]
            if archive["class_names"].tolist() != CLASS_NAMES:
                raise ValueError("model class order differs from the simulation API")

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        one = x.ndim == 1
        if one:
            x = x[None, :]
        if x.shape[1] != FEATURE_SIZE * FEATURE_SIZE:
            raise ValueError("expected 576 input features")
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            if index < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        x = x - np.max(x, axis=1, keepdims=True)
        p = np.exp(x)
        p /= np.sum(p, axis=1, keepdims=True)
        return p[0] if one else p

    def predict(self, image: Image.Image | np.ndarray) -> dict:
        p = self.probabilities(preprocess(image))
        label = int(np.argmax(p))
        return {"label": label, "name": CLASS_NAMES[label],
                "confidence": float(p[label]), "probabilities": p.tolist()}




def readability(image):
    """Pixel-only contrast gate for the explicitly tested covered-panel case.

    Not an open-set detector and not a guarantee of readable arrows.
    Cropping excludes the decorative border. Returns gray-level contrast.
    """
    a = np.asarray(image.convert("L"), dtype=np.float32)[18:78,18:78]
    return float(np.percentile(a, 95)-np.percentile(a, 5))
