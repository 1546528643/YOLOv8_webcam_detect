import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class TargetSpec:
    center_x: int
    center_y: int
    outer_radius: int
    ring_count: int
    ring_thickness: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def draw_concentric_target(canvas: np.ndarray, spec: TargetSpec) -> None:
    # 黑白十环样式：最外圈为白色，内外交替，中心为黑色
    for i in range(spec.ring_count):
        radius = spec.outer_radius - i * spec.ring_thickness
        if radius <= 0:
            break
        # i 偶数为白色，奇数为黑色，从外到内交替，10 环时中心为黑色
        color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
        cv2.circle(
            canvas,
            (spec.center_x, spec.center_y),
            radius,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def add_background_noise(img: np.ndarray, intensity: float = 0.05) -> None:
    if intensity <= 0:
        return
    noise = np.random.normal(loc=0.0, scale=255 * intensity, size=img.shape).astype(
        np.float32
    )
    img[:] = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_affine(
    img: np.ndarray, bbox_xywh: Tuple[float, float, float, float], max_rotate_deg: float
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    h, w = img.shape[:2]

    # Random rotation and scale
    angle = random.uniform(-abs(max_rotate_deg), abs(max_rotate_deg))
    scale = random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Transform bbox corners
    cx, cy, bw, bh = bbox_xywh
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    corners = np.array(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]], dtype=np.float32
    )
    transformed = (M @ corners.T).T
    x_min = float(np.clip(np.min(transformed[:, 0]), 0, w - 1))
    y_min = float(np.clip(np.min(transformed[:, 1]), 0, h - 1))
    x_max = float(np.clip(np.max(transformed[:, 0]), 0, w - 1))
    y_max = float(np.clip(np.max(transformed[:, 1]), 0, h - 1))
    new_bw = max(1.0, x_max - x_min)
    new_bh = max(1.0, y_max - y_min)
    new_cx = x_min + new_bw / 2.0
    new_cy = y_min + new_bh / 2.0

    return img_rot, (new_cx, new_cy, new_bw, new_bh)


def to_yolo_line(
    class_id: int,
    bbox_xywh: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> str:
    cx, cy, bw, bh = bbox_xywh
    nx = cx / img_w
    ny = cy / img_h
    nw = bw / img_w
    nh = bh / img_h
    nx = min(max(nx, 0.0), 1.0)
    ny = min(max(ny, 0.0), 1.0)
    nw = min(max(nw, 1.0 / img_w), 1.0)
    nh = min(max(nh, 1.0 / img_h), 1.0)
    return f"{class_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"


def generate_one_image(
    img_size: int,
    min_outer_frac: float,
    max_outer_frac: float,
    rings: int,
    add_noise: bool,
    max_rotate_deg: float,
) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
    h = w = img_size
    background = np.full((h, w, 3), 255, dtype=np.uint8)  # 纯白背景

    # Random target spec within image
    outer_radius = int(random.uniform(min_outer_frac, max_outer_frac) * img_size / 2.0)
    ring_count = max(1, int(rings))
    ring_thickness = max(2, int(math.ceil(outer_radius / ring_count)))

    margin = outer_radius + 2
    cx = random.randint(margin, w - margin - 1)
    cy = random.randint(margin, h - margin - 1)

    spec = TargetSpec(
        center_x=cx,
        center_y=cy,
        outer_radius=outer_radius,
        ring_count=ring_count,
        ring_thickness=ring_thickness,
    )

    draw_concentric_target(background, spec)

    # Bounding box of the full target (outer ring)
    x_min = cx - outer_radius
    y_min = cy - outer_radius
    x_max = cx + outer_radius
    y_max = cy + outer_radius
    bw = x_max - x_min
    bh = y_max - y_min
    bbox_xywh = (cx, cy, float(bw), float(bh))

    # 旋转/缩放仿射增强，允许更大角度
    background, bbox_xywh = random_affine(background, bbox_xywh, max_rotate_deg=max_rotate_deg)

    # Add mild noise
    if add_noise:
        add_background_noise(background, intensity=0.03)

    return background, [bbox_xywh]


def write_image_and_label(
    img: np.ndarray,
    bboxes_xywh: List[Tuple[float, float, float, float]],
    img_path: Path,
    label_path: Path,
    class_id: int,
) -> None:
    ensure_dir(img_path.parent)
    ensure_dir(label_path.parent)
    # Save image
    cv2.imwrite(str(img_path), img)
    # Save labels
    h, w = img.shape[:2]
    lines = [to_yolo_line(class_id, b, w, h) for b in bboxes_xywh]
    label_text = "\n".join(lines) + ("\n" if lines else "")
    with label_path.open("w", encoding="utf-8") as f:
        f.write(label_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic black-white concentric-target dataset (e.g., 10-ring) for YOLOv8"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/targets"),
        help="Output dataset root (contains images/ and labels/)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    parser.add_argument("--train", type=int, default=500, help="Number of train images")
    parser.add_argument("--val", type=int, default=100, help="Number of val images")
    parser.add_argument(
        "--min-outer-frac",
        type=float,
        default=0.3,
        help="Min outer diameter as fraction of image size",
    )
    parser.add_argument(
        "--max-outer-frac",
        type=float,
        default=0.7,
        help="Max outer diameter as fraction of image size",
    )
    parser.add_argument("--rings", type=int, default=10, help="Number of rings (e.g., 10)")
    parser.add_argument(
        "--max-rotate",
        type=float,
        default=45.0,
        help="Max absolute rotation angle in degrees",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Add mild background noise",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_root: Path = args.out
    images_train = out_root / "images" / "train"
    images_val = out_root / "images" / "val"
    labels_train = out_root / "labels" / "train"
    labels_val = out_root / "labels" / "val"

    ensure_dir(images_train)
    ensure_dir(images_val)
    ensure_dir(labels_train)
    ensure_dir(labels_val)

    # Generate training set
    for i in tqdm(range(args.train), desc="Generating train"):
        img, bboxes = generate_one_image(
            img_size=args.imgsz,
            min_outer_frac=args.min_outer_frac,
            max_outer_frac=args.max_outer_frac,
            rings=args.rings,
            add_noise=args.noise,
            max_rotate_deg=args.max_rotate,
        )
        stem = f"train_{i:06d}"
        img_path = images_train / f"{stem}.jpg"
        label_path = labels_train / f"{stem}.txt"
        write_image_and_label(img, bboxes, img_path, label_path, class_id=0)

    # Generate validation set
    for i in tqdm(range(args.val), desc="Generating val"):
        img, bboxes = generate_one_image(
            img_size=args.imgsz,
            min_outer_frac=args.min_outer_frac,
            max_outer_frac=args.max_outer_frac,
            rings=args.rings,
            add_noise=args.noise,
            max_rotate_deg=args.max_rotate,
        )
        stem = f"val_{i:06d}"
        img_path = images_val / f"{stem}.jpg"
        label_path = labels_val / f"{stem}.txt"
        write_image_and_label(img, bboxes, img_path, label_path, class_id=0)

    print("[INFO] Dataset generated at:", out_root.resolve())
    data_yaml = out_root / "data.yaml"
    if data_yaml.exists():
        print("[INFO] Using existing:", data_yaml)
    else:
        try:
            with data_yaml.open("w", encoding="utf-8") as f:
                f.write(
                    "path: .\ntrain: images/train\nval: images/val\n\nnc: 1\nnames: [biaoba]\n"
                )
            print("[INFO] Wrote:", data_yaml)
        except Exception as e:
            print("[WARN] Failed to write data.yaml:", e)


if __name__ == "__main__":
    main()

