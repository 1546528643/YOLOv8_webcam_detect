import argparse
import time
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 webcam real-time object detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to model weights or model name (e.g., yolov8n.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto/cpu/cuda:0",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default: 0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision (FP16) when supported",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Display FPS on the output window",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save the annotated video (e.g., out.mp4)",
    )
    return parser.parse_args()


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=class_id + 42)
    color = rng.integers(low=64, high=256, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def draw_detections(
    frame: np.ndarray,
    names: dict,
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    class_ids: np.ndarray,
) -> np.ndarray:
    for xyxy, conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        class_id_int = int(cls_id)
        color = get_color_for_class(class_id_int)
        label = f"{names.get(class_id_int, str(class_id_int))} {float(conf):.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        top = max(y1, text_height + 6)
        cv2.rectangle(
            frame,
            (x1, top - text_height - 6),
            (x1 + text_width + 6, top + baseline - 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, top - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return frame


def maybe_init_writer(
    save_path: str, frame_width: int, frame_height: int, fps: float
) -> cv2.VideoWriter | None:
    if not save_path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"[WARN] 无法打开视频写入器: {save_path}")
        return None
    print(f"[INFO] 保存视频到: {save_path} ({frame_width}x{frame_height} @ {fps:.1f} fps)")
    return writer


def main() -> None:
    args = parse_args()

    model_path = args.model
    device = args.device
    image_size = args.imgsz
    confidence_threshold = args.conf
    iou_threshold = args.iou
    use_half_precision = args.half

    # 解析推理设备：当传入 auto 时，根据是否可用 CUDA 自动选择
    resolved_device = device
    try:
        if isinstance(device, str) and device.lower() == "auto":
            try:
                import torch  # 延迟导入以避免无谓依赖
                if torch.cuda.is_available():
                    resolved_device = "cuda:0"
                else:
                    resolved_device = "cpu"
            except Exception:
                resolved_device = "cpu"
    except Exception:
        resolved_device = "cpu"

    # 若未使用 CUDA，则禁用 FP16 以避免潜在错误
    if use_half_precision and not (isinstance(resolved_device, str) and resolved_device.startswith("cuda")):
        use_half_precision = False
        print("[INFO] 未检测到 CUDA，已禁用 FP16（--half）。")

    print(f"[INFO] 加载模型: {model_path}")
    print(f"[INFO] 推理设备: {resolved_device}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头索引 {args.camera}")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    writer = maybe_init_writer(args.save, frame_width, frame_height, input_fps)

    last_time = time.time()
    fps_smooth = 0.0

    window_title = "YOLOv8 Webcam Detection (press 'q' to quit)"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 从摄像头读取失败，尝试继续...")
                time.sleep(0.01)
                continue

            results = model(
                frame,
                imgsz=image_size,
                conf=confidence_threshold,
                iou=iou_threshold,
                device=resolved_device,
                half=use_half_precision,
                verbose=False,
            )

            result = results[0]
            names = result.names if hasattr(result, "names") else model.names

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                frame = draw_detections(frame, names, boxes_xyxy, confidences, class_ids)

            current_time = time.time()
            instantaneous_fps = 1.0 / max(current_time - last_time, 1e-6)
            last_time = current_time
            fps_smooth = fps_smooth * 0.9 + instantaneous_fps * 0.1

            if args.show_fps:
                text = f"FPS: {fps_smooth:.1f}"
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(frame)

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，正在退出...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

    #python src/webcam_detect.py --model .\yolov8n.pt