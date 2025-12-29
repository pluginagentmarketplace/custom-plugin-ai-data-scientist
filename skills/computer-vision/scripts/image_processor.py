#!/usr/bin/env python3
"""
Computer Vision Image Processing Utilities
Preprocessing, augmentation, and visualization tools
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import json


class ImageProcessor:
    """Comprehensive image processing utilities for CV tasks."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image from file path."""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def resize_with_padding(self, image: np.ndarray,
                           keep_aspect: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Resize image with letterbox padding to maintain aspect ratio.

        Returns:
            Tuple of (resized_image, scale_info)
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        if keep_aspect:
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create padded image
            padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

            # Center the image
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

            scale_info = {
                'scale': scale,
                'pad_w': pad_w,
                'pad_h': pad_h,
                'original_size': (w, h),
                'new_size': (new_w, new_h)
            }
        else:
            padded = cv2.resize(image, (target_w, target_h))
            scale_info = {
                'scale_x': target_w / w,
                'scale_y': target_h / h,
                'original_size': (w, h)
            }

        return padded, scale_info

    def normalize(self, image: np.ndarray,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
        """Normalize image with ImageNet statistics."""
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(mean)) / np.array(std)
        return image

    def denormalize(self, image: np.ndarray,
                   mean: List[float] = [0.485, 0.456, 0.406],
                   std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
        """Denormalize image back to [0, 255] range."""
        image = image * np.array(std) + np.array(mean)
        image = (image * 255).clip(0, 255).astype(np.uint8)
        return image


class BoundingBoxUtils:
    """Utilities for bounding box operations."""

    @staticmethod
    def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x_center, y_center, width, height]."""
        result = boxes.copy()
        result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
        result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
        result[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        result[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        return result

    @staticmethod
    def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]."""
        result = boxes.copy()
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return result

    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    @staticmethod
    def non_max_suppression(boxes: np.ndarray, scores: np.ndarray,
                           iou_threshold: float = 0.5) -> List[int]:
        """Apply Non-Maximum Suppression to filter overlapping boxes."""
        if len(boxes) == 0:
            return []

        # Sort by scores
        indices = np.argsort(scores)[::-1]
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Calculate IoU with remaining boxes
            remaining = indices[1:]
            ious = np.array([
                BoundingBoxUtils.calculate_iou(boxes[current], boxes[i])
                for i in remaining
            ])

            # Keep boxes with IoU below threshold
            indices = remaining[ious < iou_threshold]

        return keep


class Visualizer:
    """Visualization utilities for CV results."""

    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]

    @staticmethod
    def draw_boxes(image: np.ndarray, boxes: np.ndarray,
                  labels: Optional[List[str]] = None,
                  scores: Optional[np.ndarray] = None,
                  class_names: Optional[List[str]] = None) -> np.ndarray:
        """Draw bounding boxes on image."""
        img = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            color = Visualizer.COLORS[i % len(Visualizer.COLORS)]

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if labels is not None or scores is not None:
                label_parts = []
                if labels is not None:
                    label_parts.append(str(labels[i]))
                if scores is not None:
                    label_parts.append(f"{scores[i]:.2f}")
                label = " ".join(label_parts)

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img

    @staticmethod
    def draw_segmentation_mask(image: np.ndarray, mask: np.ndarray,
                               alpha: float = 0.5) -> np.ndarray:
        """Overlay segmentation mask on image."""
        colored_mask = np.zeros_like(image)
        for class_id in np.unique(mask):
            if class_id == 0:  # Skip background
                continue
            color = Visualizer.COLORS[class_id % len(Visualizer.COLORS)]
            colored_mask[mask == class_id] = color

        return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


def main():
    """Demo usage of CV utilities."""
    print("Computer Vision Utilities Demo")
    print("=" * 50)

    # Initialize processor
    processor = ImageProcessor(target_size=(640, 640))

    # Example bounding box operations
    boxes = np.array([
        [100, 100, 200, 200],
        [150, 150, 250, 250],
        [400, 400, 500, 500]
    ])
    scores = np.array([0.9, 0.8, 0.7])

    # Convert formats
    xywh = BoundingBoxUtils.xyxy_to_xywh(boxes)
    print(f"XYXY to XYWH conversion:")
    print(f"  Original: {boxes[0]}")
    print(f"  Converted: {xywh[0]}")

    # Calculate IoU
    iou = BoundingBoxUtils.calculate_iou(boxes[0], boxes[1])
    print(f"\nIoU between box 0 and 1: {iou:.4f}")

    # Apply NMS
    keep = BoundingBoxUtils.non_max_suppression(boxes, scores, iou_threshold=0.3)
    print(f"\nNMS kept indices: {keep}")

    print("\n[SUCCESS] CV utilities ready for use!")


if __name__ == '__main__':
    main()
