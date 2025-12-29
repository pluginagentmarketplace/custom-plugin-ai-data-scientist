# Computer Vision Tasks Guide

## Task Selection Decision Tree

```
What is your CV task?
│
├─► Image Classification
│   └─► "What object is in this image?"
│       • Single label per image
│       • Models: ResNet, EfficientNet, ViT
│
├─► Object Detection
│   └─► "What objects are where in this image?"
│       • Bounding boxes + labels
│       • Models: YOLO, Faster R-CNN, DETR
│
├─► Semantic Segmentation
│   └─► "What class is each pixel?"
│       • Pixel-level classification
│       • Models: U-Net, DeepLab, FCN
│
├─► Instance Segmentation
│   └─► "Which pixels belong to which object instance?"
│       • Separate masks per object
│       • Models: Mask R-CNN, YOLACT
│
├─► Pose Estimation
│   └─► "Where are the body keypoints?"
│       • Joint/keypoint locations
│       • Models: OpenPose, HRNet, MediaPipe
│
├─► Face Recognition
│   └─► "Whose face is this?"
│       • Face detection + embedding
│       • Models: ArcFace, FaceNet, InsightFace
│
└─► Image Generation
    └─► "Generate new images"
        • Text-to-image, style transfer
        • Models: Stable Diffusion, DALL-E, GAN
```

## Model Selection Matrix

| Task | Real-time | High Accuracy | Edge Device |
|------|-----------|---------------|-------------|
| Classification | MobileNet | EfficientNet-B7 | MobileNetV3 |
| Detection | YOLOv8n | Faster R-CNN | YOLOv8n-INT8 |
| Segmentation | BiSeNet | DeepLabV3+ | ENet |
| Pose | MoveNet | HRNet-W48 | MoveNet Lightning |

## Metrics by Task

### Classification
```
Accuracy = Correct / Total
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Object Detection
```
mAP@0.5 = Mean AP at IoU threshold 0.5
mAP@0.5:0.95 = Mean AP averaged over IoU 0.5-0.95
AP (Area) = AP for small/medium/large objects
```

### Segmentation
```
IoU (Jaccard) = Intersection / Union
Dice = 2 × Intersection / (Pred + Truth)
Pixel Accuracy = Correct Pixels / Total Pixels
```

## Data Augmentation by Task

| Augmentation | Classification | Detection | Segmentation |
|--------------|----------------|-----------|--------------|
| Flip H/V | ✅ | ✅ (adjust boxes) | ✅ (adjust mask) |
| Rotation | ✅ | ⚠️ (complex) | ⚠️ (complex) |
| Color Jitter | ✅ | ✅ | ✅ |
| Mosaic | ❌ | ✅ | ❌ |
| MixUp | ✅ | ✅ | ⚠️ |
| CutOut | ✅ | ⚠️ | ❌ |
| Elastic | ⚠️ | ⚠️ | ✅ (medical) |

## Common Datasets

| Dataset | Task | Classes | Size |
|---------|------|---------|------|
| ImageNet | Classification | 1000 | 14M |
| COCO | Detection/Seg | 80 | 330K |
| Pascal VOC | Detection/Seg | 20 | 11K |
| ADE20K | Segmentation | 150 | 25K |
| MPII | Pose | 16 joints | 25K |
| LFW | Face | 5749 people | 13K |

## Preprocessing Pipeline

```python
# Standard CV preprocessing pipeline
def preprocess_pipeline(image, task='classification'):
    # 1. Load and decode
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Resize (task-specific)
    if task == 'classification':
        img = cv2.resize(img, (224, 224))  # Square resize
    elif task == 'detection':
        img = letterbox_resize(img, (640, 640))  # Maintain aspect

    # 3. Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # 4. To tensor format [C, H, W]
    img = np.transpose(img, (2, 0, 1))

    return img
```

## Deployment Considerations

| Factor | Consideration |
|--------|---------------|
| Latency | Use lightweight models (MobileNet, YOLOv8n) |
| Memory | Quantization (INT8), pruning |
| Accuracy | Larger models, ensemble |
| Edge | TensorRT, ONNX, CoreML |
| Cloud | GPU inference, batching |

## References

- [COCO Dataset](https://cocodataset.org/)
- [Papers With Code - CV](https://paperswithcode.com/area/computer-vision)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
