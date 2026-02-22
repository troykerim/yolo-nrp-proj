# Step 1
from ultralytics import YOLO
import os


def main():
    OUTPUT_ROOT = "/workspace/output/yolov11-3rd"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    model = YOLO("/workspace/models/yolo/yolo11m.pt")

    print(f"Using device: {model.device}")

    train_results = model.train(
        # data="/workspace/data/yolov11-Feb11th-dataset/data.yaml", jam-material-YOLO
        data="/workspace/data/jam-material-YOLO/data.yaml",
        workers=2,   # if an error occurs, change to either 1 or 2
        batch=8,
        epochs=300,
        optimizer="AdamW",
        imgsz=640,
        weight_decay=0.001,
        lr0=0.01,
        lrf=0.001,
        iou=0.5,
        mosaic=0.7,
        patience=40,     # patience=0 to disable EarlyStopping

        fliplr=0.5,   # horizontal flip probability
        flipud=0.5,   # vertical flip probability
        bgr=0.3,      # BGR channel swap probability

        project=OUTPUT_ROOT,
        name="train_session"
    )


if __name__ == "__main__":
    main()

# Should save to: /workspace/output/yolov11/train_session...


# 2-12 (NIGHT) was saved here: /workspace/output/yolov11-2nd/train_session2
