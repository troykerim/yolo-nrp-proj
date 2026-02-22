from ultralytics import YOLO
import os
import torch


def main():
    OUTPUT_ROOT = "/workspace/output/yolov11-3rd"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Load model
    model = YOLO("/workspace/models/yolo/yolo11m.pt")

    # Force specific GPU device 
    DEVICE_ID = 0  # change if needed
    print(f"[INFO] Forcing training on GPU device {DEVICE_ID}")

    if torch.cuda.is_available():
        print(f"[INFO] CUDA is available")
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(DEVICE_ID)}")
    else:
        print("[WARNING] CUDA not available. Training will use CPU.")

    # Print GPU memory usage before training 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(DEVICE_ID) / 1024**2
        reserved = torch.cuda.memory_reserved(DEVICE_ID) / 1024**2
        total = torch.cuda.get_device_properties(DEVICE_ID).total_memory / 1024**2

        print(f"[INFO] GPU Memory Allocated: {allocated:.2f} MiB")
        print(f"[INFO] GPU Memory Reserved : {reserved:.2f} MiB")
        print(f"[INFO] GPU Total Memory    : {total:.2f} MiB")

    # Train Configuration
    train_results = model.train(
        data="/workspace/data/jam-material-YOLO/data.yaml",
        workers=2,
        batch=8,
        epochs=300,
        optimizer="AdamW",
        imgsz=640,
        weight_decay=0.001,
        lr0=0.01,
        lrf=0.001,
        iou=0.5,
        mosaic=0.7,
        patience=0,

        fliplr=0.5,
        flipud=0.5,
        bgr=0.3,

        device=DEVICE_ID,   # FORCE GPU
        project=OUTPUT_ROOT,
        name="train_session2"
    )


if __name__ == "__main__":
    main()
