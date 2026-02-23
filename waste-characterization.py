from ultralytics import YOLO
import os
import torch


def print_gpu_stats(device_id):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_id)

        allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        reserved = torch.cuda.memory_reserved(device_id) / 1024**2
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
        free = total - reserved

        print("\n========== GPU MEMORY STATS ==========")
        print(f"GPU Name             : {torch.cuda.get_device_name(device_id)}")
        print(f"GPU Memory Allocated : {allocated:.2f} MiB")
        print(f"GPU Memory Reserved  : {reserved:.2f} MiB")
        print(f"GPU Memory Free Est. : {free:.2f} MiB")
        print(f"GPU Total Memory     : {total:.2f} MiB")
        print("=======================================\n")
    else:
        print("\n[INFO] CUDA not available.\n")


def main():
    OUTPUT_ROOT = "/workspace/output/yolov11-3rd"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    model = YOLO("/workspace/models/yolo/yolo11m.pt")

    DEVICE_ID = 0
    print(f"[INFO] Training on GPU device {DEVICE_ID}")

    train_results = model.train(
        data="/workspace/data/jam-material-YOLO/data.yaml",
        workers=2,                  # 1, 2, 8; NRP complained about 8 last time
        batch=8,
        epochs=300,
        optimizer="AdamW",
        imgsz=640,
        weight_decay=0.001,
        lr0=0.01,
        lrf=0.001,
        iou=0.5,
        mosaic=0.7,
        patience=0,                # 40, = 0 means no early stopping
        fliplr=0.5,
        flipud=0.5,
        bgr=0.3,
        device=DEVICE_ID,
        project=OUTPUT_ROOT,
        name="train_session"
    )

    # GPU stats print after training is done
    print_gpu_stats(DEVICE_ID)


if __name__ == "__main__":
    main()
