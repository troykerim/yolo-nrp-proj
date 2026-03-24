from ultralytics import YOLO
import os
import json
import torch
import csv


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


def save_text_table(path, title, rows):
    with open(path, "w") as f:
        f.write(title + "\n\n")
        for row in rows:
            f.write(str(row) + "\n")


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_ROOT = "/workspace/yolo-output/yolov11-3-24"
    RUN_NAME = "train_session"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    model = YOLO("/workspace/models/yolo/yolo11m.pt")

    DEVICE_ID = 0
    print(f"[INFO] Training on GPU device {DEVICE_ID}")

    model.train(
        data="/workspace/data/jam-causing-material-aug/data.yaml",
        workers=2,
        batch=8,
        epochs=400,
        optimizer="AdamW",
        imgsz=640,
        weight_decay=0.001,
        lr0=0.005, # 0.005
        lrf=0.01,
        iou=0.5,
        mosaic=0.5,     # Was 0.3
        patience=40,
        fliplr=0.5,
        flipud=0.5,
        bgr=0.1,        # Was 0.3
        device=DEVICE_ID,
        project=OUTPUT_ROOT,
        name=RUN_NAME
    )

    run_dir = os.path.join(OUTPUT_ROOT, RUN_NAME)
    best_model_path = os.path.join(run_dir, "weights", "best.pt")

    print(f"[INFO] Loading best model from: {best_model_path}")
    best_model = YOLO(best_model_path)

    print("[INFO] Running validation on best.pt...")
    val_results = best_model.val(
        data="/workspace/data/jam-causing-material-aug/data.yaml",
        imgsz=640,
        batch=8,
        device=DEVICE_ID,
        project=OUTPUT_ROOT,
        name=f"{RUN_NAME}_val",
        verbose=True
    )

    metrics_dir = os.path.join(run_dir, "saved_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Overall metrics dictionary
    overall_metrics = val_results.results_dict

    with open(os.path.join(metrics_dir, "overall_metrics.json"), "w") as f:
        json.dump(overall_metrics, f, indent=4)

    with open(os.path.join(metrics_dir, "overall_metrics.txt"), "w") as f:
        for k, v in overall_metrics.items():
            f.write(f"{k}: {v}\n")

    # Per-class summary table
    per_class_summary = val_results.summary()

    with open(os.path.join(metrics_dir, "per_class_metrics.json"), "w") as f:
        json.dump(per_class_summary, f, indent=4)

    save_csv(os.path.join(metrics_dir, "per_class_metrics.csv"), per_class_summary)
    save_text_table(
        os.path.join(metrics_dir, "per_class_metrics.txt"),
        "Per-class YOLO validation metrics",
        per_class_summary
    )

    print(f"[INFO] Saved overall metrics to: {os.path.join(metrics_dir, 'overall_metrics.json')}")
    print(f"[INFO] Saved per-class metrics to: {os.path.join(metrics_dir, 'per_class_metrics.csv')}")

    print_gpu_stats(DEVICE_ID)


if __name__ == "__main__":
    main()