from ultralytics import YOLO

model = YOLO("/workspace/yolo-output/yolov11-3-21/train_session2/weights/best.pt")

model.predict(
    source = "/workspace/data/videos/moment_sort_upper_2026-01-29T15_04_46.mp4", # source="/workspace/data/videos/moment_sort_upper_2026-01-28T04_30_51.mp4",   # Upload a new video first
    save=True,
    conf=0.2,       # Confidence value: 0.25 initially
    device=0,
    project="/workspace/yolo-output",
    name="video_test-3-31"
)
