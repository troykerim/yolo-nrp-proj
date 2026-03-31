from ultralytics import YOLO

model = YOLO("/workspace/yolo-output/yolov11-3-21/train_session2/weights/best.pt")

model.predict(
    source="/workspace/data/videos/moment_sort_upper_2026-01-28T04_30_51.mp4",  
    save=True,
    conf=0.2,       # 0.25 defualt
    device=0,
    project="/workspace/yolo-output",
    name="video_test-3-25"
)