from ultralytics import YOLO
import os


'''
For each time you run this script inside Nautilus, update these paths:
1. model_path = "/workspace/.../weights/best.pt"
2. input_folder = "/workspace/data/.../test/images"
3. output_folder = "/workspace/.../prediction-output"

This script:
- Uses the trained YOLO model
- Runs inference on test images
- Saves only .txt files
- Each .txt file contains:
    Label-Name xmin xmax ymin ymax

Example output line:
Plastic-G 993 1059 805 896
'''

def main():
    # Path to trained YOLO weights
    model_path = "/workspace/yolo-output/yolov11-4-11/train_session/weights/best.pt"

    # Folder containing test images
    input_folder = "/workspace/data/yolo-dataset-4-11/test/images"

    # Main output folder
    output_folder = "/workspace/yolo-predictions/pascal-voc-predictions-4-11"

    # Folder where all .txt prediction files will be stored
    txt_output_folder = os.path.join(output_folder, "labels")

    os.makedirs(txt_output_folder, exist_ok=True)

    # Load model
    model = YOLO(model_path)

    # Collect image files
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith(valid_extensions)
    )

    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Output folder: {output_folder}")
    print(f"[INFO] TXT prediction folder: {txt_output_folder}")
    print(f"[INFO] Found {len(image_files)} test images")

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        # Run inference
        results = model(image_path, verbose=False)
        result = results[0]

        # Create matching .txt filename
        base_name = os.path.splitext(image_file)[0]
        txt_output_path = os.path.join(txt_output_folder, f"{base_name}.txt")

        with open(txt_output_path, "w") as f:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, class_id in zip(boxes_xyxy, class_ids):
                    xmin, ymin, xmax, ymax = box

                    # Convert to integers
                    xmin = int(round(xmin))
                    xmax = int(round(xmax))
                    ymin = int(round(ymin))
                    ymax = int(round(ymax))

                    # Get class label from model names
                    label_name = result.names[class_id]

                    # Save in requested format:
                    # Label-Name xmin xmax ymin ymax
                    f.write(f"{label_name} {xmin} {xmax} {ymin} {ymax}\n")

        print(f"[INFO] Saved: {txt_output_path}")

    print("\n[INFO] Done.")
    print(f"[INFO] All prediction .txt files are located here:")
    print(f"       {txt_output_folder}")


if __name__ == "__main__":
    main()