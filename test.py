# Step 2
from ultralytics import YOLO
import cv2
import os



'''
For each time you run this script inside NRP, you need to update the following paths:
1. model = YOLO(../weights/best.pt)
2. input_folder = (..dataset-name/test/images)
3. output_folder =(../output-location)


CHANGE PATHS!!! espicially input and output folders!  Double check YOLO location too!
 from step 1: /workspace/output/yolov11-2nd/train_session2

'''

def main():
    # Path to trained YOLOv11 weights on PVC  (Best model)
    model = YOLO(
        "/workspace/yolo-output/yolov11-3-21/train_session2/weights/best.pt"
        # "/workspace/yolo-output/yolov11/train_session2/weights/best.pt"
        # /workspace/output/yolov11-2nd/train_session2/weights/best.pt"  
    )

    # Folder containing test images
    input_folder = (
        "/workspace/data/jam-causing-material-aug/test/images"
        # "/workspace/data/Jam-Causing-Material.v1i.yolov11-3-18/test/images"
        # "/workspace/data/yolov11-Feb11th-dataset/test/images" 
        # "/workspace/data/jam-causing-material-CURRENT.yolov11/test/images"
    )

    # Folder where predictions will be saved
    output_folder = (
        "/workspace/yolo-predictions/predictions-mar-22"
        # "/workspace/yolo-predictions/predictions-feb-12" 
    )  

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Collect all JPG images from test set
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".jpg")
    ]

    print(f"[INFO] Found {len(image_files)} test images")

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        # Run inference
        results = model(image_path)

        # Draw bounding boxes
        img = results[0].plot()

        # Save prediction image
        output_image_path = os.path.join(
            output_folder, f"predicted_{image_file}"
        )
        cv2.imwrite(output_image_path, img)

        print(f"[INFO] Saved: {output_image_path}")


if __name__ == "__main__":
    main()

