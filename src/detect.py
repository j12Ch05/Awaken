from ultralytics import YOLO

# Load a pretrained YOLO11 nano model (downloads automatically)
model = YOLO("./models/yolo11s.pt")

# Run inference on an image
results = model("./data/raw/")

# print(results)
for r in results:
    print(f"detect {len(r)} objects in this image")