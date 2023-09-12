from ultralytics import YOLO

# Load a model
model = YOLO('TrainModelMalva.pt')  

# Predict using the model
results = model.predict(source=0, show=True, save=True, conf=0.5, save_txt=False, save_crop=False)

