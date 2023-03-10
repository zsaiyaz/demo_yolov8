from ultralytics import YOLO
from PIL import Image
# import cv2

# Load a model
model = YOLO("best.pt")  # load a custom model

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
results = model.predict(source="data", show=False, save=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
# im1 = Image.open("data/test1.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images


# from ndarray
# im2 = cv2.imread("data/test2.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# Predict with the model
# results = model("test1.jpg", show=True, save=True) 

# from list of PIL/ndarray
# results = model.predict(source=[im1, im2], show=True, save=True)