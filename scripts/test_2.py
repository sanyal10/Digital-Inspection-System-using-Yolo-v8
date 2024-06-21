from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np


# Load a pretrained YOLOv8n model
model = YOLO('yolov8s.pt')
model2=YOLO(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\project_finalyear\Building facade.v2i.yolov8\runs-20231118T161130Z-001\runs\detect\train\weights\best.pt')


# Run inference on 'bus.jpg'
#results = model(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\yolo\datasets\door.v1i.yolov8\valid\images\door33_jpg.rf.0a25321c5b8ec9e23fbb5886d6eafcac.jpg')  # results list

results=model.predict(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\project_finalyear\Building facade.v2i.yolov8\sample_images\sample_image_4.jpeg.png',save=False, imgsz=640, show_labels=False, conf=0.6,iou=0.6,show_conf=False)

print("boxes_position---->",results[0].boxes.xyxy )
print("boxes_conf---->",results[0].boxes.conf)
print("boxes_classes---->",results[0].boxes.cls)
print("boxes_dimension---->",results[0].boxes.xywh)

#boxes in xywh format
img_box=results[0].boxes.xywh
img_box=img_box.tolist()
print('image boxes \n',img_box) #image boxes


# Show the results

original_image = results[0].plot()  # plot a BGR numpy array of predictions 
im = Image.fromarray(original_image[..., ::-1])  # RGB PIL image
#im.show()  # show image
#im.save('results.jpg')  # save image

results2=model2.predict(original_image,save=False, imgsz=640,show_labels=False, conf=0.5,iou=0.7,show_conf=False)
for r in results2:
    original_image2 = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(original_image2[..., ::-1])  # RGB PIL image
    im.show()  # show image