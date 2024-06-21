from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2
from PIL import Image

#modle prediction

model = YOLO(r"C:\Users\DEBAJYOTI\OneDrive\Desktop\project_finalyear\Building facade.v2i.yolov8\runs-20231118T161130Z-001\runs\detect\train\weights\best.pt")   # YOLOv8 custom/pretrained model
#print(model.names)
im0 = cv2.imread(r"C:\Users\DEBAJYOTI\OneDrive\Desktop\project_finalyear\Building facade.v2i.yolov8\sample_images\sample_image_15.jpeg")  # path to image file
h, w = im0.shape[:2]  # image height and width

# Heatmap Init
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                     imw=w,
                     imh=h,
                     view_img=True,
                     shape="circle")

results = model.track(im0, persist=True,conf=0.2,iou=0.5)
im0 = heatmap_obj.generate_heatmap(im0, tracks=results)
im = Image.fromarray(im0[..., ::-1])
im.show()