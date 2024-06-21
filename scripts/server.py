import streamlit as st
import pandas as pd
from io import StringIO
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

st.title('Welcome to Digital Inspection System')

uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is not None:
    print(type(uploaded_file))
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    print(img_array)

    st.image(uploaded_file, caption='Given Image')

    model = YOLO(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\project_finalyear\Building facade.v2i.yolov8\runs-20231118T161130Z-001\runs\detect\train\weights\best.pt')
    results=model.predict(img_array,save=False, imgsz=640, show_labels=True, conf=0.5,iou=0.75,show_conf=True)
    img_box=results[0].boxes.xywh
    img_box=img_box.tolist()
    print('image boxes',img_box) #image boxes

    # Show the results
    for r in results:
        original_image = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(original_image[..., ::-1])  # RGB PIL image
        #im.show()  # show image
        #im.save('results.jpg')  # save image
        st.image(original_image,caption="Predicted")