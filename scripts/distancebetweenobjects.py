#finding distance between two given objects

from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\project_finalyear\Building facade.v2i.yolov8\runs-20231118T161130Z-001\runs\detect\train\weights\best.pt')
results=model.predict(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\project_finalyear\Building facade.v2i.yolov8\sample_images\sample_image_14.jpeg',save=False, imgsz=640, show_labels=True, conf=0.3,iou=0.7,show_conf=True)

print("classnames",results[0].names)
'''
print(results[0].boxes.xyxy )
print(results[0].boxes.conf)
print(results[0].boxes.cls)
print(results[0].boxes.xywh)
'''
img_box=results[0].boxes.xywh
img_box=img_box.tolist()
img_box=[[int(x) for x in y] for y in img_box]
#print('image boxes',img_box) #image boxes


# Show the results
for r in results:
    original_image = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(original_image[..., ::-1])  # RGB PIL image
    im.show()  # show image
    #im.save('results.jpg')  # save image


dict={}

#original_image = cv2.imread(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\object det\runs\detect\predict\240_A_jpg.rf.5f4981d12955b58c264b92c23c83e2d3.jpg')

cv2.namedWindow('image',cv2.WINDOW_NORMAL)

for i in range(len(img_box)):
    dict[i+1]=img_box[i]

    c_x=int(img_box[i][0])
    c_y=int(img_box[i][1])

    cv2.putText(original_image,str(i+1),(c_x-5,c_y-5),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),8,cv2.LINE_AA) 

im = Image.fromarray(original_image[..., ::-1])  # RGB PIL image
im.show()  # show image

print ("class objects",dict) 

user_input1=int(input("Enter object 1:  "))
user_input2=int(input("Enter object 2:  "))
x1,y1=dict[user_input1][0],dict[user_input1][1]
x2,y2=dict[user_input2][0],dict[user_input2][1]
print(x1,y1,x2,y2)

point1 = np.array((x1,y1))
point2 = np.array((x2,y2))

dist = np.linalg.norm(point1 - point2)
dist=int(dist)

print("distance",dist)

cv2.line(original_image,(x1,y1),(x2,y2), (36,255,12), 2)
cv2.putText(original_image,"Distance "+str(dist),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(200,0,0),2,cv2.LINE_AA) 

while(1):
    cv2.imshow('image',original_image)
    if cv2.waitKey(20) & 0xFF == 27:
        break
