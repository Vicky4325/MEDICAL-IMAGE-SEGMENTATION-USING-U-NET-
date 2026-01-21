%pip install opencv-python
#import necessary packages and load the model
import cv2
import numpy as np
#Load the model
net = cv2.dnn.readNetFromTensorflow(
'dnn/frozen_inference_graph_coco.pb',
'dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
)
#Store Coco names in a list
classesFile = "coco.names"
with open(classesFile, "r") as f:
classNames = f.read().strip().split("\n")
print("Loaded class names:", classNames)
#Load image
img = cv2.imread("dnn\dog.jpg")
if img is None:
raise FileNotFoundError("Image file not found")
height, width, _ = img.shape
#Create a label mask
blank_mask = np.zeros((height, width, 3), dtype=np.uint8)
#Create blob from the image
blob = cv2.dnn.blobFromImage(img, swapRB=True)
#Set input to the network
net.setInput(blob)
#Get outputs
boxes, masks = net.forward(["detection_out", "detection_masks"])
detection_count = boxes.shape[2]
print(f"Detections found: {detection_count}")
for i in range(detection_count):
box = boxes[0, 0, i]
class_id = int(box[1])
score = box[2]
if score < 0.6:
continue
class_name = classNames[class_id]
x = int(box[3] * width)
y = int(box[4] * height)
x2 = int(box[5] * width)
y2 = int(box[6] * height)
roi = blank_mask[y:y2, x:x2]
roi_height, roi_width, _ = roi.shape
#Get mask and resize
mask = masks[i, class_id]
mask = cv2.resize(mask, (roi_width, roi_height))
_, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
#Find countours
contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
color = np.random.randint(0, 255, (3,), dtype="uint8").tolist()
for cnt in contours:
 cv2.fillPoly(roi, [cnt], color)
#Draw bounding box and label
cv2.rectangle(img, (x, y), (x2, y2), color, 2)
cv2.putText(img, f"{class_name} {score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,
255), 1)
#Blend original image with mask
mask_img = cv2.addWeighted(img, 1, blank_mask, 0.8, 0)
#Display images
cv2.imshow("Black Mask", blank_mask)
cv2.imshow("Final Output", mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
