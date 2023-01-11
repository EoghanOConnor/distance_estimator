import cv2
import torch
import pickle
import numpy

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
yolo_model.classes = 0
model = pickle.load(open('svm_regressor.sav','rb'))


font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2

def get_inputs(results):
    try:
        inputs = []
        out = results.xyxy[0].numpy()
        for obj in out:
            if obj[-1] == 0:
                inputs.append(obj[0:4])
        return inputs
    
    except:
        print('Empty')

cap = cv2.VideoCapture(0)
while True:
    _,image = cap.read()
    results = yolo_model(image)
    try:
        yolo_xywh = get_inputs(results)
        xmin , ymin , xmax,ymax = yolo_xywh[0]
        prediction = model.predict(yolo_xywh)
        image = cv2.putText( image,"Prediction %.2f " %prediction, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),2)
    except:
        pass
    cv2.imshow("Distance estimator",image)
    if(cv2.waitKey(1)==ord("q")):
        break

cap.release()
cv2.destroyAllWindows()