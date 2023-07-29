import cv2
from ultralytics import YOLO



model = YOLO("yolov8n.pt")
vid = cv2.VideoCapture(0)
names = model.names

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = vid.read()

    results = model.predict(frame, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            predicton = names[int(box.cls.tolist()[0])]
            confidence = box.conf.tolist()[0]
            
            cv2.putText(frame, "{} {:.2f}".format(predicton, confidence), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                      
            


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()