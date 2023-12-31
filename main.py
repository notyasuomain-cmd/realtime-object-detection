from dotenv import dotenv_values
import cv2
from ultralytics import YOLO


# load model
model = YOLO("yolov8n.pt")
config = dotenv_values(".env")


# webcam
vid = cv2.VideoCapture(f"rtsp://{config['RTSP_USER']}:{config['RTSP_PASSWD']}@{config['CAM_IP']}:{config['RTSP_PORT']}/Streaming/channels/101")
# class names for prediction
NAMES = model.names
print(type(NAMES))

# set resulution
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
# vid.set(cv2.CAP_PROP_FPS, 10)

fps = int(vid.get(5))
print("fps:", fps)

while True:
    # get frame
    ret, frame = vid.read()

    if not ret:
        print("Camera is disconnected")
        vid.release()
        break

    # make prediction with YOLOV8
    results = model.predict(frame)  # verbose=False to hide log

    # for each result draw the boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # box coordinates
            x1, y1, x2, y2 = box.xyxy.tolist()[0]

            predicton = NAMES.get(int(box.cls.tolist()[0]))
            confidence = box.conf.tolist()[0]

            # if confidence is greater than 75% and prediction in our classes
            #  then draw boxes and put text
            
            #classes = ["person", "bottle", "remote"]
            #if int(confidence*100) >= 75 and predicton in classes:
             
            # draw boxes
            cv2.rectangle(frame, (int(x1), int(y1)),
                            (int(x2), int(y2)), (0, 0, 255), 2)
            # puttext with class name and confidence
            cv2.putText(frame, f"{predicton} {confidence:.2f}", (int(
                x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # show frame
    cv2.imshow('frame', frame)
    # if q is pressed break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# release video and destroy all windows
vid.release()
cv2.destroyAllWindows()
