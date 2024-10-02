from flask import Flask, Response
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#outdated
# layer_names = net.getLayerNames()    
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = net.getUnconnectedOutLayersNames()


app = Flask(__name__)

# Initialize camera capture
camera_indices = [0, 1]  # Update as needed
caps = [cv2.VideoCapture(idx) for idx in camera_indices]

def generate_frames(camera_index):
    while True:
        success, frame = caps[camera_index].read()
        if not success:
            break
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        # Convert to bytes
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generatePersonDetectionFrames(camera_index):

    while True:
        #Read a frame from the cideo feed
        ret, frame = caps[camera_index].read()
        if not ret:
            break
        
        #prepare the frame for the YOLO model
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0,), True, crop=False)
        net.setInput(blob)

        #get predictions
        outputs = net.forward(output_layers)

        # Initialize lists for detected boxes, confidence, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        #Loop over each  output layer
        for output in outputs:
            for detection in output:
                scores = detection[5:] #get class scores
                class_id = np.argmax(scores) #get the class with the highest score
                confidence = scores[class_id] # get the confidence

                # only consider persons (class_id = 0)
                if confidence > 0.5 and class_id == 0:
                    #print("person detected")
                    cx = int(detection[0] * width) #center in x
                    cy = int(detection[1] * height) # center in y direction
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    #rectangle coordinates
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)

                    #save the box coordinates and confidence
                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maxima suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


        #Draw bounding boxes on the frame
        for i in indices:
            #i = i[0]
            box = boxes[i]
            x, y, w, h = box
            cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{classes[class_ids[i]]}: {confidences[i]:.2f}", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # # Display the resulting frame
        # cv2.imshow('Body Detection with YOLO', frame)

        # # Break the loop on 'q' key press
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     # Release the camera and close all OpenCV windows
        #     caps[camera_index].release()
        #     cv2.destroyAllWindows()
        #     break


        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        # Convert to bytes
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    return Response(generate_frames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ai_feed/<int:camera_index>')
def ai_feed(camera_index):
    return Response(generatePersonDetectionFrames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
