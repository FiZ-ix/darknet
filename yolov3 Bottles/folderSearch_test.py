import glob
import cv2
import numpy as np

count = 1
good = 0
bad = 0
inputNumber = 0

for inputCounter in glob.iglob('dataProcess\input\*.jpg', recursive=True):
    inputNumber += 1

while count <= inputNumber:
    for filepath in glob.iglob('dataProcess\input\*.jpg', recursive=True):

        def get_output_layers(net):
            layer_names = net.getLayerNames()

            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            return output_layers


        def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
            #label = str(classes[class_id])
            label = str(classes)

            color = COLORS[class_id]

            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

            cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        image = cv2.imread(filepath)

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        #with open(classes, 'r') as f:
        #    classes = [line.strip() for line in f.readlines()]
        classes = 'bottle'

        COLORS = [255,255,255]

        net = cv2.dnn.readNet('yolov3-obj_last.weights', 'yolov3-obj.cfg')

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.01
        nms_threshold = 0.10

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.01:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        title = str(len(indices))

        cv2.imshow(title, image)
        #cv2.waitKey(0) #use this line to manually flip through images
        filename ='output'+ str(count) + '.jpg'

        count += 1
        cv2.destroyAllWindows()

        print(len(indices))
        if len(indices) == 12:
            good = good + 1
            path = str(r'dataProcess\output\good/' + filename)
            cv2.imwrite(path, image)
        else:
            bad += 1
            path = str(r'dataProcess\output\bad/' + filename)
            cv2.imwrite(path, image)

