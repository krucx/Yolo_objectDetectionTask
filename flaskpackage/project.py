from flask import url_for
from flaskpackage import app
import cv2
import os
import numpy as np

def gen_bounding_boxes(model,objects,size=(416,416),threshold=0.5):

    layer_names = model.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(objects), 3))

    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg'))
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1.0/255.0, size, (0, 0, 0), True, crop=False)

    model.setInput(blob)
    outputs = model.forward(output_layers)

    predicted_class_ids = []
    confidences = []
    bounding_boxes = []
    for i in outputs:
        for grid_cell_prediction in i:
            class_id = np.argmax(grid_cell_prediction[5:])
            confidence = grid_cell_prediction[5+class_id]

            if confidence >= threshold:
                center_x = int(grid_cell_prediction[0] * width)
                center_y = int(grid_cell_prediction[1] * height)
                w = int(grid_cell_prediction[2] * width)
                h = int(grid_cell_prediction[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bounding_boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                predicted_class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)
    nms_bounding_boxes = [bounding_boxes[i] for i in idxs]
    nms_predicted_classes = [predicted_class_ids[i] for i in idxs]
    nms_confidence = [confidences[i] for i in idxs]

    resultantant_text = []

    for i in range(len(nms_bounding_boxes)):
        x, y, w, h = nms_bounding_boxes[i]
        label = f"{i+1}-{objects[nms_predicted_classes[i]]}"
        resultantant_text.append(f"Bounding box no. {i+1}  ----  {objects[nms_predicted_classes[i]]}({round(nms_confidence[i],3)})")
        color = colors[nms_predicted_classes[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    img = cv2.resize(img, (512, 512))
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "answer.jpg"), img)
    return resultantant_text



