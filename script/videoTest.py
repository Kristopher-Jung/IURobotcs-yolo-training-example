import numpy as np
import imutils
import time
import cv2
labelsPath = '../obj.names'
weightsPath = '../backup/yolo-obj_1000.weights'
configPath = '../yolo-obj.cfg'
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
vs = cv2.VideoCapture('../jianshi_buoy.MP4')
writer = None
(W, H) = (None, None)
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("{} frames in video".format(total))

except:
	total = -1
while True:
	(grabbed, curr_frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = curr_frame.shape[:2]
	blob = cv2.dnn.blobFromImage(curr_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	boxes = []
	confidences = []
	classIDs = []
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > 0.5:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(curr_frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(curr_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter('test.mp4', fourcc, 30, (curr_frame.shape[1], curr_frame.shape[0]), True)
		if total > 0:
			elap = (end - start)
			print("single frame took {:.4f} seconds".format(elap))
			print("estimated total time to finish: {:.4f}".format(elap * total))
	writer.write(curr_frame)
writer.release()
vs.release()