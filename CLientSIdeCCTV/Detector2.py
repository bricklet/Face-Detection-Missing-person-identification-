import numpy as np
import cv2
import dlib
import datetime
import time
import logging
import os
import threading
from skimage.metrics import structural_similarity as compare_ssim

class Frame:

	# Constructor: Initialize paths, labels, and DNN model
	def __init__(self, DatasetPath, path=0, label=1, threshold=0.9):
		self.path = path
		self.DatasetPath = DatasetPath
		self.cap = cv2.VideoCapture(self.path)
		self.label = label
		self.threshold = threshold
		self.cache = []
		self.stop_threads = False

		# Load DNN face detector model
		self.net = cv2.dnn.readNetFromCaffe(
			'C:/Users/Janis Reji/Desktop/Automated-CCTV-surveillance-master/CLientSIdeCCTV\deploy.prototxt',  # Path to the Caffe model architecture file
			'C:/Users/Janis Reji/Desktop\Automated-CCTV-surveillance-master/CLientSIdeCCTV/res10_300x300_ssd_iter_140000_fp16.caffemodel'  # Path to the pre-trained model weights
		)

		# Create the directory if it does not exist
		if not os.path.exists(self.DatasetPath):
			os.makedirs(self.DatasetPath)

	def destroy(self):
		self.cap.release()
		cv2.destroyAllWindows()
		self.stop_threads = True

	def preprocess(self, img):
		# Placeholder for additional preprocessing if needed
		return img

	# Save image to disk with error handling
	def save(self, image, mode):
		image = self.preprocess(image)
		image = cv2.resize(image, (224, 224))
		localtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		safe_mode = mode.replace(":", "_")
		title = os.path.join(self.DatasetPath, f"{self.label}_{localtime}_{safe_mode}.jpg")

		if not os.path.exists(self.DatasetPath):
			os.makedirs(self.DatasetPath)

		success = cv2.imwrite(title, image)
		if success:
			print('[Info] Saved:', title)
		else:
			print('[Error] Failed to save image:', title)

	# DNN face detection
	def detect_faces_dnn(self, frame):
		h, w = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
		self.net.setInput(blob)
		detections = self.net.forward()
		faces = []

		for i in range(detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > self.threshold:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(x, y, x1, y1) = box.astype("int")
				faces.append((x, y, x1 - x, y1 - y))
		return faces

	def match(self, roi, face):
		roi = roi.astype(np.uint8)
		face = face.astype(np.uint8)
		res = cv2.matchTemplate(face, roi, cv2.TM_CCOEFF_NORMED)
		locations = np.where(res >= self.threshold)
		if locations is None:
			self.label += 1
			return False
		return len(locations) > 1

	def run(self):
		frame_count = 0
		while True:
			if not self.cap.isOpened():
				self.cap.open(self.path)

			flag, img = self.cap.read()
			if not flag:
				print('[Error] Bad frame detected')
				break

			frame_count += 1
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = self.detect_faces_dnn(img)
			print(f"[Info] Number of faces detected: {len(faces)}")

			# Process and save detected faces
			for (x, y, w, h) in faces:
				cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
				roi = gray[y:y+h, x:x+w]
				if roi is not None:
					self.save(roi, f"{frame_count}_{x}_{y}")

			# Display image with bounding boxes
			cv2.imshow("Video Streaming", img)

			# Check if the window is closed or 'q' key is pressed
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			if cv2.getWindowProperty("Video Streaming", cv2.WND_PROP_VISIBLE) < 1:
				break

		self.destroy()
