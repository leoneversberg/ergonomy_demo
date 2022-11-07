#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class Ergonomy:
	def __init__(self):
		self.trunk_angle=0

	def update_joints(self, landmarks_3d):
		"""update all needed joints based on landmarks_3d.landmark from mp"""
		try:
			# media pipe joints (BlazePose GHUM 3D)
			left_shoulder = np.array([landmarks_3d.landmark[11].x, landmarks_3d.landmark[11].y, landmarks_3d.landmark[11].z])
			right_shoulder = np.array([landmarks_3d.landmark[12].x, landmarks_3d.landmark[12].y, landmarks_3d.landmark[12].z])
			left_hip = np.array([landmarks_3d.landmark[23].x, landmarks_3d.landmark[23].y, landmarks_3d.landmark[23].z])
			right_hip = np.array([landmarks_3d.landmark[24].x, landmarks_3d.landmark[24].y, landmarks_3d.landmark[24].z])
			left_knee = np.array([landmarks_3d.landmark[25].x, landmarks_3d.landmark[25].y, landmarks_3d.landmark[25].z])
			right_knee = np.array([landmarks_3d.landmark[26].x, landmarks_3d.landmark[26].y, landmarks_3d.landmark[26].z])
			
			# helper joints
			mid_shoulder = (left_shoulder + right_shoulder) / 2
			mid_hip = (left_hip + right_hip) / 2
			mid_knee = (left_knee + right_knee) / 2

			# angles
			self.trunk_angle = self.get_angle(mid_knee, mid_hip, mid_shoulder, mid_hip, adjust=True)

		except:
			# could not retrieve all needed joints
			pass

	def get_angle(self, a, b, c, d, adjust):
		"""return the angle between two vectors"""
		vec1 = a - b
		vec2 = c - d

		cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
		angle = np.arccos(cosine_angle)

		# adjust by substracting 180 deg if needed 
		# this lets the angle start at 0 instead of 180
		if (adjust):
			angle_adjusted = abs(np.degrees(angle) - 180)
			return int(angle_adjusted)
		else:
			return int(abs(np.degrees(angle)))

	def get_trunk_color(self):
		"""returns (B,G,R) colors for visualization"""
		if self.trunk_angle < 20:
			return (0,255,0)
		elif self.trunk_angle <= 60:
			return (0,255,255)
		else:
			return (0,0,255)


if __name__ == '__main__':
	MyErgonomy = Ergonomy()
	cap = cv2.VideoCapture(0)  # webcam input
	with mp_pose.Pose(
		model_complexity=1,
		smooth_landmarks=True,
		min_detection_confidence=0.3,
		min_tracking_confidence=0.3) as pose:
		while cap.isOpened():
			success, image = cap.read()
			if not success:
				print("Ignoring empty camera frame.")
				# If loading a video, use 'break' instead of 'continue'.
				continue

			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			results = pose.process(image)
			landmarks_3d = results.pose_world_landmarks

			# Draw the pose annotation on the image.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			mp_drawing.draw_landmarks(
				image,
				results.pose_landmarks,
				mp_pose.POSE_CONNECTIONS,
				landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


			MyErgonomy.update_joints(landmarks_3d)

			# visualization: text + HP bar
			image = cv2.putText(image, text="trunk angle: "+str(MyErgonomy.trunk_angle), 
				org=(5,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=MyErgonomy.get_trunk_color(), thickness=3)
			image = cv2.rectangle(image, (5,5), (145*2, 30), color=(255,255,255), thickness=-1)
			image = cv2.rectangle(image, (5,5), (145*2-(MyErgonomy.trunk_angle * 2), 30), color=MyErgonomy.get_trunk_color(), thickness=-1)
		
			cv2.imshow('MediaPipe Pose Demo', image)

			if cv2.waitKey(5) & 0xFF == 27:
				break
	cap.release()