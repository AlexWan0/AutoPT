import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
	# Windows Import
	if platform == "win32":
		# Change these variables to point to the correct folder (Release/x64 etc.)
		sys.path.append(dir_path + '/openpose/Release');
		os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose/x64/Release;' +  dir_path + '/openpose/bin;'
		import pyopenpose as op
	else:
		# Change these variables to point to the correct folder (Release/x64 etc.)
		sys.path.append('../../python');
		# If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
		# sys.path.append('/usr/local/python')
		from openpose import pyopenpose as op
except ImportError as e:
	print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
	raise e

class PoseAPI():
	def __init__(self, net_resolution="320x160"):
		params = dict()
		params["model_folder"] = "openpose/models/"
		params["net_resolution"] = net_resolution

		self.opWrapper = op.WrapperPython()
		self.opWrapper.configure(params)
		self.opWrapper.start()

	def detect(self, img, bodyOnly=True):
		datum = op.Datum()

		datum.cvInputData = img
		self.opWrapper.emplaceAndPop([datum])

		if len(datum.poseKeypoints.shape) == 0:
			return None, datum.cvOutputData

		if bodyOnly:
			return datum.poseKeypoints[0, :15, :2], datum.cvOutputData

		return datum.poseKeypoints, datum.cvOutputData

	def get_poses_from_video(self, vid_fp, extracted_dir=None, vid_size=(720, 480)):
		cap = cv2.VideoCapture(vid_fp)

		all_poses = []
		
		out = None

		if not extracted_dir == None:
			new_path = os.path.join(extracted_dir, os.path.split(vid_fp)[-1])
			out = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, vid_size)

		prev_pose = None

		i = 0

		key_idx = []

		sum_diff = 0

		while cap.isOpened():
			ret, frame = cap.read()

			if not isinstance(frame, np.ndarray):
				break

			pose, e_img = self.detect(frame, bodyOnly=True)

			#print(pose)

			if isinstance(pose, np.ndarray):
				all_poses.append(pose.tolist())

				if isinstance(prev_pose, np.ndarray):
					sim = cosine_similarity(np.expand_dims(prev_pose.flatten(), axis=0), np.expand_dims(pose.flatten(), axis=0))[0]
					sim = float(sim)
					#print(sim)

					sum_diff += np.sum(np.abs(prev_pose - pose).flatten())

					if sim < 0.8 and not (not len(key_idx) == 0 and i == (key_idx[-1] + 1)):
						key_idx.append(i)
						#print('FOUND KEY IDX')

				elif not (not len(key_idx) == 0 and i == (key_idx[-1] + 1)): # prev is none, but current is not
					key_idx.append(i)
					#print('FOUND KEY IDX')
			else:
				all_poses.append(None)
				#print('none found')

			prev_pose = pose

			if not extracted_dir == None:
				out.write(cv2.resize(e_img, vid_size))

			i += 1
		
		if not extracted_dir == None:
			out.release()

		cap.release()

		print(sum_diff)

		return all_poses, key_idx, sum_diff