from flask import Flask, render_template, request, jsonify, session
import json
from flask_socketio import SocketIO
from flask_socketio import send, emit
import cv2
import numpy as np
from time import sleep
from threading import Thread, Event
from random import random
from pose_api import PoseAPI
from sklearn.metrics.pairwise import cosine_similarity
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
from web_predictor import get_label

mappings = {0:"Nose",
1:"Neck",
2:"Right Shoulder",
3:"Right Elbow",
4:"Right Wrist",
5:"Left Shoulder",
6:"Left Elbow",
7:"Left Wrist",
8:"Left Hip",
9:"Right Hip",
10:"Right Knee",
11:"Right Ankle",
12:"Left Hip",
13:"Left Knee",
14:"Left Ankle"}

pa = PoseAPI(net_resolution='160x160')

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)
app.secret_key = b'[PASSWORD]'

def load_db(db_fn='db.json'):
	with open(db_fn, 'r') as file_in:
		db = json.load(file_in)
	return db

def add_difficulty():
	for vid in db:
		difficulty = int(min(vid['sum_diff'], 100000)/10000)
		vid['difficulty'] = difficulty

@app.route('/')
def index():
	return render_template('index.html', videos=db[:100])

@app.route('/food')
def food():
	return render_template('food.html')

@app.route('/exercise')
def exercise():
	global idx_split
	global key_idx

	idx_split = [int(idx) for idx in request.args.get('idx').split(',')]
	#key_idx = [a for a in request.args.get('key').split(',')]

	db_new = []
	for idx in idx_split:
		db_new.append(db[idx])

	return render_template('exercise.html', videos=db_new)

thread = Thread()
thread_stop_event = Event()
class VidThread(Thread):
	def __init__(self):
		super(VidThread, self).__init__()

	def action(self):
		global idx_split
		global key_idx

		webcam_cap = cv2.VideoCapture(0)

		#while not thread_stop_event.isSet():
		for i_all, idx in enumerate(idx_split):
			sleep(5)

			#print(idx)

			raw_fp = db[idx]['file_path']
			posevid_fp = db[idx]['pose_path']

			raw_cap = cv2.VideoCapture(raw_fp)
			posevid_cap = cv2.VideoCapture(posevid_fp)

			i = 0

			tl = (600, 120)
			br = (720, 0)
			skip = False

			while raw_cap.isOpened() and posevid_cap.isOpened() and webcam_cap.isOpened():
				_, raw_frame = raw_cap.read()
				_, posevid_frame = posevid_cap.read()
				_, webcam_frame = webcam_cap.read()

				webcam_frame = cv2.resize(webcam_frame, (720, 480))

				webcam_frame = cv2.rectangle(webcam_frame, tl, br, (0, 0, 255), 5)

				keypoints, output_data = pa.detect(webcam_frame)

				if isinstance(keypoints, np.ndarray) and isinstance(keypoints[7], np.ndarray) and keypoints[7][0] < br[0] and keypoints[7][0] > tl[0] and keypoints[7][1] > br[1] and keypoints[7][1]  < tl[1]:
					print('skip is selected')

					webcam_frame = cv2.rectangle(webcam_frame, tl, br, (0, 255, 0), 6)

					skip = True

				#print(keypoints)
				#print(db[idx]['pose_data'][i])

				sim = -1

				if len(db[idx]['pose_data']) <= i:
					break

				if isinstance(db[idx]['pose_data'][i], list) and isinstance(keypoints, np.ndarray):
					sim = cosine_similarity(np.expand_dims(keypoints.flatten(), axis=0), np.expand_dims(np.array(db[idx]['pose_data'][i]).flatten(), axis=0))[0]
					sim = float(sim)
					#print(sim)

					#webcam_frame = output_data

				if not isinstance(raw_frame, np.ndarray) or not isinstance(posevid_frame, np.ndarray):
					break

				raw_frame = cv2.resize(raw_frame, (720, 480))
				posevid_frame = cv2.resize(posevid_frame, (720, 480))

				if not isinstance(keypoints, np.ndarray):
					raw_str = cv2.imencode('.jpg', raw_frame)[1].tostring()
					posevid_str = cv2.imencode('.jpg', posevid_frame)[1].tostring()
					webcam_str = cv2.imencode('.jpg', webcam_frame)[1].tostring()

					with app.test_request_context('/'):
						socketio.emit('frame', {'raw': raw_str, 'posevid': posevid_str, 'webcam':webcam_str, 'sim': sim})

					continue

				pose_resized = []
				pose_resized2 = []

				max_dist = 0
				max_idx = -1

				for new_idx, ((x, y), (x2, y2)) in enumerate(zip(db[idx]['pose_data'][i], keypoints)):
					x = (x/1280.0) * (720)
					y = (y/720) * 480

					pose_resized.append((int(x), int(y)))
					pose_resized2.append((int(x2), int(y2)))

					if not (x + y) == 0:
						webcam_frame = cv2.circle(webcam_frame, (int(x), int(y)), 2, (0, 255, 0), 2)

					if not (x2 + y2) == 0:
						webcam_frame = cv2.circle(webcam_frame, (int(x2), int(y2)), 2, (255, 0, 0), 2)

					new_dist = (x2 - x)**2 + (y2 - y)**2

					if new_dist >= max_dist:
						max_dist = new_dist
						max_idx = new_idx

				mx, my = keypoints[max_idx]
				if not (mx + my) == 0 and not db[idx]['pose_data'][i][max_idx][0] + db[idx]['pose_data'][i][max_idx][1] == 0:
					webcam_frame = cv2.circle(webcam_frame, (int(mx), int(my)), 5, (0, 0, 255), 2)

					print(db[idx]['pose_data'][i][max_idx])

				for a in ["1,8","1,2,","1,5","2,3","3,4","5,6","6,7","8,9","9,10","10,11","8,12","12,13","1,0","13,14"]:
					p1 = int(a.split(',')[0])
					p2 = int(a.split(',')[1])

					#print(p1, p2)
					#print(pose_resized[p1])
					#print(pose_resized[p2])
					if not pose_resized[p1][0] == 0 and not pose_resized[p1][1] == 0 and not pose_resized[p2][0] == 0 and not pose_resized[p2][1] == 0:
						webcam_frame = cv2.line(webcam_frame, pose_resized[p1], pose_resized[p2], (255, 255, 0), 3)

					if not pose_resized2[p1][0] == 0 and not pose_resized2[p1][1] == 0 and not pose_resized2[p2][0] == 0 and not pose_resized2[p2][1] == 0:
						webcam_frame = cv2.line(webcam_frame, pose_resized2[p1], pose_resized2[p2], (0, 255, 0), 3)



				raw_str = cv2.imencode('.jpg', raw_frame)[1].tostring()
				posevid_str = cv2.imencode('.jpg', posevid_frame)[1].tostring()
				webcam_str = cv2.imencode('.jpg', webcam_frame)[1].tostring()

				#print('frame')
				#print(raw_str, posevid_str, webcam_str)

				with app.test_request_context('/'):
					socketio.emit('frame', {'status': 'active', 'raw': raw_str, 'posevid': posevid_str, 'webcam':webcam_str, 'sim': sim, 'enum': i_all, 'part': mappings[max_idx]})
				#emit('frame', {'data': 'test'})

				if skip:
					break

				i += 1

			with app.test_request_context('/'):
				socketio.emit('frame', {'status': 'done', 'has_next': i_all < (len(idx_split) - 1)})

			sleep(5)

			raw_cap.release()
			posevid_cap.release()

	def run(self):
		self.action()

class FoodThread(Thread):
	def __init__(self):
		super(FoodThread, self).__init__()

	def action(self):
		food_set = set()

		webcam_cap = cv2.VideoCapture(0)

		while webcam_cap.isOpened():
			_, webcam_frame = webcam_cap.read()

			global orig_frame

			orig_frame = webcam_frame.copy()

			webcam_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2GRAY)

			barcodes = pyzbar.decode(webcam_frame)

			if len(barcodes) > 0:
				print(barcodes)

				upc = barcodes[0].data.decode("utf-8")[1:]

				if not upc in food_set:
					food_set.add(upc)

					with app.test_request_context('/'):
						socketio.emit('food', {'data': upc}, namespace='/food')

			webcam_frame = cv2.resize(webcam_frame, (720, 480))

			webcam_str = cv2.imencode('.jpg', cv2.resize(orig_frame, (720, 480)))[1].tostring()

			with app.test_request_context('/'):
				socketio.emit('frame', {'status': 'active', 'img': webcam_str}, namespace='/food')

	def run(self):
		self.action()

@socketio.on('detect', namespace='/food')
def detect():
	global orig_frame

	print('detect')

	socketio.emit('return_detection', {'label': get_label(orig_frame)}, namespace='/food')

@socketio.on('connected', namespace='/food')
def connected_food(message):
	print(message)

	global thread

	if not thread.isAlive():
		print("Starting Thread")
		thread = FoodThread()
		thread.start()

@socketio.on('connected')
def connected(message):
	print(message)

	emit('frame', {'data': 'asdf'})

	global thread

	if not thread.isAlive():
		print("Starting Thread")
		thread = VidThread()
		thread.start()

if __name__ == '__main__':
	global db
	global idx_split
	idx_split = []

	global key_idx
	key_idx = {}

	db = load_db()
	add_difficulty()
	socketio.run(app)