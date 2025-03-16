import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import cv2
import requests, json

def send_data_to_server(data):
    url = 'http://localhost:3000'
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            print("Data sent successfully")
        else:
            print(f"Failed to send data: {response.status_code}")
    except Exception as e:
        print(f"Error sending data: {e}")

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
	image.flags.writeable = False                  # Image is no longer writeable
	results = model.process(image)                 # Make prediction
	image.flags.writeable = True                   # Image is now writeable 
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
	return image, results
	
def draw_landmarks(image, results):
	mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
	mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
	mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
	mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
	
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    
def camera_tracker():
	cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

	with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

		if not cap.isOpened():
			print("Error: Could not open camera")
			exit()
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
		cap.set(cv2.CAP_PROP_FPS, 30)
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

		while cap.isOpened():
			ret, frame = cap.read()  
			if not ret:
				print("Error: Frame capture failed")
				break
			
			image, results = mediapipe_detection(frame, holistic)
			#print(results)
			
			# Draw landmarks
			draw_styled_landmarks(image, results)
			
			cv2.imshow('Camera Feed', image)
			#exit on q
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

		cap.release() 
		cv2.destroyAllWindows()
		
	# INPUTTING SIMPLIFIED DATA INTO ARRAYS FOR PROCESSING

	pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
	face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
	lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
	rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

def extract_keypoints(results):
	pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
	face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
	lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
	rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
	return np.concatenate([pose, face, lh, rh])
	
	
# SETUP FILE SYSTEM

DATA_PATH = os.path.join('MP_Data')

#ARRAY TO TRAIN ASL RECOGNITION
actions = np.array(['hello', 'how_are_you','thank_you']) 
no_sequences = 30
sequence_length = 30
start_folder = 30

def init_actions():
	for action in actions:
		# Build and ensure the action directory exists
		action_path = os.path.join(DATA_PATH, action)
		os.makedirs(action_path, exist_ok=True)
		
		# Get all directory names that are purely digits and convert them to integers
		subdirs = [int(name) for name in os.listdir(action_path) 
				   if name.isdigit() and os.path.isdir(os.path.join(action_path, name))]
		
		# Use 0 as the maximum if no digit-based directories exist
		dirmax = max(subdirs) if subdirs else 0

		# Create new directories for each sequence number
		for sequence in range(1, no_sequences + 1):
			new_dir = os.path.join(action_path, str(dirmax + sequence))
			os.makedirs(new_dir, exist_ok=True)
			
			
	# COLLECTING VALUES FOR TRAINING AND TESTING

	cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	cap.set(cv2.CAP_PROP_FPS, 30)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

	# Set mediapipe model 
	with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
		
		# NEW LOOP
		# Loop through actions
		for action in actions:
			# Loop through sequences aka videos
			for sequence in range(start_folder,start_folder+no_sequences):
				# Loop through video length aka sequence length
				for frame_num in range(sequence_length):

					# Read feed
					ret, frame = cap.read()

					# Make detections
					image, results = mediapipe_detection(frame, holistic)

					# Draw landmarks
					#draw_styled_landmarks(image, results)
					
					# NEW Apply wait logic
					if frame_num == 0: 
						cv2.putText(image, 'STARTING COLLECTION', (120,200), 
								   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
						cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
								   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
						# Show to screen
						cv2.imshow('OpenCV Feed', image)
						cv2.waitKey(500)
					else: 
						cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
								   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
						# Show to screen
						cv2.imshow('OpenCV Feed', image)
					
					# NEW Export keypoints
					keypoints = extract_keypoints(results)
					npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
					np.save(npy_path, keypoints)
	   
					if cv2.waitKey(10) & 0xFF == ord('q'):
						break
				
		cap.release()
		cv2.destroyAllWindows()

def predictor_feed():
	#Tensorflow logs directory
	log_dir = os.path.join('Logs')
	tb_callback = TensorBoard(log_dir=log_dir)

	label_map = {label:num for num,label in enumerate(actions)}
	sequences, labels = [], []
	#preprocess data with labels and features
	for action in actions:
		for sequence in range(start_folder,start_folder+no_sequences):
			window = []
			#append numpy array of video sequences and labels
			for frame_num in range(sequence_length):
				res = np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
				window.append(res)
			sequences.append(window)
			labels.append(label_map[action])
			
	x = np.array(sequences)
	y = to_categorical(labels).astype(int)
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05)

	model = Sequential()
	model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,1662)))
	model.add(LSTM(128,return_sequences=True,activation='relu'))
	model.add(LSTM(64,return_sequences=False,activation='relu'))
	model.add(Dense(64,activation='relu'))
	model.add(Dense(32,activation='relu'))
	model.add(Dense(actions.shape[0],activation='softmax'))
	
	#tensorboard model creation
	#model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
	#model.fit(x_train,y_train,epochs=2000,callbacks=[tb_callback])
	
	#model.save('action.h5') #model to load from
	
	yhat = model.predict(x_train)
	ytrue = np.argmax(y_train,axis=1).tolist()
	yhat = np.argmax(yhat,axis=1).tolist()
	
	print(multilabel_confusion_matrix(ytrue,yhat))
	print(accuracy_score(ytrue,yhat))
	
	sequence = []
	sentence = []
	threshold = 0.4	
	cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

	with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

		if not cap.isOpened():
			print("Error: Could not open camera")
			exit()
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
		cap.set(cv2.CAP_PROP_FPS, 30)
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

		while cap.isOpened():
			ret, frame = cap.read()  
			if not ret:
				print("Error: Frame capture failed")
				break
			
			image, results = mediapipe_detection(frame, holistic)
			#print(results)
			
			# Draw landmarks
			draw_styled_landmarks(image, results)
			
			keypoints = extract_keypoints(results)
			sequence.insert(0,keypoints)
			sequence = sequence[:30]
			if len(sequence)==30:
				res = model.predict(np.expand_dims(sequence,axis=0))[0]
				print(actions[np.argmax(res)])
				
			
			#if res[np.argmax(res)] > threshold:
			
			
			cv2.imshow('Camera Feed', image)
			#exit on q
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

		cap.release() 
		cv2.destroyAllWindows()

predictor_feed()
