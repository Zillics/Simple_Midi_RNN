import numpy as np
import pickle
import glob
import re
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


SEQUENCE_LENGTH = 50
TRAIN_DATA_SIZE = 20000 # Maximum number of sequences in each training input (i.e. X.shape[0])

def pickleToMatrix(pickle_file):
	with open(pickle_file, 'rb') as filepath:
		matrix = pickle.load(filepath)
	return matrix

def prepare_sequences(matrix, sequence_length=SEQUENCE_LENGTH):
	if(matrix.shape[0] < sequence_length + 1):
		raise ValueError("Matrix contains less notes than sequence length (%d vs %d)" % (matrix.shape[0],sequence_length))
	n_sequences = matrix.shape[0] - sequence_length
	n_features = matrix.shape[1]

	X = np.zeros((n_sequences,sequence_length,n_features),dtype=np.float32)
	y = np.zeros((n_sequences,n_features),dtype=np.float32)

	for i in range(0, n_sequences):
		sequence_in = matrix[i:i + sequence_length,:]
		sequence_out = matrix[i + sequence_length,:]
		X[i,:,:] = sequence_in
		y[i,:] = sequence_out
	return (X,y)

# Create one sequence matrix based on pickle files found in data_path
def picklesToSequence(data_path,sequence_length=SEQUENCE_LENGTH):
	path = data_path + '/*'
	X_all = np.zeros((0,sequence_length,130),dtype=np.float32)
	y_all = np.zeros((0,130),dtype=np.float32)
	length = 0 # Counter for number of sequences stored so far
	for pickle_file in glob.glob(path):
		print("Preparing sequences for %s....." % (pickle_file))
		matrix = np.array(pickleToMatrix(pickle_file),dtype=np.float32)
		try:
			(X,y) = prepare_sequences(matrix)
		except Exception as e:
			print(e)
			print("Preparing sequences failed. Skipping this track.")
		else:
			length += X.shape[0]
			print("sequences so far: %d : " % (length))
			X_all = np.append(X_all,X,axis=0)
			y_all = np.append(y_all,y,axis=0)
	output_path = 'data/train_data/ALL_SEQUENCES'
	print("All sequences prepared. Writing to pickle file %s" % (output_path))
	with open(output_path,'wb') as filepath:
		pickle.dump((X_all,y_all),filepath)
	return X_all,y_all



# Create sequence matrices of size [~limit,sequence_length,130].
# Limit is for restricting the shape[0] of the output matrix in order to avoid MemoryError.
def prepare_all_sequences(data_path,sequence_length=SEQUENCE_LENGTH,limit=TRAIN_DATA_SIZE):
	path = data_path + '/*'
	X_all = np.zeros((0,sequence_length,130))
	y_all = np.zeros((0,130))
	length = 0 # Counter for number of sequences stored so far
	i = 0 # Counter for number of separate training data divisions
	for pickle_file in glob.glob(path):
		print("Preparing sequences for %s....." % (pickle_file))
		matrix = pickleToMatrix(pickle_file)
		try:
			(X,y) = prepare_sequences(matrix)
		except Exception as e:
			print(e)
			print("Preparing sequences failed. Skipping this track.")
		else:
			print("sequences so far: %d : " % (length))
			if(length + X.shape[0] >= limit):
				output_path = 'data/train_data/ALL_SEQUENCES_' + str(i)
				print("Limit %d exceeded. Writing to pickle file %s" % (limit,output_path))
				with open(output_path, 'wb') as filepath:
					pickle.dump((X_all,y_all),filepath)
					X_all = np.zeros((0,sequence_length,130))
					y_all = np.zeros((0,130))
				length = 0
				i += 1
			X_all = np.append(X_all,X,axis=0)
			y_all = np.append(y_all,y,axis=0)
			length += X.shape[0]
	output_path = 'data/train_data/ALL_SEQUENCES_' + str(i)
	print("All sequences prepared. Writing to pickle file %s" % (output_path))
	with open(output_path,'wb') as filepath:
		pickle.dump((X_all,y_all),filepath)
	return X_all,y_all


def create_network(in_shape=(SEQUENCE_LENGTH,130)):
	model = Sequential()
	model.add(LSTM(
		512,
		input_shape=(in_shape),
		return_sequences=True
	))
	model.add(Dropout(0.3))
	model.add(LSTM(512, return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(512))
	model.add(Dense(256))
	model.add(Dropout(0.3))
	model.add(Dense(130))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	return model

def train_1(model, train_data_path,epoch_n=200):#network_input, network_output):
	""" train the neural network """
	print("Starting training.....")
	for i in range(0,epoch_n):
		print("Epoch %d / %d" % (i + 1,epoch_n))
		path = train_data_path + '/*'
		for pickle_file in glob.glob(path):
			print("TRAINING ON %s , epoch %d :" % (pickle_file,i))
			matrix = pickleToMatrix(pickle_file)
			print("Preparing sequences....")
			try:
				(X,y) = prepare_sequences(matrix)
			except Exception as e:
				print(e)
				print("Preparing sequences failed. Skipping this track.")
			else:
				print("Training....")
				model.fit(X, y, epochs=1, batch_size=64) #callbacks=callbacks_list)
				print("Saving weights....")
		weight_filepath = ("weights/weights-improvement-epoch%d-{loss:.4f}-bigger.hdf5" % (i))
		model.save(weight_filepath)

def train_2(model, train_data_path,epoch_n=200):#network_input, network_output):
	""" train the neural network """
	print("Starting training.....")
	for i in range(0,epoch_n):
		print("Epoch %d / %d" % (i + 1,epoch_n))
		for pickle_file in glob.glob(train_data_path + '/*'):
			print("TRAINING ON %s , epoch %d :" % (pickle_file,i))
			with open(pickle_file,'rb') as filepath:
				X,y = pickle.load(filepath)
			model.fit(X, y, epochs=1, batch_size=64) #callbacks=callbacks_list)
		print("Saving weights....")
		weight_filepath = ("output/weights-improvement-epoch%d-{loss:.4f}-bigger.hdf5" % (i))
		model.save(weight_filepath)

def train_3(model, train_data_path, epoch_n=400):

	filepath = "/output/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
	checkpoint = ModelCheckpoint(
		filepath,
		monitor='loss',
		verbose=0,
		save_best_only=True,
		mode='min'
	)
	with open(train_data_path,'rb') as filepath:
		X,y = pickle.load(filepath)
	callbacks_list = [checkpoint]
	model.fit(X, y, epochs=epoch_n, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
	train_data_location = '/mount_point/ALL_SEQUENCES'
	model = create_network()
	train_3(model,train_data_location)

