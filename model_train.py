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

def pickleToMatrix(filename):
	with open(pickle_file, 'rb') as filepath:
		matrix = pickle.load(filepath)
	return matrix

def prepare_sequences(matrix, sequence_length=SEQUENCE_LENGTH):

	sequence_length = SEQUENCE_LENGTH
	n_sequences = matrix.shape[0] - sequence_length
	n_features = matrix.shape[1]

	X = np.zeros((n_sequences,sequence_length,n_features))
	y = np.zeros((n_sequences,n_features))

	for i in range(0, n_sequences):
		sequence_in = matrix[i:i + sequence_length,:]
		sequence_out = matrix[i + sequence_length,:]
		X[i,:,:] = sequence_in
		y[i,:] = sequence_out
	return (X,y)

def create_network(X,n_vocab=130,output_shape=512):
	model = Sequential()
	model.add(LSTM(
		output_shape,
		input_shape=(X.shape[1], X.shape[2]),
		return_sequences=True
	))
	model.add(Dropout(0.3))
	model.add(LSTM(output_shape, return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(output_shape))
	model.add(Dense(256))
	model.add(Dropout(0.3))
	model.add(Dense(n_vocab))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	return model

def train(model, train_data_path,epoch_n=100):#network_input, network_output):
	""" train the neural network """
	print("Starting training.....")
	for i in range(0,epoch_n):
		print("Epoch %d / %d" % (i + 1,epoch_n))
		for pickle_file in glob.glob(train_data_path):
			print("TRAINING ON %s :" % (pickle_file))
			matrix = pickleToMatrix(pickle_file)
			print("Preparing sequences....")
			(X,y) = prepare_sequences(matrix)
			print("Training....")
			model.fit(X, y, epochs=1, batch_size=32) #callbacks=callbacks_list)
			print("Saving weights....")
			weight_filepath = ("weights/weights-improvement-epoch%d-bigger.hdf5" % (i))
			model.save(weight_filepath)

if __name__ == '__main__':
	train_data_location = 'data/bach-melody/'
	sample_matrix = pickleToMatrix('data/bach-melody/aria')
	(X_train, y_train) = prepare_sequences(sample_matrix)
	model = create_network(X_train)
	train(model,train_data_location)
	#print(X_train.shape)
	#print(y_train.shape)

