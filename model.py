import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


SEQUENCE_LENGTH = 2000

def pickleToMatrix(filename):
	with open(pickle_file, 'rb') as filepath:
		matrix = pickle.load(filepath)
	return matrix

def prepare_sequences(matrix):

	sequence_length = SEQUENCE_LENGTH
	n_sequences = matrix.shape[1] - sequence_length
	n_features = matrix.shape[0]

	X = np.zeros((n_sequences,n_features,sequence_length))
	y = np.zeros((n_sequences,n_features))

	for i in range(0, n_sequences):
		sequence_in = matrix[:,i:i + sequence_length]
		sequence_out = matrix[:,i + sequence_length]
		X[i,:,:] = sequence_in
		y[i,:] = sequence_out
	return (X,y)

def create_network(X,output_shape):
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

if __name__ == '__main__':
	#pickle_file = 'data/aria_mel'
	#matrix = pickleToMatrix(pickle_file)
	#print(matrix.shape)
	#(X_train, y_train) =
	#prepare_sequences(matrix)
	#print(X_train.shape)
	#print(y_train.shape)
