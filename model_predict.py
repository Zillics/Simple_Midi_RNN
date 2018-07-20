import numpy as np
import pickle
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from model_train import prepare_sequences
from preprocessing3 import matrixToMidi

OUTPUT_LENGTH = 500
N_FEATURES = 130
SEQUENCE_LENGTH = 50


def generate_notes(model,sample_pickle,dest_path,n_notes=OUTPUT_LENGTH):
	with open(sample_pickle, 'rb') as filepath:
		matrix_sample = pickle.load(filepath)
	(X_sample,y_sample) = prepare_sequences(matrix_sample,SEQUENCE_LENGTH)
	start_idx = np.random.randint(0,(X_sample.shape[0]-1))
	input_seq = X_sample[start_idx,:,:]
	input_seq = np.reshape(input_seq,(1,input_seq.shape[0],input_seq.shape[1])) # Input shape for LSTM: (n_samples,n_timesteps,n_features)
	output_seq = np.zeros((OUTPUT_LENGTH,130))
	for i in range(0,OUTPUT_LENGTH):
		#print(input_seq[0,:,:])
		#print("-----------------------------")
		output_seq[i,:] = model.predict(input_seq,verbose=0)
		input_seq[0,:,:] = np.roll(input_seq,-1,axis=1)
		input_seq[0,SEQUENCE_LENGTH-1,:] = output_seq[i,:]
	with open(dest_path, 'wb') as filepath:
		pickle.dump(output_seq,filepath)
	return output_seq

def predictionToMIDI(matrix,MIDI_dest_file):
	out_matrix = np.zeros(matrix.shape)
	# Select notes with highest confidence
	notes = matrix[:,0:127].argmax(axis=1)
	for i in range(0,notes.shape[0]):
		out_matrix[i,notes[i]] = 1
	# Add the duration and the rest before note
	out_matrix[:,128:130] = matrix[:,128:130]
	matrixToMidi(out_matrix,MIDI_dest_file)
	print(matrix[:,128])
	return matrix



if __name__ == '__main__':
	weights_path = 'weights/weights-improvement-epoch9-bigger.hdf5'
	sample = 'data/melody/1_aria'
	prediction_output = 'output_data/prediction_1'
	MIDI_dest = 'output_data/MIDI/test_1.mid'
	#LSTM_model = load_model(weights_path)
	#matrix = generate_notes(LSTM_model,sample,prediction_output)
	with open(prediction_output, 'rb') as filepath:
		matrix = pickle.load(filepath)
	predictionToMIDI(matrix,MIDI_dest)