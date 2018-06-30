import pretty_midi
import numpy as np
import pickle
import glob
import re
import os
from sklearn.preprocessing import OneHotEncoder

SAMPLE_FREQUENCY = 200

# Return vector with all zeros except element (>0) of highest index of input vector
def getHighest(vector):
	n = vector.shape[0]
	out_vector = np.zeros(n)
	i = n - 1
	while(i >= 0):
		if vector[i] > 0:
			out_vector[i] = vector[i]
			return out_vector
		else:
			i -= 1
	return out_vector
# Converts a pretty midi object into compact numpy matrix
# returns matrix of shape (129,time_steps)
def midiToMatrix(pretty_midi_obj,sample_frequency):

	interval_s = 1/sample_frequency
	piano_roll = pretty_midi_obj.get_piano_roll(fs=sample_frequency, times=None)
	time_steps = piano_roll.shape[1]
	t = 0
	n_notes = piano_roll.shape[0]
	out_matrix = np.zeros((n_notes + 1,time_steps)) # Reserve one space for rests
	out_matrix[0:128,:] = piano_roll
	out_matrix[out_matrix > 0] = 1
	for i in range(0,time_steps):
		i_sum = out_matrix[0:128,i].sum()
		if(i_sum < 1): # If there are no notes: mark rest category with one
			out_matrix[128,i] = 1
		if(i_sum > 1):
			#Only keep note of highest pitch
			out_matrix[0:128,i] = getHighest(out_matrix[0:128,i])
	return out_matrix
# Midifile -> list of piano roll-matrices (one per instrument)
def midiToMatrices(midi_file,sample_freq):
	midi_data = pretty_midi.PrettyMIDI(midi_file)
	instr_list = midi_data.instruments
	matrix_list = []
	for instrument in instr_list:
		matrix = midiToMatrix(instrument,sample_freq)
		if(matrix.shape[1] > 0): # Ignore empty matrices
			matrix_list.append(matrix.astype(int))
			#print(matrix.shape)
	return matrix_list

# Saves matrix (from list of matrices) of preferred channel to pickle file 
def matricesToPickle(matrix_list,dest_path,ch='mel'):
	if(ch == 'mel'):
		idx = isMelody(matrix_list)
	else:
		if(ch == 'bass'):
			idx = isBass(matrix_list)
		else:
			idx = ch
	pickle_file = dest_path + '_' + str(ch)
	print("Dumping to file " + pickle_file)
	with open(pickle_file, 'wb') as filepath:
		pickle.dump(matrix_list[idx], filepath)

# Determines which of matrices is melody and returns index of that matrix
def isMelody(matrix_list):
	means = np.zeros(len(matrix_list))
	j = 0
	for matrix in matrix_list:
		note_matrix = np.zeros(matrix.shape[1])
		for i in range(0,matrix.shape[1]):
			note = np.where(matrix[:,i]==1)[0][0]
			if(note > 127): # Discard rests = 128
				note = 0
			note_matrix[i] = note
		means[j] = note_matrix.mean()
		j += 1
	idx = np.argmax(means)
	return idx
# Determines which of matrices is bass and returns index of that matrix
def isBass(matrix_list):
	means = np.zeros(len(matrix_list))
	j = 0
	for matrix in matrix_list:
		note_matrix = np.zeros(matrix.shape[1])
		for i in range(0,matrix.shape[1]):
			note = np.where(matrix[:,i]==1)[0][0]
			if(note > 127): # Discard rests = 128
				note = 0
			note_matrix[i] = note
		means[j] = note_matrix.mean()
		j += 1
	idx = np.argmin(means)
	return idx

def matrixToMidi(matrix,dest_file,sample_freq):
	interval_s = 1/sample_freq

	# Create a PrettyMIDI object
	midi_obj = pretty_midi.PrettyMIDI()
	# Create an Instrument instance for a Piano instrument
	piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
	piano = pretty_midi.Instrument(program=piano_program)
	n_notes = matrix.shape[0]
	t = 0.0
	e = t
	s = t
	note_list = [ np.where(r==1)[0][0] for r in matrix.transpose() ]
	prev_note = 130 # arbitrary number other than 0 - 128
	for note_i in note_list:
		if(prev_note == note_i):
			e += interval_s
		else:
			# add previous note (if it is not a rest)
			e += interval_s
			if(prev_note < 128):
				note = pretty_midi.Note(velocity=100, pitch=p, start=s, end=e)
				piano.notes.append(note)
				#print("start %f end %f" % (s,e))
			# start new note
			p = note_i # new note -> new pitch
			s = t # Start = current time
			e = s # initialize end time
		prev_note = note_i
		t += interval_s
	# add final note
	note = pretty_midi.Note(velocity=100, pitch=p, start=s, end=e)
	piano.notes.append(note)
	#print(matrix.shape)
	midi_obj.instruments.append(piano)
	midi_obj.write(dest_file)

def matricesToMidi(matrix_list,song_name,sample_freq=SAMPLE_FREQUENCY):
	idx = 0
	for matrix in matrix_list:
		filename = song_name + str(idx) 
		matrixToMidi(matrix,filename,sample_freq)

def midisToPickles(midi_folder,pickle_folder,sample_freq=SAMPLE_FREQUENCY):
	midi_path = midi_folder + '/*.mid'
	for midi_file in glob.glob(midi_path):
		with open(midi_file, 'rb') as filepath:
			matrix_list = midiToMatrices(midi_file,sample_freq)
			pickle_name = pickle_folder + '/' + os.path.basename(midi_file)[:-4]
			matricesToPickle(matrix_list,pickle_name)

def picklesToMidis(pickle_folder, midi_folder, sample_freq=SAMPLE_FREQUENCY):
	pickle_path = pickle_folder + '/*'
	for pickle_file in glob.glob(pickle_path):
		with open(pickle_file, 'rb') as filepath:
			matrix = pickle.load(filepath)
		dest_filepath = midi_folder + '/' + os.path.basename(pickle_file) + '.mid'
		print('Writing MIDI file: ' + dest_filepath)
		matrixToMidi(matrix,dest_filepath,sample_freq)

if __name__ == '__main__':
	midi_src_folder = 'source_midi'
	pickle_folder = 'data'
	midi_dest_folder = 'output_midi'
	midi_file = midi_src_folder + '/aria.mid'
	
	midisToPickles(midi_src_folder,pickle_folder)
	picklesToMidis(pickle_folder, midi_dest_folder)
	#matrix_list = midiToMatrices(midi_file,200)
	#isMelody(matrix_list)
	#for matrix in matrix_list:
	#	print(matrix.shape)
	#matricesToPickle(matrix_list, pickle_folder)
	#matricesToMidi(matrix_list,midi_dest_folder)