import pretty_midi
import numpy as np
import pickle
import glob
import re
import os


SAMPLE_FREQUENCY = 400

# Converts a pretty midi object into compact numpy matrix
# returns matrix of format: [Pitch_i[128],Duration_i[1],Rest_before_i[1]]
def midiToMatrix(pretty_midi_obj,sample_frequency):
	interval_s = 1/sample_frequency
	piano_roll = pretty_midi_obj.get_piano_roll(fs=sample_frequency, times=None)
	time_steps = piano_roll.shape[1]
	t = 0
	n_notes = piano_roll.shape[0] # Typically 128
	out_matrix = np.zeros((time_steps,n_notes + 2))
	i = 0 # counter for number of notes in out_matrix
	rest = 0 # counter for rest before note
	while(t < time_steps):
		#Search for highest note which is activated
		k = n_notes - 1
		while(k >= 0):
			if(piano_roll[k,t] != 0):
				# Storing the start position of the note
				t0 = t
				# Search for end of note
				while(piano_roll[k,t] != 0):
					#NOTE ON
					t = t + 1
					if(t >= time_steps):
						break
				# Insert note
				#if(rest > 0):
					#print("REST FOUND: %f" % (rest))
				out_matrix[i,k] = 1
				out_matrix[i,128] = (t - t0)*interval_s #DURATION
				out_matrix[i,129] = rest*interval_s #REST BEFORE NOTE
				rest = 0 # Reset rest
				i = i + 1
				k = -1 # Jump out of second loop
			k = k - 1
		rest = rest + 1 # If no note found on that timestep: increase rest by one step
		t = t + 1
	out_matrix.resize(i,n_notes + 2)
	return out_matrix

# Converts matrix to MIDI file
def matrixToMidi(matrix,dest_file):
	# Create a PrettyMIDI object
	midi_obj = pretty_midi.PrettyMIDI()
	# Create an Instrument instance for a Piano instrument
	piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
	piano = pretty_midi.Instrument(program=piano_program)
	t = 0.0
	note_matrix = [ np.where(r==1)[0][0] for r in matrix[0:matrix.shape[0]-1,0:128] ] # Convert one hot encoding to single integers
	n_notes = len(note_matrix) # Number of consecutive notes
	# ToDo: also add rests
	print("Writing %d notes to MIDI file" % (n_notes))
	for i in range(0,n_notes):
		p = note_matrix[i]
		rest = matrix[i,129] # Rest before note
		s = t + rest # Start = current time + rest before note
		e = s + matrix[i,128] #End = start + duration of note in seconds
		note = pretty_midi.Note(velocity=100, pitch=p, start=s, end=e)
		piano.notes.append(note)
		t = e
	midi_obj.instruments.append(piano)
	midi_obj.write(dest_file)

# Takes list of matrices from one MIDI track (one per instrument/channel) as input, estimates which matrix is the melody, and returns index of that matrix
def isMelody(matrix_list):
	n = len(matrix_list)
	means = np.zeros(n)
	i = 0
	for matrix in matrix_list:
		means[i] = matrix[:,0].mean()
		i = i + 1
	idx = np.argmax(means)
	return idx

# Takes list of matrices from one MIDI track (one per instrument/channel) as input, estimates which matrix is the bass, and returns index of that matrix
def isBass(matrix_list):
	n = len(matrix_list)
	means = np.zeros(n)
	i = 0
	for matrix in matrix_list:
		means[i] = matrix[:,0].mean()
		i = i + 1
	idx = np.argmin(means)
	return idx

# List of matrices -> MIDI files
def matricesToMidis(matrix_list, dest_file):
	idx = 0
	melody_idx = isMelody(matrix_list)
	for matrix in matrix_list:
		if(idx == melody_idx):
			dest_filename = dest_file + '_melody'  + '.mid'
		else:
			dest_filename = dest_file + str(idx) + '.mid'
		matrixToMidi(matrix,dest_filename)
		idx = idx + 1

# List of matrices -> Pickle files
def matricesToPickle(matrix_list, dest_file):
	idx = 0
	melody_idx = isMelody(matrix_list)
	for matrix in matrix_list:
		if(idx == melody_idx):
			dest_filepath = dest_file + '_ch_' + 'mel'
		else:
			dest_filepath = dest_file + '_ch_' + str(idx)
		print("Writing channel %d into pickle file %s" (idx,dest_filepath))
		with open(dest_filepath, 'wb') as filepath:
			pickle.dump(matrix, filepath)
		idx = idx + 1

# Creates multiple matrices (one per instrument/channel) based on one MIDI file. Dumps matrix/matrices corresponding to selected channel into pickle file
def midiToPickles(source_MIDI_file, dest_file,sample_freq,channel):
	midi_data = pretty_midi.PrettyMIDI(source_MIDI_file)
	instr_list = midi_data.instruments
	idx = 0
	matrix_list = []
	for instrument in instr_list:
		print("Converting MIDI channel %d to numpy matrix...." % (idx))
		matrix = midiToMatrix(instrument,sample_freq)
		if(matrix.shape[0] > 0): # Ignore empty matrices
			matrix_list.append(matrix)
		else:
			print("Skipping empty MIDI channel %d" % (idx))
		idx += 1
	# WRITE INTO PICKLE FILE(s)
	if(channel == 'mel'):
		print("Writing melody track into pickle file %s...." % (dest_file))
		melody_idx = isMelody(matrix_list)
		with open(dest_file, 'wb') as filepath:
			pickle.dump(matrix_list[melody_idx],filepath)
	else:
		if(channel == 'bass'):
			print("Writing bass track into pickle file %s...." % (dest_file))
			bass_idx = isBass(matrix_list)
			with open(dest_file, 'wb') as filepath:
				pickle.dump(matrix_list[bass_idx],filepath)
		else:
			print("Writing all tracks into separate pickle files....")
			matricesToPickle(matrix_list,dest_file)

# Creates MIDI file based on matrix stored in pickle file
def pickleToMidi(pickle_file,dest_file):
	print("Converting pickle file %s into MIDI file %s" % (pickle_file,dest_file))
	with open(pickle_file, 'rb') as filepath:
		matrix = pickle.load(filepath)
	matrixToMidi(matrix,dest_file)

# For only melody tracks: channel = 'mel'
def picklesToMidis(pickle_folder,dest_folder,channel='all'):
	if(channel == 'all'):
		path = pickle_folder + '/*'
	else:
		path = pickle_folder + '/*_ch_' + str(channel)
	for pickle_file in glob.glob(path):
		with open(pickle_file, 'rb') as filepath:
			note_matrix = pickle.load(filepath)
		dest_filepath = dest_folder + '/' + os.path.basename(pickle_file) + '.mid'
		print('Writing MIDI file: ' + dest_filepath)
		matrixToMidi(note_matrix,dest_filepath)

# Convert all MIDI files to numpy matrix format. Dump each matrix to its separate pickle file 
def midisToPickles(midi_folder,pickle_folder,sample_freq=SAMPLE_FREQUENCY,channel='mel'):
	path = midi_folder + '/**/*.mid'
	idx = 0 # Prefix for each file in order to avoid overwriting on file with identical name
	for midi_file in glob.glob(path,recursive=True):
		# pickle_folder/song_name. Remove '.mid' from path name
		dest_filepath = pickle_folder + '/' + str(idx) + '_' + os.path.basename(midi_file)[:-4]
		print("Exporting " + midi_file + " to pickle:")
		try:
			midiToPickles(midi_file,dest_filepath,sample_freq,channel)
		except Exception as e:
			print("midiToPickles failed for MIDI file %s, due to: %s" % (dest_filepath,e))
		else:
			print("MIDI to pickle file sucessful!")

# Convert all MIDI files to one numpy matrix. Dump matrix into  
def midisToPickle(midi_folder,pickle_file,sample_freq=SAMPLE_FREQUENCY,channel='mel'):
	path = midi_folder + '/**/*.mid'
	print(path)
	matrix_list = [] # List of matrices; each representing one MIDI file
	n_notes = 0 # Counter for total number of notes among all MIDI files
	for midi_file in glob.glob(path,recursive=True):
		print("Converting %s to numpy matrix format...." % (midi_file))
		midi_data = pretty_midi.PrettyMIDI(midi_file)
		instr_list = midi_data.instruments
		channel_list = []
		for instrument in instr_list:
			matrix = midiToMatrix(instrument,sample_freq)
			if(matrix.shape[0] > 0): # Ignore empty matrices
				channel_list.append(matrix)
			else:
				print("Skipping empty MIDI channel")
		if(channel=='mel'):
			idx = isMelody(channel_list)
		else:
			if(channel=='bass'):
				idx = isBass(channel_list)
			else:
				idx = 0
		matrix_list.append(channel_list[idx])
		n_notes += channel_list[idx].shape[0]
	out_matrix = np.zeros((n_notes,130))
	i = 0
	for m in matrix_list:
		out_matrix[i:(i+m.shape[0]),:] = m # Concatenate next matrix
		i += m.shape[0]
	print("Matrix of shape %d , %d ready" % (out_matrix.shape[0],out_matrix.shape[1]))
	print("Dumping into pickle file %s" % (pickle_file))
	# Finally dump the matrix into one single pickle file
	with open(pickle_file, 'wb') as filepath:
		pickle.dump(out_matrix,filepath)


if __name__ == '__main__':
	#midi_src_folder = 'source_midi/bach'
	#pickle_folder = 'data/bach-melody'
	#dest_midi_folder = 'output_midi'
	#pickleToMidi('data/bach-melody/fugue21','test_5.mid')
	with open('data/train_data/ALL_SEQUENCES_1','rb') as filepath:
		X,y = pickle.load(filepath)
	print(X[0,:,:])
	#matrixToMidi(X[1000,:,:],'SEQUENCE_TEST.mid')
