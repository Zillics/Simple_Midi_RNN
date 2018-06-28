import pretty_midi
import numpy as np
import pickle

def n_instruments(filename):
	midi_data = pretty_midi.PrettyMIDI(filename)
	instruments_n = 0
	instr = midi_data.instruments
	print(len(instr))

# Converts a pretty midi object into compact numpy matrix
# returns matrix of format: [Pitch_i,Duration_i,Rest_before_i]
def midiToMatrix(pretty_midi_obj,sample_frequency):
	interval_s = 1/sample_frequency
	piano_roll = pretty_midi_obj.get_piano_roll(fs=sample_frequency, times=None)
	time_steps = piano_roll.shape[1]
	out_matrix = np.zeros((time_steps,3))
	t = 0
	n_notes = piano_roll.shape[0]
	i = 0 # counter for number of notes in out_matrix
	rest = 0 # counter for rest before note
	while(t < time_steps):
		#Search for highest note which is activated
		rest = rest + 1
		k = n_notes - 1
		while(k >= 0):
			if(piano_roll[k,t] != 0):
				# Storing the start position of the note
				t0 = t
				#Search for end of note
				while(piano_roll[k,t] != 0):
					#NOTE ON
					t = t + 1
					if(t >= time_steps):
						#not_limit = False
						break
				# Insert note
				out_matrix[i,0] = int(k) #PITCH
				out_matrix[i,1] = (t - t0)*interval_s #DURATION
				out_matrix[i,2] = rest*interval_s #REST BEFORE NOTE
				#print(out_matrix[i,:])
				i = i + 1
				k = -1 # Jump out of second loop
			k = k - 1
			rest = 0
		t = t + 1
	out_matrix.resize(i + 1,3)
	return out_matrix

# Converts matrix of format [Pitch_i,Duration_i,Rest_before_i] to midi file
def matrixToMidi(matrix,dest_file):
	# Create a PrettyMIDI object
	midi_obj = pretty_midi.PrettyMIDI()
	# Create an Instrument instance for a Piano instrument
	piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
	piano = pretty_midi.Instrument(program=piano_program)
	n_notes = matrix.shape[0]
	t = 0.0
	for i in range(0,n_notes):
		p = int(matrix[i,0])
		s = matrix[i,2] + t # Start = rest before note + current time
		e = s + matrix[i,1] # End = start + duration of note in seconds
		note = pretty_midi.Note(velocity=100, pitch=p, start=s, end=e)
		#print("pitch: %d start: %f , end: %f" % (p,s,e))
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


def matricesToMidi(matrix_list, dest_file):
	idx = 0
	melody_idx = isMelody(matrix_list)
	for matrix in matrix_list:
		if(idx == melody_idx):
			dest_filename = dest_file + '_melody'  + '.mid'
		else:
			dest_filename = dest_file + str(idx) + '.mid'
		matrixToMidi(matrix,dest_filename)
		idx = idx + 1

# Creates multiple matrices (one per instrument/channel) of format [Pitch_i,Duration_i,Rest_before_i] based on one MIDI file
def midiToMatrices(source_MIDI_file, sample_freq, dest_file):
	midi_data = pretty_midi.PrettyMIDI(source_MIDI_file)
	instr_list = midi_data.instruments
	matrix_list = []
	for instrument in instr_list:
		matrix_list.append(midiToMatrix(instrument,sample_freq))
	#matricesToMidi(matrix_list,dest_file) # TEMPORARY, just for testing function
	matricesToPickle(matrix_list,dest_file)


# Takes list of matrices and dumps each of them into their own pickle files
def matricesToPickle(matrix_list, dest_file):
	melody_idx = isMelody(matrix_list)
	idx = 0
	for matrix in matrix_list:
		if(idx == melody_idx):
			dest_filepath = dest_file + str(idx) + '_mel'
		else:
			dest_filepath = dest_file + str(idx)
		with open(dest_filepath, 'wb') as filepath:
			pickle.dump(matrix, filepath)
		idx = idx + 1

def pickleToMidi(pickle_file,dest_file):
	with open(pickle_file, 'rb') as filepath:
		matrix = pickle.load(filepath)
	matrixToMidi(matrix,dest_file)

if __name__ == '__main__':
	#source_file = 'var6c2.mid'
	#midiToMatrices(source_file,200,'data/var6c2')
	pickleToMidi('data/var6c20_mel','output_midi/var6c20_mel.mid')
	pickleToMidi('data/var6c21','output_midi/var6c21.mid')
	pickleToMidi('data/var6c22','output_midi/var6c22.mid')
