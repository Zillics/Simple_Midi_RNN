import pretty_midi
import numpy as np

def midiToMatrix(filename,sample_frequency):
	midi_data = pretty_midi.PrettyMIDI(filename)
	interval_s = 1/sample_frequency

	piano_roll = midi_data.get_piano_roll(fs=sample_frequency, times=None)


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
	print(out_matrix[:,2])
	print(out_matrix[:,1])
	print(out_matrix[:,0])
	return out_matrix

def matrixToMidi(matrix,filename):
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
	midi_obj.write(filename)


if __name__ == '__main__':
	matrix = midiToMatrix('midi-files/aria_melody.mid',500)
	print(matrix.shape)
	matrixToMidi(matrix,'midi-files/aria_pretty_500.mid')