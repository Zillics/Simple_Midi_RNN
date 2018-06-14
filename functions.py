import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, pitch, stream, duration
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def midiToPickle(filename):
    #MIDI -> music21 Score
    midi = converter.parse(filename) # music21.stream.Score
    partStream = midi.parts.stream() # music21.stream.Score
    #Music21 score -> Music21 RecursiveIterator
    notes_to_parse = midi.flat.notes
        
    #CREATE MATRIX
    notes = np.zeros((n,2)) #notes matrix: [pitch_i,duration_i],[pitch_i+1,duration_i+1]....
    i = 0
    prev_offset = 0 # offset of previous note
    prev_dur = 0 # duration of previous note
    #Rests are expressed as pitch 0 (arbitrary, Change if needed)
    for element in notes_to_parse:
        if isinstance(element, note.Note): #Filter out everything but notes
            diff = element.offset - prev_offset # diff: difference between note offsets in quarterLengths (?)
            if(diff - prev_dur > 0): # In other words: if there is a rest
                notes[i,0] = 0 # Rest is marked as having pitch 0 (arbitrary, Change if needed)
                notes[i,1] = diff - prev_dur # duration of rest = space between end of prev note and start of current note
            else:
                notes[i,0] = element.pitch.midi #Pitch as integer    
                notes[i,1] = element.duration.quarterLength #Duration as float

    #Store Matrix in file
    with open('data/tests/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)        

    return 0

    
def pickleToMidi(filename):
    #Open numpy matrix of notes + rests
    with open(filename, 'rb') as filepath:
        note_matrix = pickle.load(filepath)
    prev_dur = 0 # temporary variable for determining offset of current note
    prev_offset = 0 # temporary variable for determining offset of current note
    note_list = []

    for i in range(0,n_notes):
        new_note = note.Note(int(note_matrix[i,0]))
        new_note.offset = prev_dur + prev_offset
        new_note.duration.quarterLength = note_matrix[i,1]
        new_note.storedInstrument = instrument.Piano()
        note_list.append(new_note)
        # Update temporary values for next element
        prev_offset = new_note.offset
        prev_dur = new_note.duration.quarterLength

    midi_stream = stream.Stream(note_list)
    midi_stream.write('midi', fp='test_aria3.mid')
    
    return 0

