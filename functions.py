import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, pitch, stream, duration, midi
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import math

#np.set_printoptions(threshold=1000)
#WORRRRKS!!!!
def testImport(filename,dest_filename):
    mf = midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()
    print(len(mf.tracks))
    stream = midi.translate.midiFileToStream(mf)
    elements = stream.flat.notesAndRests
    mf2 = midi.translate.streamToMidiFile(stream)
    mf.open(dest_filename,'wb')
    mf.write()
    mf.close()

def testImport2(filename,dest_filename):
    mf = midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()
    s = midi.translate.midiFileToStream(mf)
    notes = s.flat.notes
    noteRests = s.flat.notesAndRests
    elements = s.flat
    prev_offset = -1
    '''
    for note_i in notes:
        if isinstance(note_i,note.Note):
            if(note_i.offset == prev_offset):
                print("Offset: %f, pitch: %s, duration: %s !!!!!!!!!!!!!" % (note_i.offset,note_i.pitch,note_i.duration.type))
            else:
                print("Offset: %f, pitch: %s, duration: %s" % (note_i.offset,note_i.pitch,note_i.duration.type))
            prev_offset = note_i.offset
        else:
            print(type(note))
    '''
    element_list = []
    for element in elements:
        element_list.append(element)
    #print(element_list[0:10])
    #print(type(element_list))
    #midi_stream = stream.Stream(element_list)
    #midi_stream.write('midi', fp=dest_filename)

    print(element_list)


def midiToPickle(filename):
    #MIDI -> music21 Score
    midi = converter.parse(filename) # music21.stream.Score
    partStream = midi.parts.stream() # music21.stream.Score
    #Music21 score -> Music21 RecursiveIterator
    notes_to_parse = midi.flat
    notes = []
    for element in notes_to_parse:
        notes.append(element)

    '''
    #CREATE MATRIX
    n = 0
    #Determine number of notes in RecursiveIterator
    for note_i in notes_to_parse:
        n = n + 1
        pass
    notes = np.zeros((n,3))
    i = 0
    prev_offset = 0 # offset of previous note
    prev_dur = 0 # duration of previous note
    # notes matrix: [pitch_i,duration_i, pause_i],[pitch_i+1,duration_i+1,pause_i+1]....
    for element in notes_to_parse:
        if isinstance(element, note.Note): # Filter out everything but notes
            diff = element.offset - prev_offset # diff: difference between note offsets in quarterLengths (?)
            #CURRENT PITCH
            notes[i,0] = element.pitch.midi # Pitch as integer
            #CURRENT DURATION
            duration = element.duration.quarterLength
            #if(duration == 0):
                #duration = element.next().offset - float(element.offset) # Small value, but not maybe optimal??
            notes[i,1] = duration #Duration as float in quarterLength
            pause = abs(diff - prev_dur)
            if(i > 0):
                notes[i-1,2] = pause # Pause after the note
            # Updating temp values to current values
            prev_dur = duration
            prev_offset = element.offset 
            i = i + 1
    #Store Matrix in file
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)        
    #print(notes[0:10,:])
    '''
    midi_stream = stream.Stream(notes)
    midi_stream.write('midi', fp='aria_test9.mid')

    return 0

    
def pickleToMidi(src_filename,dest_filename):
    #Open numpy matrix of notes + rests
    with open(src_filename, 'rb') as filepath:
        note_matrix = pickle.load(filepath)

    n_notes = note_matrix.shape[0]
    prev_offset = 0
    prev_pause = 0
    prev_dur = 0
    note_list = []
    for i in range(0,n_notes):
        new_note = note.Note(int(note_matrix[i,0]))
        new_note.duration.quarterLength = note_matrix[i,1]
        new_note.storedInstrument = instrument.Piano() 
        offset = float(prev_offset) + float(prev_dur) + float(prev_pause)
        if(math.isclose(prev_offset,offset)):
            offset = prev_offset + 0.1
        new_note.offset = offset
        #if(new_note.offset == prev_offset):
        #    print("offset: %f , prev_offset: %f , new_note.offset: %f, condition: %s" % (offset,prev_offset,new_note.offset,math.isclose(prev_offset,offset)))
        note_list.append(new_note)
        # Update temporary values for next element
        prev_offset = new_note.offset
        prev_dur = note_matrix[i,1] #new_note.duration.quarterLength
        prev_pause = note_matrix[i,2]

    #print(note_matrix)
    prev_offset = -1
    for note_i in note_list:
        print("Offset: %f, pitch: %s, duration: %s" % (note_i.offset,note_i.pitch,note_i.duration.type))
        #if(prev_offset == note_i.offset):
        #    print("Offset: %f, pitch: %s, duration: %s" % (note_i.offset,note_i.pitch,note_i.duration.type))
        #prev_offset = note_i.offset

    midi_stream = stream.Stream(note_list)
    midi_stream.write('midi', fp=dest_filename)
    
    return 0

def midiToMidi(src_filename,dest_filename):
    midiToPickle(src_filename)
    #pickleToMidi('data/notes',dest_filename)
    return 0

if __name__ == '__main__':
    #midiToMidi('aria_melody.mid','aria_test8.mid')
    testImport2('aria_melody.mid','testWrite2.mid')
