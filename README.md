# Simple_Midi_RNN

## Goal:
To generate MIDI music in the style of the training samples. 
The implementation steps:
1. Proof of concept for methodology: preprocessing steps, input data format, ML model type, rough understanding of hyperparameters
2. Generate basic monophonic melodies
3. Generate multiple melodies in separate layers (melody,bass,harmonies etc.)

## 1. Proof of concept
### Preprocessing
Data: MIDI files (one per song)  
Preprocessing steps:
1. Convert MIDI file to PrettyMIDI object
2. Convert each channel/instrument into numpy matrix
3. Determine which matrix belongs to melody track
4. Store melody matrix in Pickle file

#### Matrix format
n = total number of notes  
**Shape:** 130xn  
0-127: Pitch of note (one-hot encoded)  
128: Duration of note  
129: Duration of rest before note  

This matrix format allows for a compressed representation of a monophonic melody. Its shape is independent of the time resolution.
The standard pianoroll format grows very rapidly in the time dimension. 
The resulting matrix from one song after splitting the piano roll up into sequences for the LSTM was so large that it reached the memory limit of my python setup.
The length of the compressed matrix format is simply equal to the number of notes, as opposed to time steps (which is quite excessive).

### Training
The training will be implemented on some type of LSTM neural network. 
The reason for this is mainly that the sequential nature of music makes the whole memory-aspect of the network quite central. 
Also, after some searching on the web I concluded that many similar projects tend to use LSTM networks
