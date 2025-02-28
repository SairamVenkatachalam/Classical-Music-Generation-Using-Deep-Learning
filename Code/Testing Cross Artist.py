#%%
import numpy as np
import pretty_midi
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, BatchNormalization, LSTM, Dense, Softmax, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if tf.config.experimental.list_physical_devices('GPU'):
    print("Using GPU")
else:
    print("Using CPU")


def load_midi_details(directory):
    """Load all MIDI files in the directory and extract detailed note sequences."""
    all_sequences = []
    for filename in os.listdir(directory):
        if filename.endswith('.mid'):
            path = os.path.join(directory, filename)
            midi_data = pretty_midi.PrettyMIDI(path)
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    all_sequences.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'start': note.start,
                        'end': note.end
                    })
    return all_sequences

def create_input_target_sequences(sequence, seq_length):
    """Create input and target sequences from detailed note sequences."""
    input_sequences = []
    target_sequences = []
    for i in range(len(sequence) - seq_length):
        # input_seq = [[event['pitch'], event['velocity'], event['start'], event['end']] for event in sequence[i:i+seq_length]]
        # target_seq = [[event['pitch'], event['velocity'], event['start'], event['end']] for event in sequence[i+1:i+1+seq_length]]
        input_seq = [[event['pitch']] for event in sequence[i:i + seq_length]]
        target_pitch = sequence[i + seq_length]['pitch']
        target_seq = to_categorical(target_pitch, num_classes=vocab_size)
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    print(len(input_sequences))
    print(input_sequences[0])
    return np.array(input_sequences), np.array(target_sequences)

#%%
def build_model(seq_length, vocab_size, embedding_dim):
    inputs = Input(shape=(seq_length,))

    # Embedding layer: Converts input sequence of token indices to sequences of vectors
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length)(inputs)

    # LSTM layer: You can stack more LSTM layers or adjust the number of units
    x = LSTM(units=256, return_sequences=True)(x)  # return_sequences=True if next layer is also RNN
    x = LSTM(units=256)(x)  # Last LSTM layer does not return sequences

    # Output layer: Linear layer (Dense) with 'vocab_size' units to predict the next pitch
    outputs = Dense(vocab_size, activation='softmax')(x)  # Using softmax for output distribution

    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Directory containing MIDI files
directory = 'Data Sources/mozart/'

# Load and preprocess data

seq_length = 30  # Length of the input sequences
vocab_size = 128  # Number of unique pitches (for MIDI, typically 128)
embedding_dim = 100 # Define the length of sequences for input and target
sequences = load_midi_details(directory)
input_sequences, target_sequences = create_input_target_sequences(sequences, seq_length)


print("Input sequences shape:", input_sequences.shape)
print("Target sequences shape:", target_sequences.shape)

#%%


import matplotlib.pyplot as plt
import numpy as np

def midi_to_note(pitch):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return notes[pitch % 12] + str(pitch // 12 - 1)

pitches = [event['pitch'] for event in sequences]

note_counts = {}
for pitch in pitches:
    note = midi_to_note(pitch)
    if note in note_counts:
        note_counts[note] += 1
    else:
        note_counts[note] = 1

sorted_note_counts = {k: v for k, v in sorted(note_counts.items(), key=lambda item: item[1], reverse=True)}

plt.figure(figsize=(22, 6))
plt.bar(sorted_note_counts.keys(), sorted_note_counts.values(), color='skyblue', edgecolor='black')
plt.xlabel('Note')
plt.ylabel('Frequency')
plt.title('Note-wise Distribution')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.show()


#%%
# Recreate and compile the model
model = build_model(seq_length, vocab_size, embedding_dim)
model.summary()

model.fit(input_sequences, target_sequences, epochs=50, batch_size=32)

#%%
def generate_music(model, seed_sequence, length=10, steps_per_second=5, temperature=1):
    """Generate music from a seed sequence aiming for a total duration using a temperature parameter."""
    generated_sequence = np.copy(seed_sequence)  # Copy to avoid modifying the original seed
    total_steps = length * steps_per_second  # Total steps needed for desired duration

    for _ in range(total_steps):
        # Predict the next step using the last 'seq_length' elements in the generated_sequence
        prediction = model.predict(np.expand_dims(generated_sequence[-seq_length:], axis=0))[0]

        # Apply temperature to the prediction probabilities and normalize
        prediction = np.log(prediction + 1e-8) / temperature  # Smoothing and apply temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)

        # Sample an index from the probability array
        predicted_pitch = np.random.choice(len(prediction), p=prediction)

        # Append the predicted pitch to the generated sequence
        generated_sequence = np.vstack([generated_sequence, predicted_pitch])

    print("Generated sequence with variability:", generated_sequence[-30:])
    return generated_sequence


#%%
def generated_to_midi(generated_sequence, fs=100, total_duration=6):
    """Convert generated sequence to MIDI file, ensuring all notes are audible."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    # Calculate the duration per step based on total duration and number of notes
    duration_per_step = total_duration / len(generated_sequence)
    min_duration = 0.1  # Set a minimum note duration for clarity

    # Initialize the current time to track the start time of each note
    current_time = 0

    for step in generated_sequence:
        pitch = int(np.clip(step[0], 21, 108))  # Scale and clip pitch values to MIDI range
        velocity = 100  # Fixed velocity for all notes

        # Set start and end time for each note
        start = current_time
        end = start + max(min_duration, duration_per_step)

        # Create a MIDI note with the determined pitch, velocity, start, and end times
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end
        )
        instrument.notes.append(note)

        # Update the current time to the end of this note for the next note
        current_time = end

    # Add the instrument to the PrettyMIDI object
    pm.instruments.append(instrument)
    print("The generated MIDI")
    return pm

# Assuming you have a trained model and a seed sequence
# seed_pitches = np.array([60, 62, 64, 65, 67, 61, 45, 55, 47, 68])  # Example MIDI pitches
seed=1000
seed_pitches = input_sequences[seed]
# Normalize the pitches and reshape for model input
# Normalize by dividing by the maximum MIDI pitch value (127)
# Reshape to match expected model input shape (n, 1)
seed_sequence = (seed_pitches).reshape(-1, 1)  # Reshape to (n, 1)

length=10
steps_per_second=5
# Call the generate_music function with the reshaped and normalized seed sequence
generated_music = generate_music(model, seed_sequence, length=length, steps_per_second=steps_per_second,temperature=1)   # Generate enough steps
generated_music_midi = generated_to_midi(generated_music, total_duration=10)
generated_music_midi.write('mozart predicted2.mid')

def actual_song(total_duration):

    full_sequence=seed_sequence.copy()
    for i in range(length*steps_per_second):
        full_sequence = np.vstack([full_sequence,input_sequences[seed+i][-1]])
    # print(full_sequence[-30:])
    return full_sequence

full_sequence=actual_song(30)
#%%

generated_music_midi = generated_to_midi(full_sequence, total_duration=10)
generated_music_midi.write('mozart actual2.mid')

