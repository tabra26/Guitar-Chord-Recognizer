import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import queue

#model path and labels
MODEL_PATH = "../models/guitar_chord_recognition.h5"
CHORD_LABELS = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']

#load trained model
model = tf.keras.models.load_model(MODEL_PATH)

#audio settings
SAMPLE_RATE = 22050  #must match training data
BUFFER_SECONDS = 2  #analyze every 2 seconds
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SECONDS)  #buffer size in samples

q = queue.Queue()

def callback(indata, frames, time, status):
    """Callback function for real-time audio recording."""
    if status:
        print(status, flush=True)
    q.put(indata.copy())

def extract_features(audio):
    """Extracts MFCC and chroma features from audio."""
    
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)
    
    mfcc_mean = np.mean(mfcc, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    
    return np.hstack((mfcc_mean, chroma_mean)).reshape(1, -1)

def predict_chord(audio):
    """Predicts the chord from the audio segment."""
    
    features = extract_features(audio)
    prediction = model.predict(features)
    chord_class = np.argmax(prediction)
    
    return CHORD_LABELS[chord_class]

def listen():
    """Continuously captures system audio and predicts chords."""
    
    print("🎸 Listening for chords... Press Ctrl+C to stop.")
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=BUFFER_SIZE):
        buffer = np.zeros(BUFFER_SIZE)

        while True:
            
            while not q.empty():
                data = q.get()
                buffer = np.roll(buffer, -len(data))
                buffer[-len(data):] = data.flatten()
            
            #predict chord
            predicted_chord = predict_chord(buffer)
            print(f"🎶 Detected Chord: {predicted_chord}")

if __name__ == "__main__":
    listen()