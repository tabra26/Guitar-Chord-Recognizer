import librosa
import numpy as np
import tensorflow as tf

#model and label definitions
MODEL_PATH = "../models/guitar_chord_recognition.h5"
CHORD_LABELS = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']

#load trained model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_chord(audio_file):
    """Loads an audio file, extracts features, and predicts a chord."""
    
    #load the audio file
    y, sr = librosa.load(audio_file, sr=22050)
    
    #extract MFCC and chroma features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    mfcc_mean = np.mean(mfcc, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    


    features = np.hstack((mfcc_mean, chroma_mean)).reshape(1, -1)
    


    #make prediction

    prediction = model.predict(features)
    



    #extract predicted class
    chord_class = np.argmax(prediction)
    
    return CHORD_LABELS[chord_class]

if __name__ == "__main__":
    


    test_audio = "../data/archive/test/Em/Em_Electric2_LInda_4..wav"
    


    #predict the chord
    predicted_chord = predict_chord(test_audio)
    
    #print the result
    print(f"Predicted Chord: {predicted_chord}")
