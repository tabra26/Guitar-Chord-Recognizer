import os
import librosa
import numpy as np

#dataset paths
DATASET_PATH = "../data/archive"
TRAIN_PATH = os.path.join(DATASET_PATH, "training")
TEST_PATH = os.path.join(DATASET_PATH, "test")
PROCESSED_PATH = os.path.join(DATASET_PATH, "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)

#chord labels and mapping
CHORD_LABELS = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']
CHORD_TO_INDEX = {chord: i for i, chord in enumerate(CHORD_LABELS)}

#feature extraction parameters
SAMPLE_RATE = 22050
N_MFCC = 13

def extract_features(file_path):
    try:
        #load audio file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        #extract MFCC and chroma features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        #compute mean feature values
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        
        return np.hstack((mfcc_mean, chroma_mean))
    except Exception:
        return None

def process_dataset(folder_path, save_file):
    print(f"Processing {save_file} dataset...")
    data, labels = [], []
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    for chord in CHORD_LABELS:
        chord_path = os.path.join(folder_path, chord)
        if not os.path.exists(chord_path):
            continue

        for file in os.listdir(chord_path):
            file_path = os.path.join(chord_path, file)
            if not file.endswith(".wav"):
                continue
            
            features = extract_features(file_path)
            if features is not None:
                data.append(features)
                labels.append(CHORD_TO_INDEX[chord])
    
    if data:
        data, labels = np.array(data), np.array(labels)
        
        #save processed data
        np.save(os.path.join(PROCESSED_PATH, save_file + "_X.npy"), data)
        np.save(os.path.join(PROCESSED_PATH, save_file + "_y.npy"), labels)
        print(f"✅ {save_file} dataset processed and saved in {PROCESSED_PATH}")

if __name__ == "__main__":
    
    print(" Starting preprocessing...")
    process_dataset(TRAIN_PATH, "train")
    process_dataset(TEST_PATH, "test")
    print(f" Preprocessing complete! Processed files are saved in {PROCESSED_PATH}")
