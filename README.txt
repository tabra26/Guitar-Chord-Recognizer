Real-Time Guitar Chord Recognizer

Overview
This program listens to audio from your computer and predicts the currently played guitar chord in real-time using a trained machine learning model.

Installation
Install dependencies:
pip install numpy librosa tensorflow sounddevice

If using WSL, install PulseAudio:
sudo apt install pulseaudio
Start PulseAudio on Windows:
"C:\Program Files\PulseAudio\bin\pulseaudio.exe" --start --exit-idle-time=-1
Configure WSL:
echo "export PULSE_SERVER=tcp:$(hostname -I | awk '{print $1}')" >> ~/.bashrc
source ~/.bashrc

Running the Program
Run the real-time listener:
python src/realtime_predict.py

Training a Model
Preprocess the dataset:
python src/preprocess.py
Train the model:
python src/train.py

Troubleshooting
If no sound devices are found in WSL, install PulseAudio and check sounddevice.query_devices().
If the model only predicts one chord, check dataset balance and retrain.
If the listener fails after closing PulseAudio, restart PulseAudio and re-export PULSE_SERVER.

Next Steps
Improve the model, add a GUI, and expand chord recognition.