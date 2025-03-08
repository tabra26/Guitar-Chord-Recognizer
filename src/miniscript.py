import numpy as np

y_train = np.load("../data/archive/processed/train_y.npy")

# Count occurrences of each chord
unique, counts = np.unique(y_train, return_counts=True)
for chord, count in zip(unique, counts):
    print(f"Chord {chord}: {count} samples")
