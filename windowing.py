import numpy as np

normal_scaled = np.load('normal_scaled.npy')
mixed_scaled  = np.load('mixed_scaled.npy')
mixed_labels  = np.load('mixed_labels.npy', allow_pickle=True)

WINDOW_SIZE = 50
STEP        = 1

def make_windows(data, window_size, step):
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i : i + window_size])
    return np.array(windows)

def make_label_windows(labels, window_size, step):
    window_labels = []
    for i in range(0, len(labels) - window_size + 1, step):
        window_labels.append(labels[i + window_size - 1])
    return np.array(window_labels)

normal_windows      = make_windows(normal_scaled, WINDOW_SIZE, STEP)
mixed_windows       = make_windows(mixed_scaled,  WINDOW_SIZE, STEP)
mixed_window_labels = make_label_windows(mixed_labels, WINDOW_SIZE, STEP)

print('normal_windows shape     :', normal_windows.shape)
print('mixed_windows  shape     :', mixed_windows.shape)
print('mixed_window_labels shape:', mixed_window_labels.shape)

np.save('normal_windows.npy',      normal_windows)
np.save('mixed_windows.npy',       mixed_windows)
np.save('mixed_window_labels.npy', mixed_window_labels)

print('All three window files saved successfully!')