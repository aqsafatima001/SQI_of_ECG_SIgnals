import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import wfdb
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def Z18_SQI(sample_ecg, sampling_rate):

    sample_filtered = sample_ecg
    # Apply bandpass filter
    try:
        sample_filtered = butter_bandpass_filter(sample_ecg, 1, 100, sampling_rate)
    except Exception as e:
        print(f"Error filtering sample {sample_ecg}: {e}")

    # Plot the filtered signal
    t = np.linspace(0, len(sample_filtered) / sampling_rate, num=len(sample_filtered))
    plt.figure(figsize=(10, 2))
    plt.plot(t, sample_filtered)
    plt.title(f'Filtered ECG Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, len(sample_filtered) / sampling_rate)
    plt.ylim(-1.5, 1.5)
    plt.show()

    # Compute the quality of the filtered ECG signal
    try:
        quality = nk.ecg_quality(sample_filtered, sampling_rate=sampling_rate, method='zhao2018', approach='fuzzy')
    except IndexError as e:
        print(f"Error computing quality for sample {sample_file}: {e}")
        return None

    return quality

# Function to read ECG signal from a .mat file
def read_ecg_from_physionet(record_path):
    # Read the record using WFDB
    record = wfdb.rdrecord(record_path)
    # Extract the signal and metadata
    ecg_signal = record.p_signal.flatten()
    sampling_rate = record.fs
    return ecg_signal, sampling_rate

# Function to add randomly add noise to an ecg waveform based on the noise level parameter specified
# Types of noise to be added:
# 1. Baseline wander
# 2. Powerline interference
# 3. Muscle artifact
# 4. Motion artifact
# 5. Gaussian noise
# 6. ECG signal dropout
# 7. ECG electrode motion artifact
# 8. ECG electrode contact noise (popout/popin)
# 9. ECG electrode disconnect/short circuit/open circuit
def noisify(ecg, noise_level, fs):
    # Time vector
    t = np.linspace(0, len(ecg), len(ecg) * fs, endpoint=False)

    # Validate noise_level
    noise_level = np.clip(noise_level, 0.0, 1.0)

    # Add baseline wander for realism
    baseline_wander = 0.05 * np.sin(2 * np.pi * 0.05 * len(ecg))

    # Add noise based on the noise_level
    max_noise_amplitude = 0.5  # This is an arbitrary choice for maximum noise amplitude
    noise = noise_level * max_noise_amplitude * np.random.normal(0, 1, size=len(ecg))

    # Combine all parts to simulate a real ECG signal
    simulated_ecg = ecg + noise + baseline_wander

    return simulated_ecg


# MAIN CODE: Generate noisy ECG signal
if __name__ == "__main__":    
    # Specify the directory where the ECG files are stored
    ecg_directory = "C:\\Users\\d_kar\\Downloads\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\\WFDBRecords\\01\\010"

    # List all the .hea files in the directory
    ecg_files = [f for f in os.listdir(ecg_directory) if os.path.splitext(f)[1] == '.hea']

    # Use the first 1 ECG files
    ecg_files = ecg_files[:1]

    # Read and plot each ECG record
    for ecg_file in ecg_files:
        # Construct the full path for the .hea file
        record_path = os.path.join(ecg_directory, os.path.splitext(ecg_file)[0])
        # Read the ECG signal
        ecg_signal, sampling_rate = read_ecg_from_physionet(record_path)
        clean = ecg_signal
        noisy = noisify(ecg_signal, 0.5, sampling_rate)

        # Calculate the percent difference between the clean and noisy signals
        percent_diff = np.abs((ecg_signal - clean) / clean) * 100
        print(f"Percent difference between clean and noisy signals: {percent_diff}")

        results = Z18_SQI(noisy, sampling_rate)
        print(results)