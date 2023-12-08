import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import neurokit2 as nk

# Function to generate one ECG waveform
def ecg_simulate(fs, signal_length, heart_rate, noise_level=0.0):
    # Time vector
    t = np.linspace(0, signal_length, signal_length * fs, endpoint=False)

    # Validate noise_level
    noise_level = np.clip(noise_level, 0.0, 1.0)

    # ECG signal components (P wave, QRS complex, T wave)
    p_wave = 0.15 * np.exp(-np.power(t - 0.2, 2) / (2 * np.power(0.03, 2)))
    qrs_complex = np.exp(-np.power(t - 0.3, 2) / (2 * np.power(0.09, 2)))
    t_wave = 0.35 * np.exp(-np.power(t - 0.45, 2) / (2 * np.power(0.12, 2)))

    # Combine components to make one heartbeat
    one_beat = p_wave + qrs_complex + t_wave

    # Calculate the number of beats in 'signal_length' seconds
    n_beats = int((heart_rate / 60) * signal_length)

    # Repeat the one_beat to simulate 'n_beats' beats in the ECG signal
    ecg_waveform = np.tile(one_beat, n_beats)

    # Trim the excess to match the desired signal length
    ecg_waveform = ecg_waveform[:t.size]

    # Add noise based on the noise_level
    max_noise_amplitude = 0.5  # This is an arbitrary choice for maximum noise amplitude
    noise = noise_level * max_noise_amplitude * np.random.normal(0, 1, size=t.size)
    ecg_waveform += noise
    
    # Add baseline wander for realism
    baseline_wander = 0.05 * np.sin(2 * np.pi * 0.05 * t)
    
    # Combine all parts to simulate a real ECG signal
    simulated_ecg = ecg_waveform + noise + baseline_wander

    return simulated_ecg

# Plot an example ECG waveform
fs = 250  # Hz
signal_length = 10  # Seconds
heart_rate = 60  # BPM
noise_level = 0.5  # Proportion of maximum noise amplitude
ecg_waveform = ecg_simulate(fs, signal_length, heart_rate, noise_level)
t = np.linspace(0, len(ecg_waveform) / fs, num=len(ecg_waveform))
plt.plot(t, ecg_waveform)
plt.title(f'ECG Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, len(ecg_waveform) / fs)
plt.ylim(-1.5, 1.5)
plt.show()
quality = nk.ecg_quality(ecg_waveform, sampling_rate=fs, method='zhao2018', approach='fuzzy')
print(quality)


# Plot the peaks
peaks, _ = find_peaks(ecg_waveform, height=0.5)
plt.plot(ecg_waveform)
plt.plot(peaks, ecg_waveform[peaks], "x")
plt.plot(np.zeros_like(ecg_waveform), "--", color="gray")
plt.show()

