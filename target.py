import numpy as np
import matplotlib.pyplot as plt

f = 1* (10**9)
fs = 1.33*(10**9)
t2 = np.arange(0, 15000/f, 1/fs)

# Generate signals
y1 = np.sin(2*np.pi*f*t2)
y2 = np.zeros(len(t2))

for i in range(len(t2)):
    if t2[i] <= 5e-6:
        y2[i] = 1
    else:
        y2[i] = 0

# Convolution
y = y1*y2

# Signal-to-noise ratio (SNR) calculation
snr_db = 0
snr = 10 ** (snr_db / 10)
Power = 10 * np.log10(1 / (2 * snr))

# Generate noise
A = y.shape[0]
noise = np.random.normal(scale=10**(Power/10), size=(A,))
y3 = y + noise

# Process the detected signal
y4 = y3[0:2048]
y5 = np.concatenate((np.zeros(10), y4, np.zeros(10)))
t = np.zeros_like(y4)
detected_signal = []

for n in range(len(y4)):
    s1 = y5[n:n+10]
    s2 = y5[n+11:n+21]
    m1 = np.mean(np.abs(s1))
    m2 = np.mean(np.abs(s2))
    m = m1 + m2 / 2
    t[n] = m

    if y4[n] > m:
        detected_signal.append(y4[n])
    else:
        detected_signal.append(0)

plt.figure(figsize=(20,10))
plt.plot(np.abs(y4))
plt.plot(np.abs(t), 'r')
plt.plot(np.abs(detected_signal), 'go')
plt.title('Adaptive Threshold Mapping')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(['Noisy Signal', 'Threshold', 'Detected Signal'])
plt.show()
