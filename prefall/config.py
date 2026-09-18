"""Constants taken from the seminar deck (slides 6, 9, 13)."""
FS_RAW = 200      # simulated sensor output rate before decimation (real MPU-6050 runs up to 1 kHz)
FS = 50           # working sample rate (slide 9)
WIN = 50          # 1 s window (slide 9)
HOP = 25          # 50% overlap (slide 9)
LPF_HZ = 25.0     # 2nd-order Butterworth anti-alias cutoff (slide 9)
G = 9.81          # m/s^2
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]   # accel in g, gyro in deg/s
PREFALL_MIN_LEAD_S = 0.200   # useful pre-impact window is 200-400 ms (slide 2)
PREFALL_MAX_LEAD_S = 0.400
