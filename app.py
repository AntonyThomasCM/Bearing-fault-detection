import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("xgb_model_clean.pkl")
scaler = joblib.load("scaler.pkl")

# Define constants
Fs = 25600  # Sampling rate
bpfo_range = (95, 115)
bpfi_range = (150, 170)

# ---- Feature Extraction Functions ----
def extract_time_features(signal):
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)
    features['crest_factor'] = features['max'] / features['rms']
    features['impulse_factor'] = features['max'] / features['mean']
    return features

def extract_frequency_features(signal, Fs, bpfo_range, bpfi_range):
    N = len(signal)
    fft_vals = fft(signal)
    amplitudes = 2.0 / N * np.abs(fft_vals[:N // 2])
    frequencies = fftfreq(N, 1 / Fs)[:N // 2]

    features = {}

    # Dominant frequency
    dominant_idx = np.argmax(amplitudes)
    features['dominant_freq'] = frequencies[dominant_idx]
    features['dominant_amp'] = amplitudes[dominant_idx]

    # Top 5 peak frequencies
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(amplitudes, height=0.01)
    top_idx = np.argsort(amplitudes[peaks])[-5:]
    top_freqs = frequencies[peaks][top_idx]
    top_amps = amplitudes[peaks][top_idx]

    for i in range(len(top_freqs)):
        features[f'peak_freq_{i+1}'] = top_freqs[i]
        features[f'peak_amp_{i+1}'] = top_amps[i]

    # Band power near BPFO and BPFI
    bpfo_mask = (frequencies >= bpfo_range[0]) & (frequencies <= bpfo_range[1])
    features['band_power_bpfo'] = np.sum(amplitudes[bpfo_mask] ** 2)

    bpfi_mask = (frequencies >= bpfi_range[0]) & (frequencies <= bpfi_range[1])
    features['band_power_bpfi'] = np.sum(amplitudes[bpfi_mask] ** 2)

    # Spectral centroid
    features['spectral_centroid'] = np.sum(frequencies * amplitudes) / np.sum(amplitudes)

    return features

# ---- Streamlit UI ----
st.set_page_config(page_title="Bearing Fault Prediction", layout="wide")
st.title("🛠️ Bearing Fault Prediction using XGBoost")

uploaded_file = st.file_uploader("Upload vibration data file (.csv or .xlsx)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=None)
        else:
            df = pd.read_excel(uploaded_file, header=None)

        if df.shape[1] != 4:
            st.error("Uploaded file must have exactly 4 columns in the following order:\n"
                     "x_acceleration, y_acceleration, bearing_temperature, ambient_temperature.")
        else:
            # Assign columns
            df.columns = ["x_acc", "y_acc", "bearing_temp", "ambient_temp"]

            # Handle NaNs
            df = df.interpolate(method='linear', limit_direction='both')

            # Extract signals
            x_signal = df["x_acc"].values
            y_signal = df["y_acc"].values
            #bearing_temp = df["bearing_temp"].mean()
            #ambient_temp = df["ambient_temp"].mean()
            #temp_diff = bearing_temp - ambient_temp

            # Extract features for x and y signals
            time_x = extract_time_features(x_signal)
            freq_x = extract_frequency_features(x_signal, Fs, bpfo_range, bpfi_range)
            time_y = extract_time_features(y_signal)
            freq_y = extract_frequency_features(y_signal, Fs, bpfo_range, bpfi_range)

            x_feats = {f"x_{k}": v for k, v in {**time_x, **freq_x}.items()}
            y_feats = {f"y_{k}": v for k, v in {**time_y, **freq_y}.items()}
            combined = {**x_feats, **y_feats}

            # Create DataFrame and scale
            features_df = pd.DataFrame([combined])
            scaled = scaler.transform(features_df)

            # Predict
            prediction = model.predict(scaled)[0]
            proba = model.predict_proba(scaled)[0]

            # Decode class label
            label_map = {0: "degrading", 1: "faulty", 2: "normal"}
            predicted_label = label_map.get(prediction, "Unknown")

            st.success(f"🔍 Predicted Bearing Condition: **{predicted_label.upper()}**")
            st.write("📊 Prediction Probabilities:")
            st.dataframe(pd.DataFrame([proba], columns=[label_map[i] for i in range(3)]))

            st.subheader("🔎 Extracted Features")
            st.dataframe(features_df.T.rename(columns={0: "value"}))

    except Exception as e:
        st.error(f"Error processing the file: {e}")
