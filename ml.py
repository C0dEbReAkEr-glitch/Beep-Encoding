#!/usr/bin/env python3
"""
Enhanced Audio Steganography Analysis Tool - Streamlit Application
FIXES: FSK/Morse decoding, ML model persistence, comprehensive encoding support
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import wave
import json
import pickle
import os
from datetime import datetime
from typing import Tuple, List, Dict, Optional
try:
    import scipy.signal
    from scipy.fft import fft, fftfreq
    from scipy.stats import pearsonr
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SCIPY_AVAILABLE = True
    ML_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ML_AVAILABLE = False
    st.warning("scipy and/or sklearn not available. Some advanced features may be limited.")
import base64

# Persistent data directory
DATA_DIR = "steg_data"
MODEL_FILE = os.path.join(DATA_DIR, "ml_model.pkl")
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "training_data.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class MLEncodingClassifier:
    """Machine Learning classifier for automatic encoding type detection"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.feature_names = [
            'num_frequencies', 'freq_variance', 'amplitude_variance', 'zero_crossing_rate',
            'spectral_centroid', 'spectral_rolloff', 'rms_energy', 'duration',
            'peak_freq_1', 'peak_freq_2', 'freq_ratio', 'timing_variance',
            'signal_segments', 'silence_ratio'
        ]
        # Load existing data on initialization
        self.load_training_data()
        if len(self.training_data) >= 5:
            self.train_model()
    
    def extract_features(self, signal: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        """Extract features from audio signal for ML classification"""
        if len(signal) == 0:
            return np.zeros(len(self.feature_names))
        
        # Basic signal properties
        duration = len(signal) / sample_rate
        rms_energy = np.sqrt(np.mean(signal ** 2))
        
        # Frequency analysis
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(fft_result)//2]
        
        # Find significant frequencies
        threshold = np.max(magnitude) * 0.1
        peak_indices = []
        for i in range(1, len(magnitude)-1):
            if magnitude[i] > threshold and magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                if 50 < freqs[i] < 3000:
                    peak_indices.append(i)
        
        significant_freqs = sorted([freqs[i] for i in peak_indices])
        num_frequencies = len(significant_freqs)
        freq_variance = np.var(significant_freqs) if len(significant_freqs) > 1 else 0
        
        # Peak frequencies
        peak_freq_1 = significant_freqs[0] if len(significant_freqs) > 0 else 0
        peak_freq_2 = significant_freqs[1] if len(significant_freqs) > 1 else 0
        freq_ratio = peak_freq_2 / peak_freq_1 if peak_freq_1 > 0 and peak_freq_2 > 0 else 1
        
        # Amplitude analysis
        amplitude_variance = np.var(np.abs(signal))
        
        # Zero crossing rate
        zero_crossings = np.diff(np.signbit(signal)).sum()
        zero_crossing_rate = zero_crossings / len(signal)
        
        # Spectral features
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        cumsum_magnitude = np.cumsum(magnitude)
        spectral_rolloff = freqs[np.where(cumsum_magnitude >= 0.85 * cumsum_magnitude[-1])[0][0]] if len(cumsum_magnitude) > 0 else 0
        
        # Timing analysis
        threshold_timing = np.max(np.abs(signal)) * 0.01
        segments = []
        is_signal = False
        segment_start = 0
        
        for i, sample in enumerate(signal):
            amplitude = abs(sample)
            if not is_signal and amplitude > threshold_timing:
                is_signal = True
                segment_start = i
            elif is_signal and amplitude <= threshold_timing:
                segments.append((segment_start, i))
                is_signal = False
        
        signal_segments = len(segments)
        silence_ratio = 1 - (sum(end - start for start, end in segments) / len(signal)) if segments else 0
        
        # Timing variance
        segment_durations = [(end - start) / sample_rate for start, end in segments]
        timing_variance = np.var(segment_durations) if len(segment_durations) > 1 else 0
        
        features = np.array([
            num_frequencies, freq_variance, amplitude_variance, zero_crossing_rate,
            spectral_centroid, spectral_rolloff, rms_energy, duration,
            peak_freq_1, peak_freq_2, freq_ratio, timing_variance,
            signal_segments, silence_ratio
        ])
        
        return features
    
    def add_training_data(self, signal: np.ndarray, encoding_type: str, sample_rate: int = 44100):
        """Add new training data to the classifier"""
        features = self.extract_features(signal, sample_rate)
        self.training_data.append({
            'features': features.tolist(),  # Convert to list for JSON serialization
            'label': encoding_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save training data
        self.save_training_data()
        
        # Retrain if we have enough data
        if len(self.training_data) >= 5:
            self.train_model()
    
    def train_model(self):
        """Train the ML model with collected data"""
        if len(self.training_data) < 5:
            return False
        
        try:
            # Prepare training data
            X = np.array([item['features'] for item in self.training_data])
            y = np.array([item['label'] for item in self.training_data])
            
            # Handle NaN/inf values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Save trained model
            self.save_model()
            
            return True
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return False
    
    def predict(self, signal: np.ndarray, sample_rate: int = 44100) -> Tuple[str, float]:
        """Predict encoding type with confidence"""
        if not self.is_trained:
            return "unknown", 0.0
        
        try:
            features = self.extract_features(signal, sample_rate).reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            features_scaled = self.scaler.transform(features)
            
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return "unknown", 0.0
    
    def save_model(self):
        """Save the trained model"""
        if self.is_trained:
            try:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names
                }
                with open(MODEL_FILE, 'wb') as f:
                    pickle.dump(model_data, f)
            except Exception as e:
                st.error(f"Model saving error: {str(e)}")
    
    def load_model(self):
        """Load a trained model"""
        try:
            if os.path.exists(MODEL_FILE):
                with open(MODEL_FILE, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data.get('feature_names', self.feature_names)
                self.is_trained = True
                return True
        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
        return False
    
    def save_training_data(self):
        """Save training data to JSON file"""
        try:
            with open(TRAINING_DATA_FILE, 'w') as f:
                json.dump(self.training_data, f, indent=2)
        except Exception as e:
            st.error(f"Training data saving error: {str(e)}")
    
    def load_training_data(self):
        """Load training data from JSON file"""
        try:
            if os.path.exists(TRAINING_DATA_FILE):
                with open(TRAINING_DATA_FILE, 'r') as f:
                    self.training_data = json.load(f)
                
                # Try to load model if it exists
                self.load_model()
                return True
        except Exception as e:
            st.error(f"Training data loading error: {str(e)}")
        return False

class AudioSteganalysisSystem:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.ml_classifier = MLEncodingClassifier() if ML_AVAILABLE else None
        
        # Morse Code Dictionary
        self.morse_code = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
            'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
            'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
            'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
            'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
            '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
            '8': '---..', '9': '----.', ' ': '/'
        }
        
        # Reverse morse dictionary
        self.reverse_morse = {v: k for k, v in self.morse_code.items()}
        
        # DTMF Frequencies
        self.dtmf_freqs = {
            '1': [697, 1209], '2': [697, 1336], '3': [697, 1477], 'A': [697, 1633],
            '4': [770, 1209], '5': [770, 1336], '6': [770, 1477], 'B': [770, 1633],
            '7': [852, 1209], '8': [852, 1336], '9': [852, 1477], 'C': [852, 1633],
            '*': [941, 1209], '0': [941, 1336], '#': [941, 1477], 'D': [941, 1633]
        }
        
        # Reverse DTMF lookup
        self.reverse_dtmf = {}
        for char, (f1, f2) in self.dtmf_freqs.items():
            self.reverse_dtmf[(f1, f2)] = char
    
    def generate_tone(self, frequency: float, duration: float, amplitude: float = 0.3, phase: float = 0) -> np.ndarray:
        """Generate a pure tone"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    def generate_dual_tone(self, freq1: float, freq2: float, duration: float, amplitude: float = 0.3) -> np.ndarray:
        """Generate dual-tone signal (for DTMF)"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone1 = (amplitude * 0.5) * np.sin(2 * np.pi * freq1 * t)
        tone2 = (amplitude * 0.5) * np.sin(2 * np.pi * freq2 * t)
        return tone1 + tone2
    
    def generate_silence(self, duration: float) -> np.ndarray:
        """Generate silence"""
        return np.zeros(int(self.sample_rate * duration))
    
    def text_to_binary(self, text: str) -> str:
        """Convert text to binary string"""
        return ''.join(format(ord(char), '08b') for char in text)
    
    def binary_to_text(self, binary: str) -> str:
        """Convert binary string to text - IMPROVED"""
        if not binary:
            return ""
        
        # Pad binary to multiple of 8
        if len(binary) % 8 != 0:
            binary = binary.ljust((len(binary) + 7) // 8 * 8, '0')
        
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                char_code = int(byte, 2)
                # Only add printable ASCII characters
                if 32 <= char_code <= 126:
                    text += chr(char_code)
                elif char_code == 0:
                    break  # Stop at null terminator
        return text
    
    def analyze_frequencies(self, signal: np.ndarray, window_size: int = 4096) -> List[float]:
        """Analyze frequencies in the signal using FFT - IMPROVED"""
        if len(signal) < window_size:
            window_size = len(signal)
        
        # Use windowing for better frequency resolution
        windowed_signal = signal * np.hanning(len(signal))
        
        # Perform FFT
        fft_result = np.fft.fft(windowed_signal, n=max(window_size, len(signal)))
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        freqs = np.fft.fftfreq(len(fft_result), 1/self.sample_rate)[:len(fft_result)//2]
        
        # Find peaks with better algorithm
        if SCIPY_AVAILABLE:
            peaks, properties = scipy.signal.find_peaks(
                magnitude, 
                height=np.max(magnitude) * 0.1,
                distance=10
            )
        else:
            threshold = np.max(magnitude) * 0.1
            peaks = []
            for i in range(2, len(magnitude)-2):
                if (magnitude[i] > threshold and 
                    magnitude[i] > magnitude[i-1] and 
                    magnitude[i] > magnitude[i+1]):
                    peaks.append(i)
        
        # Extract significant frequencies
        frequencies = []
        for peak_idx in peaks:
            freq = freqs[peak_idx]
            if 50 < freq < 3000:
                frequencies.append(round(freq, 1))
        
        return sorted(list(set(frequencies)))
    
    def encode_dtmf(self, message: str, tone_duration: float = 0.2, gap_duration: float = 0.1, amplitude: float = 0.5) -> np.ndarray:
        """Encode message as DTMF"""
        signal_parts = []
        
        for char in message.upper():
            if char == ' ':
                signal_parts.append(self.generate_silence(gap_duration * 3))
            elif char in self.dtmf_freqs:
                freq1, freq2 = self.dtmf_freqs[char]
                dual_tone = self.generate_dual_tone(freq1, freq2, tone_duration, amplitude)
                signal_parts.append(dual_tone)
                signal_parts.append(self.generate_silence(gap_duration))
            else:
                signal_parts.append(self.generate_silence(gap_duration))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def decode_dtmf(self, signal: np.ndarray, tone_duration: float = 0.2, gap_duration: float = 0.1) -> str:
        """Decode DTMF from signal - IMPROVED"""
        if len(signal) == 0:
            return ""
        
        decoded = ""
        window_size = int(self.sample_rate * tone_duration)
        step_size = int(self.sample_rate * (tone_duration + gap_duration) * 0.8)  # Overlapping windows
        
        for i in range(0, len(signal) - window_size, step_size):
            window = signal[i:i + window_size]
            
            # Skip if window is too quiet
            if np.max(np.abs(window)) < 0.01:
                continue
            
            # Analyze frequencies in this window
            frequencies = self.analyze_frequencies(window)
            
            if len(frequencies) >= 2:
                # Find best DTMF match
                best_match = None
                min_error = float('inf')
                
                for char, (target_f1, target_f2) in self.dtmf_freqs.items():
                    for detected_f1 in frequencies:
                        for detected_f2 in frequencies:
                            if detected_f1 != detected_f2:
                                error1 = abs(detected_f1 - target_f1) + abs(detected_f2 - target_f2)
                                error2 = abs(detected_f1 - target_f2) + abs(detected_f2 - target_f1)
                                error = min(error1, error2)
                                
                                if error < min_error and error < 80:  # Increased tolerance
                                    min_error = error
                                    best_match = char
                
                if best_match and (not decoded or decoded[-1] != best_match):
                    decoded += best_match
        
        return decoded
    
    def classify_encoding(self, frequencies: List[float], signal: np.ndarray, timing: List[Dict]) -> str:
        """Classify encoding type - IMPROVED"""
        
        # Try ML prediction first if available and trained
        if self.ml_classifier and self.ml_classifier.is_trained:
            ml_prediction, confidence = self.ml_classifier.predict(signal, self.sample_rate)
            if confidence > 0.6:  # Lowered threshold
                return ml_prediction
        
        # Fallback to rule-based classification
        freq_count = len(frequencies)
        signal_segments = [s for s in timing if s['type'] == 'signal']
        
        # Enhanced DTMF detection
        if freq_count >= 2:
            dtmf_pairs = 0
            for freq in frequencies:
                dtmf_match = any(abs(freq - dtmf_freq) < 80 
                               for dtmf_freqs in self.dtmf_freqs.values() 
                               for dtmf_freq in dtmf_freqs)
                if dtmf_match:
                    dtmf_pairs += 1
            
            if dtmf_pairs >= 2:
                return 'dtmf'
        
        # FSK detection (two distinct frequencies)
        if freq_count == 2:
            ratio = max(frequencies) / min(frequencies) if min(frequencies) > 0 else 1
            if 1.2 < ratio < 3.0:
                return 'fsk'
        
        # Morse detection (timing patterns with single/few frequencies)
        if len(signal_segments) > 2 and freq_count <= 3:
            durations = [s.get('duration', 0) for s in signal_segments]
            if durations and len([d for d in durations if d > 0]) > 1:
                duration_cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
                if duration_cv > 0.2:
                    return 'morse'
        
        # ASK detection (amplitude modulation)
        rms_values = []
        window_size = max(1024, len(signal) // 50)
        for i in range(0, len(signal) - window_size, window_size):
            rms = np.sqrt(np.mean(signal[i:i+window_size]**2))
            rms_values.append(rms)
        
        amplitude_variance = np.var(rms_values) / np.mean(rms_values) if np.mean(rms_values) > 0 else 0
        
        if freq_count <= 2 and amplitude_variance > 0.2:
            return 'ask'
        
        # PSK detection (single frequency with phase changes)
        if freq_count == 1:
            return 'psk'
        
        # Manchester detection (regular transitions)
        if len(signal_segments) > 5:
            durations = [s.get('duration', 0) for s in signal_segments]
            if durations:
                duration_std = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
                if duration_std < 0.3:
                    return 'manchester'
        
        return 'morse'  # Default fallback
    
    def analyze_timing(self, signal: np.ndarray, threshold: float = 0.02) -> List[Dict]:
        """Analyze timing patterns in the signal - IMPROVED"""
        segments = []
        is_signal = False
        segment_start = 0
        
        # Use adaptive threshold
        signal_rms = np.sqrt(np.mean(signal ** 2))
        adaptive_threshold = max(threshold, signal_rms * 0.05)  # Lowered threshold
        
        # Smooth signal to reduce noise
        if len(signal) > 100:
            smoothed = np.convolve(np.abs(signal), np.ones(50)/50, mode='same')
        else:
            smoothed = np.abs(signal)
        
        for i, amplitude in enumerate(smoothed):
            if not is_signal and amplitude > adaptive_threshold:
                # Start of signal
                if segments and segments[-1]['type'] == 'silence':
                    segments[-1]['duration'] = (i - segment_start) / self.sample_rate
                is_signal = True
                segment_start = i
            elif is_signal and amplitude <= adaptive_threshold:
                # End of signal
                duration = (i - segment_start) / self.sample_rate
                if duration > 0.005:  # Minimum segment duration (5ms)
                    segments.append({
                        'type': 'signal',
                        'start': segment_start / self.sample_rate,
                        'duration': duration
                    })
                is_signal = False
                segment_start = i
                segments.append({'type': 'silence', 'start': segment_start / self.sample_rate})
        
        # Handle final segment
        if is_signal:
            duration = (len(signal) - segment_start) / self.sample_rate
            if duration > 0.005:
                segments.append({
                    'type': 'signal',
                    'start': segment_start / self.sample_rate,
                    'duration': duration
                })
        
        return segments
    
    def decode_morse(self, timing: List[Dict]) -> str:
        """Decode Morse code from timing analysis - COMPLETELY REWRITTEN"""
        signal_segments = [s for s in timing if s['type'] == 'signal' and 'duration' in s]
        silence_segments = [s for s in timing if s['type'] == 'silence' and 'duration' in s]
        
        if not signal_segments:
            return ""
        
        durations = [s['duration'] for s in signal_segments if s['duration'] > 0]
        if not durations:
            return ""
        
        # Improved threshold calculation using clustering
        sorted_durations = sorted(durations)
        
        if len(sorted_durations) >= 3:
            # Use k-means like approach to find dot/dash threshold
            short_group = []
            long_group = []
            
            median_duration = np.median(sorted_durations)
            
            for d in sorted_durations:
                if d <= median_duration:
                    short_group.append(d)
                else:
                    long_group.append(d)
            
            if short_group and long_group:
                dot_threshold = (np.mean(short_group) + np.mean(long_group)) / 2
            else:
                dot_threshold = median_duration
        else:
            dot_threshold = np.mean(sorted_durations) if sorted_durations else 0.1
        
        # Process timing sequence
        morse_chars = []
        current_char = ""
        
        # Combine timing information
        combined_sequence = []
        for i, segment in enumerate(timing):
            if segment['type'] == 'signal' and 'duration' in segment:
                combined_sequence.append(('signal', segment['duration']))
            elif segment['type'] == 'silence' and 'duration' in segment:
                combined_sequence.append(('silence', segment['duration']))
        
        for i, (seg_type, duration) in enumerate(combined_sequence):
            if seg_type == 'signal':
                # Determine if it's a dot or dash
                if duration < dot_threshold:
                    current_char += '.'
                else:
                    current_char += '-'
                    
            elif seg_type == 'silence' and current_char:
                # Determine type of gap
                if duration > dot_threshold * 2.5:  # Word break
                    morse_chars.append(current_char)
                    morse_chars.append('/')  # Word separator
                    current_char = ""
                elif duration > dot_threshold * 0.8:  # Character break
                    morse_chars.append(current_char)
                    current_char = ""
        
        # Don't forget the last character
        if current_char:
            morse_chars.append(current_char)
        
        # Convert morse to text
        decoded_text = ""
        for morse_char in morse_chars:
            if morse_char == '/':
                decoded_text += ' '
            elif morse_char in self.reverse_morse:
                decoded_text += self.reverse_morse[morse_char]
            else:
                decoded_text += '?'  # Unknown morse pattern
        
        return decoded_text.strip()
    
    def decode_fsk(self, signal: np.ndarray, frequencies: List[float], bit_duration: float = 0.1) -> str:
        """Decode FSK signal - COMPLETELY REWRITTEN"""
        if len(frequencies) < 2 or len(signal) == 0:
            return ""
        
        # Use the two most prominent frequencies
        freq0, freq1 = sorted(frequencies[:2])
        window_size = int(self.sample_rate * bit_duration)
        
        if window_size <= 0:
            return ""
        
        binary_string = ""
        
        # Process signal in overlapping windows for better detection
        overlap = window_size // 4
        
        for i in range(0, len(signal) - window_size, window_size - overlap):
            window = signal[i:i + window_size]
            
            # Skip very quiet segments
            if np.max(np.abs(window)) < 0.01:
                continue
            
            # Calculate power at each frequency using FFT
            fft_result = np.fft.fft(window)
            magnitude = np.abs(fft_result[:len(fft_result)//2])
            freqs = np.fft.fftfreq(len(window), 1/self.sample_rate)[:len(fft_result)//2]
            
            # Find power at target frequencies
            power0 = 0
            power1 = 0
            
            tolerance = 50  # Hz tolerance
            
            for j, freq in enumerate(freqs):
                if abs(freq - freq0) < tolerance:
                    power0 += magnitude[j] ** 2
                elif abs(freq - freq1) < tolerance:
                    power1 += magnitude[j] ** 2
            
            # Alternative method: correlation with reference signals
            if power0 == 0 and power1 == 0:
                t = np.linspace(0, bit_duration, window_size, False)
                ref0 = np.sin(2 * np.pi * freq0 * t)
                ref1 = np.sin(2 * np.pi * freq1 * t)
                
                corr0 = abs(np.corrcoef(window, ref0)[0, 1]) if len(window) == len(ref0) else 0
                corr1 = abs(np.corrcoef(window, ref1)[0, 1]) if len(window) == len(ref1) else 0
                
                # Use correlation to determine bit
                binary_string += '0' if corr0 > corr1 else '1'
            else:
                # Use power to determine bit
                binary_string += '0' if power0 > power1 else '1'
        
        # Clean up binary string
        if not binary_string:
            return ""
        
        # Remove potential noise bits at the beginning/end
        binary_string = binary_string.strip('01')
        
        return self.binary_to_text(binary_string)
    
    def decode_ask(self, signal: np.ndarray, bit_duration: float = 0.1) -> str:
        """Decode ASK signal - IMPROVED"""
        if len(signal) == 0:
            return ""
            
        window_size = int(self.sample_rate * bit_duration)
        if window_size <= 0:
            return ""
            
        binary_string = ""
        
        # Calculate overall signal statistics
        overall_rms = np.sqrt(np.mean(signal ** 2))
        
        # Use adaptive threshold based on signal distribution
        amplitudes = []
        for i in range(0, len(signal) - window_size, window_size):
            window = signal[i:i + window_size]
            rms_amplitude = np.sqrt(np.mean(window ** 2))
            amplitudes.append(rms_amplitude)
        
        if not amplitudes:
            return ""
        
        # Use median as threshold
        threshold = np.median(amplitudes)
        
        for amplitude in amplitudes:
            binary_string += '1' if amplitude > threshold else '0'
        
        return self.binary_to_text(binary_string)
    
    def decode_psk(self, signal: np.ndarray, base_freq: float = 800, bit_duration: float = 0.1) -> str:
        """Decode PSK signal - IMPROVED"""
        if len(signal) == 0:
            return ""
            
        window_size = int(self.sample_rate * bit_duration)
        if window_size <= 0:
            return ""
            
        binary_string = ""
        prev_phase = None
        
        for i in range(0, len(signal) - window_size, window_size):
            window = signal[i:i + window_size]
            
            # Skip quiet segments
            if np.max(np.abs(window)) < 0.01:
                continue
            
            # Extract phase using correlation with reference signals
            t = np.linspace(0, bit_duration, window_size, False)
            cos_ref = np.cos(2 * np.pi * base_freq * t)
            sin_ref = np.sin(2 * np.pi * base_freq * t)
            
            cos_corr = np.sum(window * cos_ref)
            sin_corr = np.sum(window * sin_ref)
            
            current_phase = np.arctan2(sin_corr, cos_corr)
            
            if prev_phase is not None:
                phase_diff = current_phase - prev_phase
                # Normalize phase difference
                while phase_diff > np.pi:
                    phase_diff -= 2 * np.pi
                while phase_diff < -np.pi:
                    phase_diff += 2 * np.pi
                
                # Determine bit based on phase change
                binary_string += '1' if abs(phase_diff) > np.pi/2 else '0'
            else:
                binary_string += '0'  # First bit assumption
            
            prev_phase = current_phase
        
        return self.binary_to_text(binary_string)
    
    def decode_manchester(self, signal: np.ndarray, base_freq: float = 800, bit_duration: float = 0.1) -> str:
        """Decode Manchester encoded signal - IMPROVED"""
        if len(signal) == 0:
            return ""
            
        half_window = int(self.sample_rate * bit_duration / 2)
        if half_window <= 0:
            return ""
            
        binary_string = ""
        
        for i in range(0, len(signal) - half_window * 2, half_window * 2):
            first_half = signal[i:i + half_window]
            second_half = signal[i + half_window:i + half_window * 2]
            
            first_rms = np.sqrt(np.mean(first_half ** 2))
            second_rms = np.sqrt(np.mean(second_half ** 2))
            
            # Manchester: '1' = low-to-high, '0' = high-to-low
            if first_rms < second_rms:
                binary_string += '1'
            else:
                binary_string += '0'
        
        return self.binary_to_text(binary_string)
    
    def encode_morse(self, message: str, base_freq: float = 800, dot_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as Morse code"""
        dash_duration = dot_duration * 3
        gap_duration = dot_duration
        word_gap_duration = dot_duration * 7
        
        signal_parts = []
        
        for char in message.upper():
            if char in self.morse_code:
                morse_pattern = self.morse_code[char]
                
                if morse_pattern == '/':
                    signal_parts.append(self.generate_silence(word_gap_duration))
                    continue
                
                for symbol in morse_pattern:
                    if symbol == '.':
                        signal_parts.append(self.generate_tone(base_freq, dot_duration, amplitude))
                    elif symbol == '-':
                        signal_parts.append(self.generate_tone(base_freq, dash_duration, amplitude))
                    
                    signal_parts.append(self.generate_silence(gap_duration))
                
                signal_parts.append(self.generate_silence(dash_duration))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def encode_fsk(self, message: str, base_freq: float = 800, bit_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as FSK"""
        binary = self.text_to_binary(message)
        freq0 = base_freq
        freq1 = base_freq * 1.5
        
        signal_parts = []
        
        for bit in binary:
            freq = freq0 if bit == '0' else freq1
            signal_parts.append(self.generate_tone(freq, bit_duration, amplitude))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def encode_ask(self, message: str, base_freq: float = 800, bit_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as ASK"""
        binary = self.text_to_binary(message)
        signal_parts = []
        
        for bit in binary:
            bit_amplitude = amplitude if bit == '1' else amplitude * 0.1
            signal_parts.append(self.generate_tone(base_freq, bit_duration, bit_amplitude))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def encode_psk(self, message: str, base_freq: float = 800, bit_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as PSK"""
        binary = self.text_to_binary(message)
        signal_parts = []
        phase = 0
        
        for bit in binary:
            if bit == '1':
                phase = np.pi if phase == 0 else 0
            signal_parts.append(self.generate_tone(base_freq, bit_duration, amplitude, phase))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def encode_manchester(self, message: str, base_freq: float = 800, bit_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as Manchester encoding"""
        binary = self.text_to_binary(message)
        signal_parts = []
        half_duration = bit_duration / 2
        
        for bit in binary:
            if bit == '1':
                signal_parts.append(self.generate_tone(base_freq, half_duration, amplitude * 0.3))
                signal_parts.append(self.generate_tone(base_freq, half_duration, amplitude))
            else:
                signal_parts.append(self.generate_tone(base_freq, half_duration, amplitude))
                signal_parts.append(self.generate_tone(base_freq, half_duration, amplitude * 0.3))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def calculate_ber(self, signal: np.ndarray, num_bands: int = 8) -> Dict:
        """Calculate Band Energy Ratio analysis"""
        # Perform FFT
        fft_result = fft(signal)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        freqs = fftfreq(len(signal), 1/self.sample_rate)[:len(fft_result)//2]
        
        # Define frequency bands
        max_freq = self.sample_rate // 2
        band_edges = np.linspace(0, max_freq, num_bands + 1)
        
        band_energies = []
        band_info = []
        
        for i in range(num_bands):
            band_start = band_edges[i]
            band_end = band_edges[i + 1]
            
            # Find indices in this band
            band_mask = (freqs >= band_start) & (freqs < band_end)
            band_energy = np.sum(magnitude[band_mask] ** 2)
            
            band_energies.append(band_energy)
            band_info.append({
                'band': i + 1,
                'freq_range': f"{band_start:.0f}-{band_end:.0f} Hz",
                'energy': band_energy,
                'peak_freq': freqs[band_mask][np.argmax(magnitude[band_mask])] if np.any(band_mask) else 0
            })
        
        total_energy = sum(band_energies)
        band_ratios = [energy / total_energy if total_energy > 0 else 0 for energy in band_energies]
        
        return {
            'band_energies': band_energies,
            'band_ratios': band_ratios,
            'band_info': band_info,
            'total_energy': total_energy
        }
    
    def calculate_correlation(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict:
        """Calculate statistical correlation between two signals"""
        # Ensure signals have same length
        min_len = min(len(signal1), len(signal2))
        s1 = signal1[:min_len]
        s2 = signal2[:min_len]
        
        # Pearson correlation
        if SCIPY_AVAILABLE:
            pearson_corr, pearson_p = pearsonr(s1, s2)
        else:
            # Manual calculation
            mean1, mean2 = np.mean(s1), np.mean(s2)
            pearson_corr = np.sum((s1 - mean1) * (s2 - mean2)) / np.sqrt(np.sum((s1 - mean1)**2) * np.sum((s2 - mean2)**2))
            pearson_p = 0.0
        
        # Cross-correlation
        cross_corr = np.correlate(s1, s2, mode='full')
        max_cross_corr = np.max(cross_corr) / (np.linalg.norm(s1) * np.linalg.norm(s2))
        
        # RMS difference
        rms_diff = np.sqrt(np.mean((s1 - s2) ** 2))
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(s1 ** 2)
        noise_power = np.mean((s1 - s2) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'max_cross_correlation': max_cross_corr,
            'rms_difference': rms_diff,
            'snr_db': snr_db
        }
    
    def calculate_embedding_capacity(self, message: str, signal: np.ndarray, encoding_type: str) -> Dict:
        """Calculate embedding capacity metrics"""
        message_bits = len(self.text_to_binary(message))
        message_chars = len(message)
        signal_duration = len(signal) / self.sample_rate
        signal_samples = len(signal)
        
        # Calculate different capacity metrics
        bits_per_second = message_bits / signal_duration if signal_duration > 0 else 0
        bits_per_sample = message_bits / signal_samples if signal_samples > 0 else 0
        chars_per_second = message_chars / signal_duration if signal_duration > 0 else 0
        
        # Efficiency based on encoding type
        efficiency_map = {
            'morse': 0.3,  # Variable length, inefficient
            'dtmf': 0.4,   # Fixed symbols, moderate
            'fsk': 0.8,    # Direct binary, efficient
            'ask': 0.8,    # Direct binary, efficient
            'psk': 0.9,    # Phase encoding, very efficient
            'manchester': 0.5  # Doubled data rate, moderate
        }
        
        theoretical_efficiency = efficiency_map.get(encoding_type.lower(), 0.5)
        embedding_efficiency = (message_bits / (signal_samples * theoretical_efficiency)) * 100 if signal_samples > 0 else 0
        
        return {
            'message_bits': message_bits,
            'message_chars': message_chars,
            'signal_duration': signal_duration,
            'signal_samples': signal_samples,
            'bits_per_second': bits_per_second,
            'bits_per_sample': bits_per_sample,
            'chars_per_second': chars_per_second,
            'embedding_efficiency_percent': embedding_efficiency,
            'theoretical_efficiency': theoretical_efficiency
        }
    
    def create_waveform_plot(self, original_signal: np.ndarray, stego_signal: np.ndarray = None) -> go.Figure:
        """Create waveform comparison plot"""
        duration = len(original_signal) / self.sample_rate
        time_axis = np.linspace(0, duration, len(original_signal))
        
        fig = make_subplots(
            rows=2 if stego_signal is not None else 1,
            cols=1,
            subplot_titles=['Original Signal'] + (['Steganographic Signal'] if stego_signal is not None else []),
            vertical_spacing=0.1
        )
        
        # Original signal
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=original_signal,
                mode='lines',
                name='Original',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Stego signal if provided
        if stego_signal is not None:
            stego_time = np.linspace(0, len(stego_signal) / self.sample_rate, len(stego_signal))
            fig.add_trace(
                go.Scatter(
                    x=stego_time,
                    y=stego_signal,
                    mode='lines',
                    name='Stego',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='Audio Waveform Analysis',
            height=400 if stego_signal is None else 600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Time (seconds)')
        fig.update_yaxes(title_text='Amplitude')
        
        return fig
    
    def create_frequency_plot(self, signal: np.ndarray, title: str = "Frequency Spectrum") -> go.Figure:
        """Create frequency spectrum plot"""
        # Perform FFT
        fft_result = fft(signal)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        freqs = fftfreq(len(signal), 1/self.sample_rate)[:len(fft_result)//2]
        
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=magnitude_db,
                mode='lines',
                name='Magnitude',
                line=dict(color='green', width=1)
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude (dB)',
            height=400
        )
        
        return fig
    
    def create_ber_plot(self, ber_data: Dict) -> go.Figure:
        """Create Band Energy Ratio plot"""
        band_info = ber_data['band_info']
        
        fig = go.Figure(data=[
            go.Bar(
                x=[info['freq_range'] for info in band_info],
                y=[info['energy'] for info in band_info],
                text=[f"{ratio:.3f}" for ratio in ber_data['band_ratios']],
                textposition='auto',
                name='Band Energy'
            )
        ])
        
        fig.update_layout(
            title='Band Energy Ratio (BER) Analysis',
            xaxis_title='Frequency Bands',
            yaxis_title='Energy',
            height=400
        )
        
        return fig
    
    def save_audio_to_bytes(self, signal: np.ndarray) -> bytes:
        """Convert audio signal to WAV bytes"""
        # Normalize signal
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal))
        signal = (signal * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(signal.tobytes())
        
        return buffer.getvalue()

def main():
    st.set_page_config(
        page_title="Enhanced Audio Steganography Analysis Tool",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Enhanced Audio Steganography Analysis Tool")
    st.markdown("Multi-format encoding with ML-powered auto-detection and persistent data storage")
    
    # Initialize system
    if 'system' not in st.session_state:
        st.session_state.system = AudioSteganalysisSystem()
    
    system = st.session_state.system
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # ML Model Status
    if system.ml_classifier:
        st.sidebar.subheader("🤖 ML Auto-Detection")
        if system.ml_classifier.is_trained:
            st.sidebar.success(f"✅ Model trained with {len(system.ml_classifier.training_data)} samples")
        else:
            st.sidebar.info("📊 Model learning from your encodings...")
        
        # Training data summary
        if system.ml_classifier.training_data:
            encoding_counts = {}
            for data in system.ml_classifier.training_data:
                enc_type = data['label']
                encoding_counts[enc_type] = encoding_counts.get(enc_type, 0) + 1
            
            st.sidebar.write("**Training Data:**")
            for enc_type, count in encoding_counts.items():
                st.sidebar.write(f"- {enc_type.upper()}: {count} samples")
        
        # Data persistence info
        st.sidebar.markdown("---")
        st.sidebar.subheader("💾 Data Persistence")
        st.sidebar.success("✅ Data automatically saved")
        if os.path.exists(MODEL_FILE):
            model_time = datetime.fromtimestamp(os.path.getmtime(MODEL_FILE))
            st.sidebar.info(f"Last model update: {model_time.strftime('%Y-%m-%d %H:%M')}")
        
        # Clear data option
        if st.sidebar.button("🗑️ Clear All Data"):
            if os.path.exists(MODEL_FILE):
                os.remove(MODEL_FILE)
            if os.path.exists(TRAINING_DATA_FILE):
                os.remove(TRAINING_DATA_FILE)
            system.ml_classifier.training_data = []
            system.ml_classifier.is_trained = False
            st.sidebar.success("All data cleared!")
            st.experimental_rerun()
    
    # Encoding parameters
    st.sidebar.subheader("Encoding Parameters")
    encoding_type = st.sidebar.selectbox(
        "Encoding Type",
        ["morse", "dtmf", "fsk", "ask", "psk", "manchester"]
    )
    
    base_freq = st.sidebar.slider("Base Frequency (Hz)", 200, 2000, 800, 50)
    duration = st.sidebar.slider("Symbol Duration (s)", 0.05, 0.5, 0.1, 0.01)
    amplitude = st.sidebar.slider("Amplitude", 0.1, 1.0, 0.5, 0.05)
    sample_rate = st.sidebar.selectbox("Sample Rate (Hz)", [22050, 44100, 48000], index=1)
    
    # Update system sample rate
    system.sample_rate = sample_rate
    
    # Main interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎵 Encoding", "🔍 Decoding", "📊 Analysis", "📈 Comparison", "🤖 ML Model"])
    
    with tab1:
        st.header("Audio Encoding")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Message Input")
            
            # Different example messages for different encodings
            example_messages = {
                'morse': "HELLO WORLD",
                'dtmf': "123456789*0#",
                'fsk': "TEST",
                'ask': "DATA",
                'psk': "SIGNAL",
                'manchester': "BINARY"
            }
            
            default_message = example_messages.get(encoding_type, "HELLO WORLD")
            message = st.text_area("Enter message to encode:", default_message, height=100)
            
            # Show encoding-specific tips
            if encoding_type == 'dtmf':
                st.info("💡 DTMF supports: 0-9, A-D, *, #")
            elif encoding_type == 'morse':
                st.info("💡 Morse supports: A-Z, 0-9, spaces")
            else:
                st.info(f"💡 {encoding_type.upper()} supports: Any ASCII text")
            
            if st.button("Generate Encoded Audio", type="primary"):
                if message:
                    # Generate encoded signal
                    with st.spinner(f"Encoding message using {encoding_type.upper()}..."):
                        try:
                            if encoding_type == 'morse':
                                signal = system.encode_morse(message, base_freq, duration, amplitude)
                            elif encoding_type == 'dtmf':
                                signal = system.encode_dtmf(message, duration, duration*0.5, amplitude)
                            elif encoding_type == 'fsk':
                                signal = system.encode_fsk(message, base_freq, duration, amplitude)
                            elif encoding_type == 'ask':
                                signal = system.encode_ask(message, base_freq, duration, amplitude)
                            elif encoding_type == 'psk':
                                signal = system.encode_psk(message, base_freq, duration, amplitude)
                            elif encoding_type == 'manchester':
                                signal = system.encode_manchester(message, base_freq, duration, amplitude)
                            
                            if len(signal) > 0:
                                st.session_state.encoded_signal = signal
                                st.session_state.original_message = message
                                st.session_state.encoding_type = encoding_type
                                
                                # Add to ML training data
                                if system.ml_classifier:
                                    system.ml_classifier.add_training_data(signal, encoding_type, sample_rate)
                                
                                st.success(f"✅ Successfully encoded '{message}' using {encoding_type.upper()}")
                                
                                # Display basic info
                                duration_sec = len(signal) / sample_rate
                                st.info(f"📊 Signal duration: {duration_sec:.2f}s | Samples: {len(signal):,}")
                                
                                # Show detected characteristics for debugging
                                frequencies = system.analyze_frequencies(signal)
                                st.info(f"🎼 Detected frequencies: {frequencies[:5]}")
                            else:
                                st.error("❌ Failed to generate signal")
                                
                        except Exception as e:
                            st.error(f"❌ Encoding error: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a message to encode")
        
        with col2:
            st.subheader("Audio Output")
            
            if 'encoded_signal' in st.session_state:
                signal = st.session_state.encoded_signal
                
                # Audio player
                audio_bytes = system.save_audio_to_bytes(signal)
                st.audio(audio_bytes, format='audio/wav')
                
                # Download button
                st.download_button(
                    label="💾 Download WAV file",
                    data=audio_bytes,
                    file_name=f"encoded_{encoding_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                    mime="audio/wav"
                )
                
                # Waveform plot
                fig_wave = system.create_waveform_plot(signal)
                st.plotly_chart(fig_wave, use_container_width=True)
    
    with tab2:
        st.header("Audio Decoding")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Signal Input")
            
            # Option 1: Use encoded signal from Tab 1
            if 'encoded_signal' in st.session_state:
                if st.button("📥 Use Signal from Encoding Tab"):
                    st.session_state.decode_signal = st.session_state.encoded_signal
                    st.session_state.decode_message = st.session_state.original_message
                    st.session_state.decode_encoding = st.session_state.encoding_type
                    st.success("✅ Loaded signal from encoding tab!")
            
            # Option 2: Upload audio file
            uploaded_file = st.file_uploader("📁 Upload WAV file", type=['wav'])
            if uploaded_file is not None:
                try:
                    # Read WAV file
                    audio_data = uploaded_file.read()
                    audio_buffer = io.BytesIO(audio_data)
                    
                    with wave.open(audio_buffer, 'rb') as wav_file:
                        frames = wav_file.readframes(-1)
                        sample_rate_file = wav_file.getframerate()
                        signal_uploaded = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                    
                    st.session_state.decode_signal = signal_uploaded
                    st.session_state.decode_sample_rate = sample_rate_file
                    st.success(f"✅ Loaded audio file! Duration: {len(signal_uploaded)/sample_rate_file:.2f}s")
                    
                except Exception as e:
                    st.error(f"❌ Error loading audio file: {str(e)}")
            
            # Decoding parameters
            st.subheader("Decoding Parameters")
            decode_encoding_type = st.selectbox(
                "Expected Encoding Type",
                ["auto", "morse", "dtmf", "fsk", "ask", "psk", "manchester"],
                key="decode_type"
            )
            
            decode_base_freq = st.slider("Expected Base Frequency (Hz)", 200, 2000, 800, 50, key="decode_freq")
            decode_bit_duration = st.slider("Expected Bit Duration (s)", 0.05, 0.5, 0.1, 0.01, key="decode_duration")
            
        with col2:
            st.subheader("Decoding Results")
            
            if st.button("🔍 Decode Signal", type="primary"):
                if 'decode_signal' in st.session_state:
                    signal_to_decode = st.session_state.decode_signal
                    
                    with st.spinner("🔍 Analyzing and decoding signal..."):
                        # Analyze signal
                        frequencies = system.analyze_frequencies(signal_to_decode)
                        timing = system.analyze_timing(signal_to_decode)
                        
                        # Auto-detect encoding if requested
                        if decode_encoding_type == "auto":
                            detected_encoding = system.classify_encoding(frequencies, signal_to_decode, timing)
                            
                            # Use ML prediction if available
                            if system.ml_classifier and system.ml_classifier.is_trained:
                                ml_prediction, ml_confidence = system.ml_classifier.predict(signal_to_decode, system.sample_rate)
                                if ml_confidence > 0.5:  # Lowered threshold
                                    detected_encoding = ml_prediction
                                    st.info(f"🤖 ML Auto-detected: {detected_encoding.upper()} (confidence: {ml_confidence:.2f})")
                                else:
                                    st.info(f"📊 Rule-based detection: {detected_encoding.upper()}")
                            else:
                                st.info(f"📊 Rule-based detection: {detected_encoding.upper()}")
                            
                            encoding_to_use = detected_encoding
                        else:
                            encoding_to_use = decode_encoding_type
                        
                        # Debug information
                        with st.expander("🔧 Debug Information"):
                            st.write(f"**Detected frequencies:** {frequencies[:10]}")
                            st.write(f"**Signal segments:** {len([s for s in timing if s['type'] == 'signal'])}")
                            st.write(f"**Using encoding:** {encoding_to_use.upper()}")
                            st.write(f"**Signal duration:** {len(signal_to_decode)/system.sample_rate:.2f}s")
                        
                        # Decode based on type
                        decoded_message = ""
                        confidence = 0
                        decode_info = {}
                        
                        try:
                            if encoding_to_use == 'morse':
                                decoded_message = system.decode_morse(timing)
                                confidence = 80 if decoded_message and '?' not in decoded_message else 40
                                decode_info = {
                                    'method': 'Morse Code',
                                    'analysis': f"Found {len([s for s in timing if s['type'] == 'signal'])} signal segments"
                                }
                                
                            elif encoding_to_use == 'dtmf':
                                decoded_message = system.decode_dtmf(signal_to_decode, decode_bit_duration, decode_bit_duration*0.5)
                                confidence = 85 if decoded_message else 20
                                decode_info = {
                                    'method': 'DTMF',
                                    'analysis': f"Detected frequencies: {frequencies[:10]}"
                                }
                                
                            elif encoding_to_use == 'fsk':
                                decoded_message = system.decode_fsk(signal_to_decode, frequencies, decode_bit_duration)
                                confidence = 70 if decoded_message else 30
                                decode_info = {
                                    'method': 'FSK',
                                    'analysis': f"Using frequencies: {frequencies[:2]}"
                                }
                                
                            elif encoding_to_use == 'ask':
                                decoded_message = system.decode_ask(signal_to_decode, decode_bit_duration)
                                confidence = 65 if decoded_message else 25
                                decode_info = {
                                    'method': 'ASK',
                                    'analysis': f"Amplitude-based decoding at {decode_base_freq}Hz"
                                }
                                
                            elif encoding_to_use == 'psk':
                                decoded_message = system.decode_psk(signal_to_decode, decode_base_freq, decode_bit_duration)
                                confidence = 75 if decoded_message else 35
                                decode_info = {
                                    'method': 'PSK',
                                    'analysis': f"Phase-based decoding at {decode_base_freq}Hz"
                                }
                                
                            elif encoding_to_use == 'manchester':
                                decoded_message = system.decode_manchester(signal_to_decode, decode_base_freq, decode_bit_duration)
                                confidence = 60 if decoded_message else 30
                                decode_info = {
                                    'method': 'Manchester',
                                    'analysis': f"Transition-based decoding at {decode_base_freq}Hz"
                                }
                                
                            else:
                                decoded_message = "Unknown encoding type"
                                confidence = 0
                                decode_info = {'method': 'Unknown', 'analysis': 'Unsupported encoding'}
                        
                        except Exception as e:
                            st.error(f"❌ Decoding error: {str(e)}")
                            decoded_message = f"Decoding error: {str(e)}"
                            confidence = 0
                            decode_info = {'method': 'Error', 'analysis': str(e)}
                    
                    # Display results
                    st.success("✅ Decoding completed!")
                    
                    # Results metrics
                    result_col1, result_col2 = st.columns(2)
                    with result_col1:
                        st.metric("Confidence", f"{confidence}%")
                        st.metric("Method", decode_info.get('method', 'Unknown'))
                    
                    with result_col2:
                        st.metric("Signal Length", f"{len(signal_to_decode):,} samples")
                        st.metric("Duration", f"{len(signal_to_decode)/system.sample_rate:.2f}s")
                    
                    # Decoded message
                    st.subheader("📄 Decoded Message")
                    if decoded_message:
                        st.code(decoded_message, language="text")
                        
                        # Compare with original if available
                        if 'decode_message' in st.session_state:
                            original = st.session_state.decode_message.upper()
                            decoded_upper = decoded_message.upper()
                            
                            st.subheader("🔍 Comparison with Original")
                            
                            col3, col4 = st.columns(2)
                            with col3:
                                st.write("**Original:**")
                                st.code(st.session_state.decode_message)
                            with col4:
                                st.write("**Decoded:**")
                                st.code(decoded_message)
                            
                            # Calculate accuracy
                            if original == decoded_upper:
                                st.success("🎉 Perfect match!")
                            else:
                                # Character-level accuracy
                                max_len = max(len(original), len(decoded_upper))
                                if max_len > 0:
                                    correct_chars = sum(1 for i in range(min(len(original), len(decoded_upper))) 
                                                      if original[i] == decoded_upper[i])
                                    accuracy = (correct_chars / max_len) * 100
                                    
                                    if accuracy > 80:
                                        st.success(f"✅ High accuracy: {accuracy:.1f}%")
                                    elif accuracy > 50:
                                        st.warning(f"⚠️ Moderate accuracy: {accuracy:.1f}%")
                                    else:
                                        st.error(f"❌ Low accuracy: {accuracy:.1f}%")
                    else:
                        st.warning("⚠️ No message decoded")
                    
                    # Analysis details
                    st.subheader("📊 Analysis Details")
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.write(f"**Method:** {decode_info.get('method', 'Unknown')}")
                        st.write(f"**Frequencies detected:** {len(frequencies)} unique")
                        st.write(f"**Timing segments:** {len(timing)} total")
                    
                    with detail_col2:
                        st.write(f"**Analysis:** {decode_info.get('analysis', 'No details')}")
                        if frequencies:
                            st.write(f"**Frequency range:** {min(frequencies):.1f} - {max(frequencies):.1f} Hz")
                    
                    # Store results for ML learning
                    if decoded_message and 'decode_message' in st.session_state:
                        original = st.session_state.decode_message.upper()
                        if original == decoded_message.upper() and system.ml_classifier:
                            # Add successful decode to training data
                            system.ml_classifier.add_training_data(signal_to_decode, encoding_to_use, system.sample_rate)
                            st.info("🤖 Added successful decode to ML training data")
                    
                    # Store results
                    st.session_state.last_decode_result = {
                        'message': decoded_message,
                        'confidence': confidence,
                        'method': decode_info.get('method', 'Unknown'),
                        'frequencies': frequencies,
                        'timing': timing
                    }
                    
                else:
                    st.warning("⚠️ Please load a signal first (from encoding tab or upload file)")
            
            # Display current signal info
            if 'decode_signal' in st.session_state:
                signal_info = st.session_state.decode_signal
                st.subheader("📊 Current Signal Info")
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.metric("Samples", f"{len(signal_info):,}")
                    st.metric("Duration", f"{len(signal_info)/system.sample_rate:.2f}s")
                
                with info_col2:
                    st.metric("Sample Rate", f"{system.sample_rate:,} Hz")
                    st.metric("RMS Level", f"{np.sqrt(np.mean(signal_info**2)):.4f}")
                
                # Quick waveform preview
                fig_decode = system.create_waveform_plot(signal_info)
                st.plotly_chart(fig_decode, use_container_width=True, key="decode_waveform")
    
    with tab3:
        st.header("📊 Signal Analysis")
        
        if 'encoded_signal' not in st.session_state:
            st.warning("⚠️ Please encode a message first in the Encoding tab")
        else:
            signal = st.session_state.encoded_signal
            message = st.session_state.original_message
            enc_type = st.session_state.encoding_type
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎼 Frequency Analysis")
                fig_freq = system.create_frequency_plot(signal, f"{enc_type.upper()} Signal Spectrum")
                st.plotly_chart(fig_freq, use_container_width=True)
                
                st.subheader("📊 Band Energy Ratio (BER)")
                ber_data = system.calculate_ber(signal)
                fig_ber = system.create_ber_plot(ber_data)
                st.plotly_chart(fig_ber, use_container_width=True)
            
            with col2:
                st.subheader("💾 Embedding Capacity Metrics")
                capacity_data = system.calculate_embedding_capacity(message, signal, enc_type)
                
                # Create metrics display
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("Message Bits", capacity_data['message_bits'])
                    st.metric("Message Characters", capacity_data['message_chars'])
                    st.metric("Signal Duration (s)", f"{capacity_data['signal_duration']:.2f}")
                    st.metric("Bits/Second", f"{capacity_data['bits_per_second']:.1f}")
                
                with metrics_col2:
                    st.metric("Signal Samples", f"{capacity_data['signal_samples']:,}")
                    st.metric("Bits/Sample", f"{capacity_data['bits_per_sample']:.6f}")
                    st.metric("Chars/Second", f"{capacity_data['chars_per_second']:.1f}")
                    st.metric("Efficiency (%)", f"{capacity_data['embedding_efficiency_percent']:.2f}")
                
                st.subheader("📈 BER Statistics")
                ber_df = pd.DataFrame(ber_data['band_info'])
                ber_df['energy_ratio'] = ber_data['band_ratios']
                st.dataframe(ber_df, use_container_width=True)
    
    with tab4:
        st.header("📈 Signal Comparison")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔄 Generate Comparison Signal")
            comparison_message = st.text_input("Comparison message:", "TEST MESSAGE")
            comparison_encoding = st.selectbox(
                "Comparison encoding:",
                ["morse", "dtmf", "fsk", "ask", "psk", "manchester"],
                key="comp_encoding"
            )
            
            if st.button("🎵 Generate Comparison Signal"):
                if comparison_message:
                    with st.spinner(f"Generating {comparison_encoding.upper()} signal..."):
                        try:
                            if comparison_encoding == 'morse':
                                comp_signal = system.encode_morse(comparison_message, base_freq, duration, amplitude)
                            elif comparison_encoding == 'dtmf':
                                comp_signal = system.encode_dtmf(comparison_message, duration, duration*0.5, amplitude)
                            elif comparison_encoding == 'fsk':
                                comp_signal = system.encode_fsk(comparison_message, base_freq, duration, amplitude)
                            elif comparison_encoding == 'ask':
                                comp_signal = system.encode_ask(comparison_message, base_freq, duration, amplitude)
                            elif comparison_encoding == 'psk':
                                comp_signal = system.encode_psk(comparison_message, base_freq, duration, amplitude)
                            elif comparison_encoding == 'manchester':
                                comp_signal = system.encode_manchester(comparison_message, base_freq, duration, amplitude)
                        
                            if len(comp_signal) > 0:
                                st.session_state.comparison_signal = comp_signal
                                st.session_state.comparison_message = comparison_message
                                st.session_state.comparison_encoding = comparison_encoding
                                st.success("✅ Comparison signal generated!")
                                
                                # Add to ML training data
                                if system.ml_classifier:
                                    system.ml_classifier.add_training_data(comp_signal, comparison_encoding, sample_rate)
                            else:
                                st.error("❌ Failed to generate comparison signal")
                        except Exception as e:
                            st.error(f"❌ Error generating signal: {str(e)}")
        
        with col2:
            st.subheader("🔍 Correlation Analysis")
            
            if 'encoded_signal' in st.session_state and 'comparison_signal' in st.session_state:
                original_signal = st.session_state.encoded_signal
                comp_signal = st.session_state.comparison_signal
                
                # Calculate correlation
                corr_data = system.calculate_correlation(original_signal, comp_signal)
                
                # Display correlation metrics
                corr_col1, corr_col2 = st.columns(2)
                
                with corr_col1:
                    st.metric("Pearson Correlation", f"{corr_data['pearson_correlation']:.4f}")
                    st.metric("Max Cross-Correlation", f"{corr_data['max_cross_correlation']:.4f}")
                
                with corr_col2:
                    st.metric("RMS Difference", f"{corr_data['rms_difference']:.4f}")
                    snr_display = f"{corr_data['snr_db']:.2f}" if corr_data['snr_db'] != float('inf') else "∞"
                    st.metric("SNR (dB)", snr_display)
                
                # Waveform comparison
                st.subheader("🌊 Waveform Comparison")
                fig_comp = system.create_waveform_plot(original_signal, comp_signal)
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Frequency comparison
                st.subheader("🎼 Frequency Spectrum Comparison")
                col3, col4 = st.columns(2)
                with col3:
                    orig_encoding = st.session_state.get('encoding_type', 'original')
                    fig_freq1 = system.create_frequency_plot(original_signal, f"Original ({orig_encoding.upper()}) Spectrum")
                    st.plotly_chart(fig_freq1, use_container_width=True)
                
                with col4:
                    fig_freq2 = system.create_frequency_plot(comp_signal, f"Comparison ({comparison_encoding.upper()}) Spectrum")
                    st.plotly_chart(fig_freq2, use_container_width=True)
            else:
                st.info("ℹ️ Generate both original and comparison signals to see correlation analysis")
    
    with tab5:
        st.header("🤖 Machine Learning Model")
        
        if not ML_AVAILABLE:
            st.error("❌ Machine Learning features require scikit-learn. Please install it to use this tab.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Model Status")
            
            if system.ml_classifier.is_trained:
                st.success("✅ Model is trained and ready!")
                st.write(f"**Training samples:** {len(system.ml_classifier.training_data)}")
                
                # Show training data distribution
                encoding_counts = {}
                for data in system.ml_classifier.training_data:
                    enc_type = data['label']
                    encoding_counts[enc_type] = encoding_counts.get(enc_type, 0) + 1
                
                st.write("**Training Data Distribution:**")
                for enc_type, count in encoding_counts.items():
                    progress = count / max(encoding_counts.values()) if encoding_counts.values() else 0
                    st.write(f"**{enc_type.upper()}:** {count} samples")
                    st.progress(progress)
                
                # Model performance estimation
                if len(encoding_counts) > 1:
                    min_samples = min(encoding_counts.values())
                    max_samples = max(encoding_counts.values())
                    balance_ratio = min_samples / max_samples
                    
                    if balance_ratio > 0.7:
                        st.success("🎯 Well-balanced training data")
                    elif balance_ratio > 0.4:
                        st.warning("⚠️ Moderately balanced training data")
                    else:
                        st.error("❌ Imbalanced training data - add more samples for minority classes")
            else:
                st.info("📚 Model is learning from your encodings...")
                if system.ml_classifier.training_data:
                    st.write(f"Collected **{len(system.ml_classifier.training_data)}** samples so far")
                    st.write("Need at least **5** samples to start training")
                    
                    # Show progress
                    progress = len(system.ml_classifier.training_data) / 5
                    st.progress(min(progress, 1.0))
                else:
                    st.write("No training data yet. Start encoding messages to build the dataset!")
            
            st.subheader("🔧 Manual Training")
            training_col1, training_col2 = st.columns(2)
            
            with training_col1:
                if st.button("🚀 Force Retrain Model"):
                    if len(system.ml_classifier.training_data) >= 5:
                        with st.spinner("Training model..."):
                            success = system.ml_classifier.train_model()
                        if success:
                            st.success("✅ Model retrained successfully!")
                        else:
                            st.error("❌ Training failed!")
                    else:
                        st.warning("⚠️ Need at least 5 training samples")
            
            with training_col2:
                if st.button("🧪 Test All Encodings"):
                    test_message = "TEST"
                    test_results = {}
                    
                    with st.spinner("Testing all encodings..."):
                        for enc_type in ["morse", "dtmf", "fsk", "ask", "psk", "manchester"]:
                            try:
                                if enc_type == 'morse':
                                    test_signal = system.encode_morse(test_message, 800, 0.1, 0.5)
                                elif enc_type == 'dtmf':
                                    test_signal = system.encode_dtmf("123", 0.1, 0.05, 0.5)
                                elif enc_type == 'fsk':
                                    test_signal = system.encode_fsk(test_message, 800, 0.1, 0.5)
                                elif enc_type == 'ask':
                                    test_signal = system.encode_ask(test_message, 800, 0.1, 0.5)
                                elif enc_type == 'psk':
                                    test_signal = system.encode_psk(test_message, 800, 0.1, 0.5)
                                elif enc_type == 'manchester':
                                    test_signal = system.encode_manchester(test_message, 800, 0.1, 0.5)
                                
                                if len(test_signal) > 0:
                                    system.ml_classifier.add_training_data(test_signal, enc_type, sample_rate)
                                    test_results[enc_type] = "✅"
                                else:
                                    test_results[enc_type] = "❌"
                            except Exception as e:
                                test_results[enc_type] = f"❌ {str(e)[:20]}"
                    
                    st.write("**Test Results:**")
                    for enc_type, result in test_results.items():
                        st.write(f"- {enc_type.upper()}: {result}")
        
        with col2:
            st.subheader("📈 Feature Analysis")
            
            if system.ml_classifier.training_data:
                # Extract features for visualization
                recent_data = system.ml_classifier.training_data[-20:]  # Last 20 samples
                features_data = []
                labels = []
                
                for data in recent_data:
                    features_data.append(data['features'])
                    labels.append(data['label'])
                
                features_array = np.array(features_data)
                
                # Create feature importance plot if model is trained
                if system.ml_classifier.is_trained:
                    try:
                        feature_importance = system.ml_classifier.model.feature_importances_
                        feature_names = system.ml_classifier.feature_names
                        
                        # Create importance dataframe and sort
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': feature_importance
                        }).sort_values('importance', ascending=True)
                        
                        fig_importance = go.Figure(data=[
                            go.Bar(
                                x=importance_df['importance'],
                                y=importance_df['feature'],
                                orientation='h',
                                text=[f"{imp:.3f}" for imp in importance_df['importance']],
                                textposition='outside'
                            )
                        ])
                        
                        fig_importance.update_layout(
                            title='🎯 Feature Importance for Encoding Classification',
                            xaxis_title='Importance',
                            yaxis_title='Features',
                            height=500,
                            margin=dict(l=150)
                        )
                        
                        st.plotly_chart(fig_importance, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error creating feature importance plot: {str(e)}")
                
                # Feature statistics
                st.subheader("📊 Training Data Statistics")
                if len(features_array) > 0:
                    feature_df = pd.DataFrame(features_array, columns=system.ml_classifier.feature_names)
                    feature_df['encoding_type'] = labels
                    
                    # Show summary statistics
                    st.write("**Feature Summary (Recent 20 samples):**")
                    summary_stats = feature_df.select_dtypes(include=[np.number]).describe()
                    st.dataframe(summary_stats.round(4), use_container_width=True)
                    
                    # Encoding type distribution
                    st.write("**Encoding Distribution:**")
                    encoding_dist = pd.Series(labels).value_counts()
                    fig_dist = go.Figure(data=[
                        go.Pie(
                            labels=encoding_dist.index,
                            values=encoding_dist.values,
                            hole=0.4
                        )
                    ])
                    fig_dist.update_layout(
                        title="Training Data Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("ℹ️ No training data available for feature analysis")
            
            # Test prediction on current signal
            st.subheader("🔮 Test Prediction")
            if 'encoded_signal' in st.session_state and system.ml_classifier.is_trained:
                test_signal = st.session_state.encoded_signal
                actual_encoding = st.session_state.encoding_type
                
                prediction, confidence = system.ml_classifier.predict(test_signal, system.sample_rate)
                
                pred_col1, pred_col2 = st.columns(2)
                
                with pred_col1:
                    st.write("**Current Signal Analysis:**")
                    st.write(f"- Actual: **{actual_encoding.upper()}**")
                    st.write(f"- Predicted: **{prediction.upper()}**")
                    st.write(f"- Confidence: **{confidence:.2f}**")
                
                with pred_col2:
                    # Confidence visualization
                    fig_conf = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Prediction Confidence"},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "yellow"},
                                {'range': [0.8, 1], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.9
                            }
                        }
                    ))
                    fig_conf.update_layout(height=300)
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                if prediction == actual_encoding:
                    st.success("🎉 Correct prediction!")
                else:
                    st.error("❌ Incorrect prediction")
                    
                    # Show all class probabilities if available
                    try:
                        features = system.ml_classifier.extract_features(test_signal, system.sample_rate).reshape(1, -1)
                        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
                        features_scaled = system.ml_classifier.scaler.transform(features)
                        
                        all_probs = system.ml_classifier.model.predict_proba(features_scaled)[0]
                        class_names = system.ml_classifier.model.classes_
                        
                        st.write("**All Class Probabilities:**")
                        for class_name, prob in zip(class_names, all_probs):
                            st.write(f"- {class_name.upper()}: {prob:.3f}")
                    except:
                        pass
            else:
                if not system.ml_classifier.is_trained:
                    st.info("ℹ️ Train the model first to test predictions")
                else:
                    st.info("ℹ️ Encode a signal first to test prediction")
    
    # Footer with comprehensive tips
    st.markdown("---")
    st.markdown("""
    ### 💡 **Tips for Best Results:**
    
    **🎵 Encoding Tips:**
    - **DTMF**: Use numbers (0-9), letters (A-D), and symbols (*, #)
    - **Morse**: Use letters, numbers, and spaces - avoid special characters
    - **FSK/ASK/PSK/Manchester**: Support any ASCII text
    
    **🔍 Decoding Tips:**
    - Use "auto" detection for unknown signals
    - Adjust bit duration for better binary decoding (FSK, ASK, PSK, Manchester)
    - For Morse: Try different base frequencies if decoding fails
    
    **🤖 ML Model Tips:**
    - The model learns automatically from your encodings
    - More diverse training data = better auto-detection
    - Data is automatically saved and persists between sessions
    - Use "Test All Encodings" to quickly build training data
    
    **📊 Analysis Tips:**
    - Compare different encoding methods in the Comparison tab
    - Check frequency plots to understand signal characteristics
    - Monitor embedding efficiency for different encoding types
    """)

if __name__ == '__main__':
    main()