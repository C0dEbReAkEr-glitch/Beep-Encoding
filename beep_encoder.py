#!/usr/bin/env python3
"""
Multi-Format Beep Encoding System - Command Line Tool
Supports: Morse Code, DTMF, FSK, ASK, PSK, Manchester Encoding
"""

import argparse
import numpy as np
import wave
import json
import os
import sys
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import scipy.signal
from scipy.fft import fft, fftfreq
import sounddevice as sd
import threading
import time

class BeepEncodingSystem:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.recording = False
        self.recorded_data = None
        
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
        
        # Reverse morse dictionary for decoding
        self.reverse_morse = {v: k for k, v in self.morse_code.items()}
        
        # DTMF Frequencies
        self.dtmf_freqs = {
            '1': [697, 1209], '2': [697, 1336], '3': [697, 1477], 'A': [697, 1633],
            '4': [770, 1209], '5': [770, 1336], '6': [770, 1477], 'B': [770, 1633],
            '7': [852, 1209], '8': [852, 1336], '9': [852, 1477], 'C': [852, 1633],
            '*': [941, 1209], '0': [941, 1336], '#': [941, 1477], 'D': [941, 1633]
        }
        
        # Reverse DTMF mapping
        self.reverse_dtmf = {}
        for key, (f1, f2) in self.dtmf_freqs.items():
            self.reverse_dtmf[f"{f1}-{f2}"] = key
    
    def generate_tone(self, frequency: float, duration: float, amplitude: float = 0.3, phase: float = 0) -> np.ndarray:
        """Generate a pure tone"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    def generate_dual_tone(self, freq1: float, freq2: float, duration: float, amplitude: float = 0.3) -> np.ndarray:
        """Generate dual-tone signal (for DTMF)"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone1 = amplitude * 0.5 * np.sin(2 * np.pi * freq1 * t)
        tone2 = amplitude * 0.5 * np.sin(2 * np.pi * freq2 * t)
        return tone1 + tone2
    
    def generate_silence(self, duration: float) -> np.ndarray:
        """Generate silence"""
        return np.zeros(int(self.sample_rate * duration))
    
    def text_to_binary(self, text: str) -> str:
        """Convert text to binary string"""
        return ''.join(format(ord(char), '08b') for char in text)
    
    def binary_to_text(self, binary: str) -> str:
        """Convert binary string to text"""
        if len(binary) % 8 != 0:
            binary = binary.ljust((len(binary) + 7) // 8 * 8, '0')
        
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            char_code = int(byte, 2)
            if 32 <= char_code <= 126:  # Printable ASCII
                text += chr(char_code)
        return text
    
    def encode_morse(self, message: str, base_freq: float = 800, dot_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as Morse code"""
        print(f"Encoding Morse: '{message}'")
        
        dash_duration = dot_duration * 3
        gap_duration = dot_duration
        word_gap_duration = dot_duration * 7
        
        signal_parts = []
        
        for char in message.upper():
            if char in self.morse_code:
                morse_pattern = self.morse_code[char]
                
                if morse_pattern == '/':  # Space between words
                    signal_parts.append(self.generate_silence(word_gap_duration))
                    continue
                
                for symbol in morse_pattern:
                    if symbol == '.':
                        signal_parts.append(self.generate_tone(base_freq, dot_duration, amplitude))
                    elif symbol == '-':
                        signal_parts.append(self.generate_tone(base_freq, dash_duration, amplitude))
                    
                    # Gap between symbols
                    signal_parts.append(self.generate_silence(gap_duration))
                
                # Gap between characters
                signal_parts.append(self.generate_silence(dash_duration))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def encode_dtmf(self, message: str, tone_duration: float = 0.2, gap_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as DTMF"""
        print(f"Encoding DTMF: '{message}'")
        
        signal_parts = []
        
        for char in message.upper():
            if char in self.dtmf_freqs:
                freq1, freq2 = self.dtmf_freqs[char]
                signal_parts.append(self.generate_dual_tone(freq1, freq2, tone_duration, amplitude))
                signal_parts.append(self.generate_silence(gap_duration))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def encode_fsk(self, message: str, base_freq: float = 800, bit_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as FSK (Frequency Shift Keying)"""
        print(f"Encoding FSK: '{message}'")
        
        binary = self.text_to_binary(message)
        freq0 = base_freq
        freq1 = base_freq * 1.5
        
        signal_parts = []
        
        for bit in binary:
            freq = freq0 if bit == '0' else freq1
            signal_parts.append(self.generate_tone(freq, bit_duration, amplitude))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def encode_ask(self, message: str, base_freq: float = 800, bit_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as ASK (Amplitude Shift Keying)"""
        print(f"Encoding ASK: '{message}'")
        
        binary = self.text_to_binary(message)
        signal_parts = []
        
        for bit in binary:
            bit_amplitude = amplitude if bit == '1' else amplitude * 0.1
            signal_parts.append(self.generate_tone(base_freq, bit_duration, bit_amplitude))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def encode_psk(self, message: str, base_freq: float = 800, bit_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as PSK (Phase Shift Keying)"""
        print(f"Encoding PSK: '{message}'")
        
        binary = self.text_to_binary(message)
        signal_parts = []
        phase = 0
        
        for bit in binary:
            if bit == '1':
                phase = np.pi if phase == 0 else 0  # Phase shift for '1'
            signal_parts.append(self.generate_tone(base_freq, bit_duration, amplitude, phase))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def encode_manchester(self, message: str, base_freq: float = 800, bit_duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Encode message as Manchester encoding"""
        print(f"Encoding Manchester: '{message}'")
        
        binary = self.text_to_binary(message)
        signal_parts = []
        half_duration = bit_duration / 2
        
        for bit in binary:
            if bit == '1':
                # '1' = low-to-high transition
                signal_parts.append(self.generate_tone(base_freq, half_duration, amplitude * 0.3))
                signal_parts.append(self.generate_tone(base_freq, half_duration, amplitude))
            else:
                # '0' = high-to-low transition
                signal_parts.append(self.generate_tone(base_freq, half_duration, amplitude))
                signal_parts.append(self.generate_tone(base_freq, half_duration, amplitude * 0.3))
        
        return np.concatenate(signal_parts) if signal_parts else np.array([])
    
    def save_audio(self, signal: np.ndarray, filename: str) -> None:
        """Save signal to WAV file"""
        # Normalize signal to prevent clipping
        signal = signal / np.max(np.abs(signal))
        signal = (signal * 32767).astype(np.int16)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(signal.tobytes())
        
        print(f"Audio saved to: {filename}")
    
    def load_audio(self, filename: str) -> np.ndarray:
        """Load audio from WAV file"""
        with wave.open(filename, 'r') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            
            if sample_rate != self.sample_rate:
                print(f"Warning: File sample rate ({sample_rate}) differs from system rate ({self.sample_rate})")
            
            signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            return signal
    
    def record_audio(self, duration: float) -> np.ndarray:
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")
        return recording.flatten()
    
    def analyze_frequencies(self, signal: np.ndarray, window_size: int = 2048) -> List[float]:
        """Analyze frequencies in the signal using FFT"""
        frequencies = []
        overlap = window_size // 2
        
        for i in range(0, len(signal) - window_size, overlap):
            window = signal[i:i + window_size]
            
            # Apply Hamming window
            windowed = window * np.hamming(window_size)
            
            # Perform FFT
            fft_result = fft(windowed)
            freqs = fftfreq(window_size, 1/self.sample_rate)
            
            # Find peaks in positive frequencies
            magnitude = np.abs(fft_result[:window_size//2])
            positive_freqs = freqs[:window_size//2]
            
            # Find peaks
            peaks, _ = scipy.signal.find_peaks(magnitude, height=np.max(magnitude) * 0.1)
            
            for peak in peaks:
                freq = positive_freqs[peak]
                if 50 < freq < 3000:  # Audio frequency range
                    frequencies.append(freq)
        
        # Return unique frequencies sorted by occurrence
        unique_freqs = list(set([round(f, 1) for f in frequencies]))
        return sorted(unique_freqs)
    
    def analyze_timing(self, signal: np.ndarray, threshold: float = 0.01) -> List[Dict]:
        """Analyze timing patterns in the signal"""
        segments = []
        is_signal = False
        segment_start = 0
        
        for i, sample in enumerate(signal):
            amplitude = abs(sample)
            
            if not is_signal and amplitude > threshold:
                # Start of signal
                if segments and segments[-1]['type'] == 'silence':
                    segments[-1]['duration'] = (i - segment_start) / self.sample_rate
                is_signal = True
                segment_start = i
            elif is_signal and amplitude <= threshold:
                # End of signal
                segments.append({
                    'type': 'signal',
                    'start': segment_start / self.sample_rate,
                    'duration': (i - segment_start) / self.sample_rate
                })
                is_signal = False
                segment_start = i
                segments.append({'type': 'silence', 'start': segment_start / self.sample_rate})
        
        # Handle final segment
        if is_signal:
            segments.append({
                'type': 'signal',
                'start': segment_start / self.sample_rate,
                'duration': (len(signal) - segment_start) / self.sample_rate
            })
        
        return segments
    
    def classify_encoding(self, frequencies: List[float], signal: np.ndarray, timing: List[Dict]) -> str:
        """Classify the encoding type based on signal analysis"""
        freq_count = len(frequencies)
        
        # Check for DTMF
        dtmf_matches = 0
        for freq in frequencies:
            if any(abs(freq - df) < 20 for df_pair in self.dtmf_freqs.values() for df in df_pair):
                dtmf_matches += 1
        
        if dtmf_matches >= 2 and freq_count >= 2:
            return 'DTMF'
        
        # Check for FSK
        if freq_count == 2:
            ratio = max(frequencies) / min(frequencies)
            if 1.2 < ratio < 2.0:
                return 'FSK'
        
        # Check for Morse Code
        signal_segments = [s for s in timing if s['type'] == 'signal']
        if len(signal_segments) > 3 and freq_count == 1:
            durations = [s['duration'] for s in signal_segments]
            if len(set([round(d, 1) for d in durations])) > 1:  # Variable durations
                return 'Morse Code'
        
        # Check for ASK (amplitude variations)
        amplitude_segments = []
        window_size = 1024
        for i in range(0, len(signal) - window_size, window_size):
            rms = np.sqrt(np.mean(signal[i:i+window_size]**2))
            amplitude_segments.append(rms)
        
        if len(amplitude_segments) > 0:
            amp_variance = np.var(amplitude_segments) / np.mean(amplitude_segments)
            if amp_variance > 0.5 and freq_count == 1:
                return 'ASK'
        
        # Check for Manchester (regular transitions)
        if len(amplitude_segments) > 5:
            transitions = sum(1 for i in range(1, len(amplitude_segments)) 
                            if abs(amplitude_segments[i] - amplitude_segments[i-1]) > 0.1)
            if transitions > len(amplitude_segments) * 0.3:
                return 'Manchester'
        
        # Default to PSK
        return 'PSK'
    
    def decode_morse(self, timing: List[Dict]) -> str:
        """Decode Morse code from timing analysis"""
        signal_segments = [s for s in timing if s['type'] == 'signal']
        silence_segments = [s for s in timing if s['type'] == 'silence' and 'duration' in s]
        
        if not signal_segments:
            return ""
        
        durations = [s['duration'] for s in signal_segments]
        avg_duration = np.mean(durations)
        threshold = avg_duration * 1.5
        
        morse_string = ""
        current_char = ""
        
        for i, segment in enumerate(timing):
            if segment['type'] == 'signal':
                if segment['duration'] < threshold:
                    current_char += '.'
                else:
                    current_char += '-'
            elif segment['type'] == 'silence' and 'duration' in segment:
                if current_char:
                    if segment['duration'] > threshold * 3:  # Word break
                        morse_string += current_char + " / "
                    else:  # Character break
                        morse_string += current_char + " "
                    current_char = ""
        
        if current_char:
            morse_string += current_char
        
        # Convert to text
        words = morse_string.split(" / ")
        decoded_words = []
        
        for word in words:
            chars = word.strip().split(" ")
            decoded_chars = []
            for char in chars:
                if char in self.reverse_morse:
                    decoded_chars.append(self.reverse_morse[char])
                else:
                    decoded_chars.append('?')
            decoded_words.append(''.join(decoded_chars))
        
        return ' '.join(decoded_words)
    
    def decode_dtmf(self, frequencies: List[float]) -> str:
        """Decode DTMF from frequency analysis"""
        decoded = ""
        
        # Group frequencies into pairs
        for i in range(0, len(frequencies) - 1, 2):
            f1, f2 = sorted([frequencies[i], frequencies[i+1]])
            
            best_match = ""
            min_distance = float('inf')
            
            for char, (df1, df2) in self.dtmf_freqs.items():
                distance = abs(f1 - min(df1, df2)) + abs(f2 - max(df1, df2))
                if distance < min_distance and distance < 50:
                    min_distance = distance
                    best_match = char
            
            if best_match:
                decoded += best_match
        
        return decoded
    
    def decode_fsk(self, signal: np.ndarray, frequencies: List[float], bit_duration: float = 0.1) -> str:
        """Decode FSK signal"""
        if len(frequencies) < 2:
            return ""
        
        freq0, freq1 = sorted(frequencies[:2])
        window_size = int(self.sample_rate * bit_duration)
        binary_string = ""
        
        for i in range(0, len(signal) - window_size, window_size):
            window = signal[i:i + window_size]
            
            # Simple frequency detection using correlation
            t = np.linspace(0, bit_duration, window_size, False)
            corr0 = np.abs(np.sum(window * np.sin(2 * np.pi * freq0 * t)))
            corr1 = np.abs(np.sum(window * np.sin(2 * np.pi * freq1 * t)))
            
            binary_string += '0' if corr0 > corr1 else '1'
        
        return self.binary_to_text(binary_string)
    
    def save_results(self, results: Dict, filename: str) -> None:
        """Save analysis and decoding results to JSON file"""
        results['timestamp'] = datetime.now().isoformat()
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Multi-Format Beep Encoding System')
    
    # Main operation modes
    parser.add_argument('mode', choices=['encode', 'decode', 'analyze'], 
                       help='Operation mode')
    
    # Encoding options
    parser.add_argument('-t', '--type', choices=['morse', 'dtmf', 'fsk', 'ask', 'psk', 'manchester'],
                       default='morse', help='Encoding type')
    parser.add_argument('-m', '--message', type=str, help='Message to encode')
    
    # Audio parameters
    parser.add_argument('-f', '--frequency', type=float, default=800, 
                       help='Base frequency in Hz')
    parser.add_argument('-d', '--duration', type=float, default=0.1,
                       help='Bit/symbol duration in seconds')
    parser.add_argument('-a', '--amplitude', type=float, default=0.3,
                       help='Signal amplitude (0.0-1.0)')
    
    # File operations
    parser.add_argument('-i', '--input', type=str, help='Input audio file')
    parser.add_argument('-o', '--output', type=str, help='Output file (audio or results)')
    parser.add_argument('-r', '--record', type=float, help='Record duration in seconds')
    
    # Playback
    parser.add_argument('-p', '--play', action='store_true', help='Play generated audio')
    
    # Sample rate
    parser.add_argument('-s', '--sample-rate', type=int, default=44100,
                       help='Sample rate in Hz')
    
    args = parser.parse_args()
    
    # Initialize system
    system = BeepEncodingSystem(sample_rate=args.sample_rate)
    
    if args.mode == 'encode':
        if not args.message:
            print("Error: Message required for encoding (-m/--message)")
            return
        
        # Generate encoded signal based on type
        print(f"Encoding '{args.message}' using {args.type.upper()}")
        
        if args.type == 'morse':
            signal = system.encode_morse(
                args.message, 
                base_freq=args.frequency,
                dot_duration=args.duration,
                amplitude=args.amplitude
            )
        elif args.type == 'dtmf':
            signal = system.encode_dtmf(
                args.message,
                tone_duration=args.duration,
                amplitude=args.amplitude
            )
        elif args.type == 'fsk':
            signal = system.encode_fsk(
                args.message,
                base_freq=args.frequency,
                bit_duration=args.duration,
                amplitude=args.amplitude
            )
        elif args.type == 'ask':
            signal = system.encode_ask(
                args.message,
                base_freq=args.frequency,
                bit_duration=args.duration,
                amplitude=args.amplitude
            )
        elif args.type == 'psk':
            signal = system.encode_psk(
                args.message,
                base_freq=args.frequency,
                bit_duration=args.duration,
                amplitude=args.amplitude
            )
        elif args.type == 'manchester':
            signal = system.encode_manchester(
                args.message,
                base_freq=args.frequency,
                bit_duration=args.duration,
                amplitude=args.amplitude
            )
        else:
            print(f"Error: Unknown encoding type: {args.type}")
            return
        
        if len(signal) == 0:
            print("Error: No signal generated")
            return
        
        # Save audio if output specified
        if args.output:
            system.save_audio(signal, args.output)
        
        # Play audio if requested
        if args.play:
            print("Playing audio...")
            sd.play(signal, system.sample_rate)
            sd.wait()
    
    elif args.mode == 'decode':
        signal = None
        
        # Get signal from file or recording
        if args.input:
            signal = system.load_audio(args.input)
            print(f"Loaded audio from: {args.input}")
        elif args.record:
            signal = system.record_audio(args.record)
        else:
            print("Error: Input file (-i) or recording duration (-r) required for decoding")
            return
        
        if signal is None or len(signal) == 0:
            print("Error: No signal to decode")
            return
        
        # Analyze signal
        print("Analyzing signal...")
        frequencies = system.analyze_frequencies(signal)
        timing = system.analyze_timing(signal)
        encoding_type = system.classify_encoding(frequencies, signal, timing)
        
        print(f"Detected encoding: {encoding_type}")
        print(f"Frequencies found: {[f'{f:.1f}Hz' for f in frequencies[:10]]}")
        print(f"Signal segments: {len([s for s in timing if s['type'] == 'signal'])}")
        
        # Decode based on detected type
        decoded_message = ""
        confidence = 0
        
        if encoding_type == 'Morse Code':
            decoded_message = system.decode_morse(timing)
            confidence = 80 if decoded_message and '?' not in decoded_message else 40
        elif encoding_type == 'DTMF':
            decoded_message = system.decode_dtmf(frequencies)
            confidence = 85 if decoded_message else 20
        elif encoding_type == 'FSK':
            decoded_message = system.decode_fsk(signal, frequencies, args.duration)
            confidence = 70 if decoded_message else 30
        else:
            decoded_message = f"Decoding not implemented for {encoding_type}"
            confidence = 0
        
        print(f"\nDecoded message: '{decoded_message}'")
        print(f"Confidence: {confidence}%")
        
        # Save results
        if args.output:
            results = {
                'mode': 'decode',
                'detected_encoding': encoding_type,
                'decoded_message': decoded_message,
                'confidence': confidence,
                'analysis': {
                    'frequencies': frequencies,
                    'signal_duration': len(signal) / args.sample_rate,
                    'segments': len(timing)
                }
            }
            system.save_results(results, args.output)
    
    elif args.mode == 'analyze':
        signal = None
        
        # Get signal from file or recording
        if args.input:
            signal = system.load_audio(args.input)
            print(f"Loaded audio from: {args.input}")
        elif args.record:
            signal = system.record_audio(args.record)
        else:
            print("Error: Input file (-i) or recording duration (-r) required for analysis")
            return
        
        if signal is None or len(signal) == 0:
            print("Error: No signal to analyze")
            return
        
        # Perform detailed analysis
        print("Performing detailed signal analysis...")
        frequencies = system.analyze_frequencies(signal)
        timing = system.analyze_timing(signal)
        encoding_type = system.classify_encoding(frequencies, signal, timing)
        
        # Calculate additional statistics
        signal_segments = [s for s in timing if s['type'] == 'signal']
        silence_segments = [s for s in timing if s['type'] == 'silence' and 'duration' in s]
        
        print(f"\n=== SIGNAL ANALYSIS REPORT ===")
        print(f"Duration: {len(signal) / args.sample_rate:.2f} seconds")
        print(f"Sample rate: {args.sample_rate} Hz")
        print(f"Detected encoding: {encoding_type}")
        print(f"\nFrequency Analysis:")
        print(f"  Dominant frequencies: {[f'{f:.1f}Hz' for f in frequencies[:10]]}")
        print(f"  Frequency count: {len(frequencies)}")
        
        print(f"\nTiming Analysis:")
        print(f"  Signal segments: {len(signal_segments)}")
        print(f"  Silence segments: {len(silence_segments)}")
        
        if signal_segments:
            durations = [s['duration'] for s in signal_segments]
            print(f"  Signal duration range: {min(durations):.3f} - {max(durations):.3f} seconds")
            print(f"  Average signal duration: {np.mean(durations):.3f} seconds")
        
        if silence_segments:
            silence_durations = [s['duration'] for s in silence_segments]
            print(f"  Silence duration range: {min(silence_durations):.3f} - {max(silence_durations):.3f} seconds")
            print(f"  Average silence duration: {np.mean(silence_durations):.3f} seconds")
        
        # Save detailed analysis
        if args.output:
            results = {
                'mode': 'analyze',
                'signal_info': {
                    'duration': len(signal) / args.sample_rate,
                    'sample_rate': args.sample_rate,
                    'length': len(signal)
                },
                'detected_encoding': encoding_type,
                'frequency_analysis': {
                    'frequencies': frequencies,
                    'frequency_count': len(frequencies)
                },
                'timing_analysis': {
                    'signal_segments': len(signal_segments),
                    'silence_segments': len(silence_segments),
                    'signal_durations': [s['duration'] for s in signal_segments],
                    'silence_durations': [s['duration'] for s in silence_segments if 'duration' in s]
                }
            }
            system.save_results(results, args.output)

if __name__ == '__main__':
    main()