#!/usr/bin/env python3
"""
CTF Encoding Test Case Generator
Generates 100+ test cases for various encoding schemes commonly used in CTF challenges.
"""

import random
import string
import itertools

class EncodingGenerator:
    def __init__(self):
        # Flag formats and sample texts
        self.flag_formats = [
            "flag{}", "ctf{}", "thm{}", "htb{}", "picoCTF{}", "hack{}", 
            "cyber{}", "sec{}", "vuln{}", "pwn{}", "rev{}", "crypto{}"
        ]
        
        # Sample flag contents
        self.flag_contents = [
            "h3ll0_w0rld", "s3cr3t_m3ss4g3", "h1dd3n_tr34sur3", "c0d3_br34k3r",
            "m4st3r_h4ck3r", "crypt0_k1ng", "b1n4ry_n1nj4", "r3v3rs3_m3",
            "st3g4n0_m4st3r", "fr3qu3ncy_4n4lys1s", "s1gn4l_pr0c3ss1ng",
            "d1g1t4l_f0r3ns1cs", "n3tw0rk_hunt3r", "w3b_3xpl01t3r",
            "buff3r_0v3rfl0w", "r0p_ch41n", "sh3llc0d3_1nj3ct", "h34p_spr4y",
            "r4c3_c0nd1t10n", "t1m3_0f_ch3ck", "s1d3_ch4nn3l", "p0w3r_4n4lys1s"
        ]
        
        # Additional test strings
        self.test_strings = [
            "HELLO", "WORLD", "TEST", "MESSAGE", "SECRET", "HIDDEN", "ENCODED",
            "FREQUENCY", "SIGNAL", "DIGITAL", "BINARY", "DECODE", "CIPHER",
            "STEGANOGRAPHY", "CRYPTOGRAPHY", "TELECOMMUNICATIONS", "MODULATION"
        ]
        
        # Morse code mapping
        self.morse_code = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
            'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
            'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
            'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
            'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
            '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
            '8': '---..', '9': '----.', ' ': '/'
        }
        
        # DTMF frequencies (Dual-Tone Multi-Frequency)
        self.dtmf_mapping = {
            '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
            '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
            '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
            '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
        }
    
    def generate_flag_content(self):
        """Generate random flag content"""
        if random.choice([True, False]):
            return random.choice(self.flag_contents)
        else:
            # Generate random content
            length = random.randint(8, 20)
            chars = string.ascii_lowercase + string.digits + '_'
            return ''.join(random.choice(chars) for _ in range(length))
    
    def create_flag(self):
        """Create a random flag"""
        format_template = random.choice(self.flag_formats)
        content = self.generate_flag_content()
        return format_template.format(content)
    
    def get_test_text(self):
        """Get random test text"""
        if random.choice([True, False, False]):  # 1/3 chance for flag
            return self.create_flag()
        else:
            return random.choice(self.test_strings)
    
    def encode_morse(self, text):
        """Encode text to Morse code"""
        result = []
        for char in text.upper():
            if char in self.morse_code:
                result.append(self.morse_code[char])
            elif char in '{}_-':
                result.append(char)  # Keep special flag characters
        return ' '.join(result)
    
    def encode_dtmf(self, text):
        """Encode text to DTMF frequencies"""
        result = []
        for char in text.upper():
            if char in self.dtmf_mapping:
                freq_low, freq_high = self.dtmf_mapping[char]
                result.append(f"[{freq_low}Hz+{freq_high}Hz]")
            elif char.isalpha():
                # For letters not in DTMF, use their position (A=1, B=2, etc.)
                pos = str((ord(char) - ord('A')) % 10)
                if pos in self.dtmf_mapping:
                    freq_low, freq_high = self.dtmf_mapping[pos]
                    result.append(f"[{freq_low}Hz+{freq_high}Hz]")
        return ' '.join(result)
    
    def encode_fsk(self, text):
        """Encode text to FSK (Frequency Shift Keying) representation"""
        # FSK uses two frequencies for 0 and 1
        freq_0, freq_1 = 1200, 2200  # Common FSK frequencies
        binary = ''.join(format(ord(char), '08b') for char in text)
        
        result = []
        for bit in binary:
            if bit == '0':
                result.append(f"{freq_0}Hz")
            else:
                result.append(f"{freq_1}Hz")
        
        return ' '.join(result)
    
    def encode_ask(self, text):
        """Encode text to ASK (Amplitude Shift Keying) representation"""
        # ASK uses amplitude variations
        binary = ''.join(format(ord(char), '08b') for char in text)
        
        result = []
        for bit in binary:
            if bit == '0':
                result.append("0V")  # Low amplitude
            else:
                result.append("5V")  # High amplitude
        
        return ' '.join(result)
    
    def encode_psk(self, text):
        """Encode text to PSK (Phase Shift Keying) representation"""
        # PSK uses phase shifts (0° and 180°)
        binary = ''.join(format(ord(char), '08b') for char in text)
        
        result = []
        for bit in binary:
            if bit == '0':
                result.append("0°")   # No phase shift
            else:
                result.append("180°") # Phase shift
        
        return ' '.join(result)
    
    def encode_manchester(self, text):
        """Encode text to Manchester encoding representation"""
        # Manchester encoding: 0 = high-to-low, 1 = low-to-high
        binary = ''.join(format(ord(char), '08b') for char in text)
        
        result = []
        for bit in binary:
            if bit == '0':
                result.append("↓")  # High-to-low transition
            else:
                result.append("↑")  # Low-to-high transition
        
        return ' '.join(result)
    
    def generate_test_cases(self, num_cases=120):
        """Generate test cases for all encoding types"""
        test_cases = []
        encoders = [
            ("MORSE", self.encode_morse),
            ("DTMF", self.encode_dtmf),
            ("FSK", self.encode_fsk),
            ("ASK", self.encode_ask),
            ("PSK", self.encode_psk),
            ("MANCHESTER", self.encode_manchester)
        ]
        
        cases_per_encoder = num_cases // len(encoders)
        
        for encoder_name, encoder_func in encoders:
            print(f"\n{'='*60}")
            print(f"{encoder_name} ENCODING TEST CASES")
            print(f"{'='*60}")
            
            for i in range(cases_per_encoder):
                # Get test text
                original_text = self.get_test_text()
                
                try:
                    # Encode the text
                    encoded_text = encoder_func(original_text)
                    
                    # Create test case
                    test_case = {
                        'id': len(test_cases) + 1,
                        'encoder': encoder_name,
                        'original': original_text,
                        'encoded': encoded_text
                    }
                    
                    test_cases.append(test_case)
                    
                    # Print test case
                    print(f"\nTest Case #{test_case['id']}:")
                    print(f"Original:  {original_text}")
                    print(f"Encoded:   {encoded_text}")
                    
                except Exception as e:
                    print(f"Error encoding '{original_text}' with {encoder_name}: {e}")
        
        return test_cases

def main():
    """Main function to run the test case generator"""
    print("CTF Encoding Test Case Generator")
    print("=" * 50)
    
    generator = EncodingGenerator()
    test_cases = generator.generate_test_cases(120)  # Generate 120 test cases
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    # Count by encoder type
    encoder_counts = {}
    flag_counts = {}
    
    for case in test_cases:
        encoder = case['encoder']
        encoder_counts[encoder] = encoder_counts.get(encoder, 0) + 1
        
        # Count flags
        if any(case['original'].startswith(fmt.split('{')[0]) for fmt in generator.flag_formats):
            flag_counts[encoder] = flag_counts.get(encoder, 0) + 1
    
    print(f"Total test cases generated: {len(test_cases)}")
    print("\nTest cases by encoder:")
    for encoder, count in encoder_counts.items():
        flag_count = flag_counts.get(encoder, 0)
        print(f"  {encoder}: {count} cases ({flag_count} with flags)")
    
    print(f"\nAll test cases have been generated and displayed above.")
    print("You can copy and use these for your CTF challenges or practice!")

if __name__ == "__main__":
    main()