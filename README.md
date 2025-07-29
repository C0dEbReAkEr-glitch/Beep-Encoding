# Beep Encoding Project

This repository contains tools and scripts for audio steganography and encoding/decoding experiments, including machine learning-based analysis and test case generation.

## Project Structure

- **ml.py**: Streamlit-based interactive tool for audio steganography analysis, encoding, decoding, and ML-powered auto-detection. Provides a web UI for experimenting with various encoding schemes.
- **beep_encoder.py**: Command-line tool for encoding and decoding messages in audio files.
- **testcase.py**: Script for generating and managing test cases, useful for training and evaluating machine learning models.
- **steg_data/**: Contains persistent data, trained ML models, and training data.
- **top_secret.wav**: Example audio file.

## Features

- Supports multiple encoding schemes: Morse, FSK, ASK, PSK, Manchester, and DTMF.
- Machine learning classifier for automatic encoding type detection (in `ml.py`).
- Persistent storage of training data and models.
- Test case generation for model training and evaluation.

## Known Issues

- **DTMF encoding and decoding is not working properly.** This feature is still under development and may produce incorrect results.

## Usage

- To use the Streamlit tool:
  ```bash
  streamlit run ml.py
  ```
- To use the command-line encoder/decoder:
  ```bash
  python beep_encoder.py --help
  ```
- To generate test cases:
  ```bash
  python testcase.py
  ```

## Contributions

Contributions and bug reports are welcome! Please open an issue or submit a pull request.

---

**Note:** This project is under active development. Some features may be experimental or incomplete.
