# VoiceInsight 🎙️

**VoiceInsight** is an elegant Streamlit application for recording, transcribing, and extracting user intent from voice inputs.

## Features

- **Voice Recording & Upload**: Accept WAV or MP3 audio files
- **Organized Storage**: Automatically creates timestamped directories for each recording
- **Real-time Transcription**: Converts speech to text with high accuracy
- **Intent Analysis**: Extracts the underlying intent using Llama4 API
- **Beautiful UI**: Modern interface with audio visualization
- **Performance Metrics**: View processing times for each step

## Installation

```bash
# Clone the repository
git clone https://github.com/MahdiMohseni0033/VoiceInsight.git
cd VoiceInsight

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Upload an audio file or record directly in the app
3. Adjust the number of output words as needed
4. Process the audio to get transcription and intent
5. Results are saved in the `result/` directory with timestamp organization

## Project Structure

```
VoiceInsight/
├── app.py                # Main Streamlit application
├── utils.py              # Utility functions
├── presets_vars.py       # API keys and constants
├── template.txt          # Prompt template
├── requirements.txt      # Dependencies
└── result/               # Generated results
    ├── 2025-04-17_12-50-22/
    │   ├── audio.wav     # Audio file
    │   ├── transcript.txt
    │   └── intent.txt
    └── ...
```

## Dependencies

- Streamlit for the web interface
- Librosa for audio processing
- NetMind API for intent analysis
- Various audio and visualization libraries

## License

MIT License