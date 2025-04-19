import streamlit as st
import os
import time
import datetime
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from utils import NetMindClient, read_template, load_model, fill_template
import streamlit.components.v1 as components
import torch

torch.classes.__path__ = []
# put your NETMIND_API_KEY here
NETMIND_API_KEY = os.environ.get("NETMIND_API_KEY")
# Set page configuration
st.set_page_config(
    page_title="Voice Intent Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to make the app beautiful
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5 !important;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #5E35B1 !important;
        margin-bottom: 1rem;
    }
    .result-container {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5;
    }
    .metrics-container {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .intent-box {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #bbdefb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .upload-section {
        background-color: #f9fafe;
        padding: 2rem;
        border-radius: 10px;
        border: 1px dashed #7986cb;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAudio {
        width: 100%;
    }
    .stProgress {
        height: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'transcribe_result' not in st.session_state:
    st.session_state.transcribe_result = ""
if 'user_intent' not in st.session_state:
    st.session_state.user_intent = ""
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'transcription_time' not in st.session_state:
    st.session_state.transcription_time = 0
if 'api_time' not in st.session_state:
    st.session_state.api_time = 0
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'waveform_fig' not in st.session_state:
    st.session_state.waveform_fig = None

# Title and introduction
st.markdown('<h1 class="main-header">🎙️ Voice Intent Analyzer</h1>', unsafe_allow_html=True)

st.markdown("""
This app helps you analyze voice recordings to understand user intent. 
Simply upload an audio file or record directly, and the AI will transcribe it and determine the underlying intent.
""")


# Function to create directory structure
def create_storage_directory():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = os.path.join("result", timestamp)
    os.makedirs(directory, exist_ok=True)
    return directory, timestamp


# Function to save audio file
def save_audio_file(uploaded_file, directory):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    target_filename = "audio" + file_extension
    target_path = os.path.join(directory, target_filename)

    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return target_path


# Function to process audio and extract intent
def process_audio(audio_path, num_words):
    # Initialize models if not already loaded
    if 'pipe' not in st.session_state:
        with st.spinner("Loading speech recognition model..."):
            st.session_state.pipe = load_model()
            st.session_state.llama4 = NetMindClient(api_key=NETMIND_API_KEY)
            st.session_state.template = read_template("prompt_scripts/template.txt")

    # Transcribe audio
    transcribe_start = time.time()
    transcribe_result = st.session_state.pipe(
        audio_path,
        generate_kwargs={
            "task": "transcribe",
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "num_beams": 1,
            "max_new_tokens": 256
        }
    )
    transcription_time = time.time() - transcribe_start

    # Generate visualization
    y, sr = librosa.load(audio_path)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()

    # Generate intent
    parameters = {
        "user_conversation": transcribe_result,
        "n_word": num_words,
    }
    filled_template = fill_template(st.session_state.template, **parameters)

    api_start = time.time()
    user_intent = st.session_state.llama4.generate_response(filled_template)
    api_time = time.time() - api_start

    total_time = transcription_time + api_time

    return transcribe_result, user_intent, total_time, transcription_time, api_time, fig


# Create sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)

    num_words = st.slider(
        "Number of output words for intent summary:",
        min_value=1,
        max_value=10,
        value=3,
        help="Specify how many words should be used to summarize the intent"
    )

    st.divider()

    st.markdown('<h2 class="sub-header">ℹ️ About</h2>', unsafe_allow_html=True)
    st.markdown("""
    This application uses:
    - Speech-to-text model for transcription
    - Llama4 API for intent extraction

    All recordings are stored in an organized directory structure for reference.
    """)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<h2 class="sub-header">🔊 Upload Audio</h2>', unsafe_allow_html=True)

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an audio file (WAV or MP3)", type=["wav", "mp3"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Create directory and save file
        directory, timestamp = create_storage_directory()
        audio_path = save_audio_file(uploaded_file, directory)
        st.session_state.audio_path = audio_path

        st.success(f"Audio saved in: result/{timestamp}/")

        # Process button
        if st.button("Process Audio", type="primary", use_container_width=True):
            with st.spinner("Processing audio..."):
                st.session_state.transcribe_result, st.session_state.user_intent, \
                    st.session_state.processing_time, st.session_state.transcription_time, \
                    st.session_state.api_time, st.session_state.waveform_fig = process_audio(audio_path, num_words)

                # Save transcript and intent to the same directory
                with open(os.path.join(directory, "transcript.txt"), "w") as f:
                    f.write(st.session_state.transcribe_result['text'])
                with open(os.path.join(directory, "intent.txt"), "w") as f:
                    f.write(st.session_state.user_intent)

            st.success("Processing complete!")

with col2:
    if st.session_state.audio_path:
        st.markdown('<h2 class="sub-header">🎵 Audio Preview</h2>', unsafe_allow_html=True)
        st.audio(st.session_state.audio_path)

        if st.session_state.waveform_fig:
            st.pyplot(st.session_state.waveform_fig)

# Display results if available
if st.session_state.transcribe_result:
    st.markdown('<h2 class="sub-header">📝 Results</h2>', unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Processing Time", f"{st.session_state.processing_time:.2f}s")
    col2.metric("Transcription Time", f"{st.session_state.transcription_time:.2f}s")
    col3.metric("API Response Time", f"{st.session_state.api_time:.2f}s")
    st.markdown('</div>', unsafe_allow_html=True)

    # Show/hide transcription
    show_transcription = st.toggle("Show Transcription", value=True)
    if show_transcription:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("### 🔤 Transcription")
        st.write(st.session_state.transcribe_result)
        st.markdown('</div>', unsafe_allow_html=True)

    # Show intent (always visible and highlighted)
    st.markdown('<div class="intent-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 User Intent")
    st.markdown(f"<h3 style='text-align: center; color: #1565C0;'>{st.session_state.user_intent}</h3>",
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center;">
    <p>Developed with ❤️ using Streamlit and AI</p>
</div>
""", unsafe_allow_html=True)