import re
import json
import re
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import streamlit as st

from openai import OpenAI
import os


class NetMindClient:
    """
    A class to interact with NetMind AI using the OpenAI-compatible API.
    """

    def __init__(self, api_key=None, base_url="https://api.netmind.ai/inference-api/openai/v1"):
        """
        Initialize the NetMind client.

        Args:
            api_key (str, optional): NetMind API key. Defaults to environment variable.
            base_url (str, optional): Base URL for the API. Defaults to NetMind API endpoint.
        """
        self.api_key = api_key or os.environ.get("NETMIND_API_KEY")
        self.base_url = base_url
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate_response(self, prompt, system_prompt="Act like you are a helpful assistant.",
                          model="meta-llama/Llama-4-Maverick-17B-128E-Instruct", max_tokens=20000):
        """
        Generate a response from the model based on the input prompt.

        Args:
            prompt (str): The input prompt for the model.
            system_prompt (str, optional): System prompt to guide model behavior.
                Defaults to "Act like you are a helpful assistant."
            model (str, optional): Model to use.
                Defaults to "meta-llama/Llama-4-Maverick-17B-128E-Instruct".
            max_tokens (int, optional): Maximum tokens in response. Defaults to 512.

        Returns:
            str: The model's response text.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )

            # Extract the text content from the response
            response = chat_completion.choices[0].message.content
            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"


@st.cache_resource
def load_model():
    # Select device and precision
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")
    print(f"Device set to use {device}")

    # Load model and processor
    model_id = "openai/whisper-large-v3-turbo"
    # Fix: Remove device_map="auto" to avoid conflict with pipeline device setting
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    # Optimize model for inference
    if device.startswith("cuda"):
        model = model.to(device)
        # Enable CUDA graph capture for faster inference
        torch.cuda.empty_cache()
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

    # Fix: Set return_attention_mask=True to address the attention mask warning
    processor = AutoProcessor.from_pretrained(model_id, return_attention_mask=True)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        # Speed optimizations
        chunk_length_s=15,  # Smaller chunks for faster processing
        stride_length_s=2.5,  # Smaller stride for better parallelism
        batch_size=24,  # Larger batch size, adjust based on VRAM
        return_timestamps=False  # Disable timestamp generation for speed
    )

    return pipe


def read_template(file_path):
    """
    Reads the template file from the specified path.

    Args:
        file_path (str): Path to the template file

    Returns:
        str: The template as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def detect_placeholders(template_text):
    """
    Extracts all placeholder variables from the template.

    Args:
        template_text (str): The template text

    Returns:
        list: List of unique placeholder names
    """
    if template_text is None:
        return []

    # Find all placeholders in the format {variable_name}
    pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
    placeholders = re.findall(pattern, template_text)

    # Return unique placeholder names
    return list(set(placeholders))


def fill_template(template_text, **kwargs):
    """
    Fills in the template with provided inputs using keyword arguments.
    Can handle any number of variables dynamically.

    Args:
        template_text (str): The template text with placeholders
        **kwargs: Key-value pairs where keys match placeholder names

    Returns:
        str: The filled template
    """
    if template_text is None:
        return None

    # Get all placeholders from the template
    placeholders = detect_placeholders(template_text)

    # Create a dictionary of replacements
    replacements = {}
    for placeholder in placeholders:
        if placeholder in kwargs and kwargs[placeholder] is not None:
            replacements[placeholder] = str(kwargs[placeholder])

    # Replace all placeholders with their values
    filled_text = template_text
    for placeholder, value in replacements.items():
        filled_text = filled_text.replace(f"{{{placeholder}}}", value)

    return filled_text


def print_required_inputs(template_text):
    """
    Prints all required placeholder inputs for a template.

    Args:
        template_text (str): The template text
    """
    placeholders = detect_placeholders(template_text)
    if placeholders:
        print("Required inputs for this template:")
        for placeholder in placeholders:
            print(f"- {placeholder}")
    else:
        print("No placeholders found in the template.")


