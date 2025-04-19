from utils import NetMindClient, read_template, load_model, fill_template
from presets_vars import NETMIND_API_KEY
import time

# ============= Initialize the Lama4-API and load prompt template==================

llama4 = NetMindClient(api_key=NETMIND_API_KEY)
template_file = "template.txt"
template = read_template(template_file)

# ============= Load models==================
pipe = load_model()

if __name__ == "__main__":
    audio_path = "x.wav"  # Path to your audio file
    transcribe_result = pipe(
        audio_path,
        generate_kwargs={
            "task": "transcribe",
            # Fix: Remove forced_decoder_ids which conflicts with task=transcribe
            "repetition_penalty": 1.0,  # Reduce computation
            "no_repeat_ngram_size": 3,  # Prevent repetitions efficiently
            "num_beams": 1,  # Use greedy decoding for speed
            "max_new_tokens": 256  # Limit output size
        }
    )

    print(transcribe_result["text"])
    parameters = {"user_conversation": transcribe_result["text"],
                  "n_word": 3, }
    filled_template = fill_template(template, **parameters)

    tik = time.time()
    response = llama4.generate_response(filled_template)
    tok = time.time() - tik
    print(f"API time: {tok:.2f}s")
    print(f"response: {response}")

