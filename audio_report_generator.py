import nltk.tokenize
import numpy as np
from openai import OpenAI
import torch
from transformers import AutoProcessor
import os
import time
from scipy.io.wavfile import write as write_wav
import nltk
from enum import Enum
from transformers import BarkModel
import logging
import logging.config

# Load the logging configuration
logging.config.fileConfig('logging.config')

# Get the logger specified in the configuration file
logger = logging.getLogger('sampleLogger')


class SupportedModels(Enum):
    Mistral7B = 1
    Dolphin_2_8_mistral_v02 = 2
    Hermes_2_Pro_Mistral = 3
    Gemma_1_1_it = 4


final_time_start = time.time()
# Can change the prompt format syntax
selected_model = SupportedModels.Hermes_2_Pro_Mistral
summarizer_temp = 0.5
bark_model = "suno/bark-small"
base_dir = "summaries"

# client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
ollama_client = OpenAI(base_url="http://localhost:11434/v1",
                       api_key="ollama")

for podcast_summary_file in os.listdir(base_dir):

    # podcast_summary_file = "./Flyvende tallerken - Hvorfor tolker vi UFO-fænomenet så forskelligt\Flyvende tallerken - Hvorfor tolker vi UFO-fænomenet så forskelligt.md"
    file_name = os.path.join(base_dir, podcast_summary_file)
    waw_file_name = file_name + "_narritive.wav"
    if os.path.exists(waw_file_name) or file_name.find(".wav") != -1:
        continue

    logger.info(f"Creating narritive for file: {file_name}")

    with open(file_name, "r", encoding='utf-8') as md_file:
        podcast_data = md_file.readlines()

    user_prompt = f"""
    Take this markup formatted document and create a short summary as a scripted narritive.
    The script should be clear an concise, suitable for spoken language.
    It must NOT contain anything you wouldn't normally say or mention, like technical info, links, URLs or timestamps.

    # Document:
    {podcast_data}
    """

    history = []
    completion = {}
    # history = [{"role": "user", "content": f"<podcast_text_start>{merged_summary}<podcast_text_end>\n{system_prompt}\n{example_layout}\n{user_prompt}"}]
    prompt = ""

    if (SupportedModels.Dolphin_2_8_mistral_v02 == selected_model or SupportedModels.Hermes_2_Pro_Mistral == selected_model):
        prompt = f"<|im_start|>system\nYou are an expert analyst and narrator.\n<|im_end|>\n \
            <|im_start|>user\n {user_prompt}<|im_end|>\n \
            <|im_start|>assistant\n"
    if (SupportedModels.Mistral7B == selected_model):
        prompt = f"<s>[INST]{user_prompt}[/INST]"
    if (SupportedModels.Gemma_1_1_it == selected_model):
        prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model"

   # completion = client.completions.create(
   #     model="local-model",  # this field is currently unused
   #     prompt=prompt,
   #     top_p=0.95,
   #
   #     temperature=summarizer_temp,
   #     max_tokens=500,
   #     n=100,
   #     stream=False,
   #
   # )
    # narritive = completion.choices[0].text

    ollama_completion = ollama_client.chat.completions.create(
        model="mixtral",  # this field is currently unused
        messages=[{
            'role': 'user',
            'content': prompt
        }],
        top_p=0.95,
        temperature=summarizer_temp,
        max_tokens=500,
        n=100,
        stream=False,
    )

    narritive = ollama_completion.choices[0].message.content

    checkpoint_text_filename = f"{file_name}_narritive_checkpoint.txt"
    with open(checkpoint_text_filename, "w", encoding='utf-8') as checkpoint_text_file:
        checkpoint_text_file.writelines(narritive)

    time_start = time.time()

    processor = AutoProcessor.from_pretrained(bark_model)
    # model = AutoModel.from_pretrained("suno/bark")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Infrence on device: {device}")

    model = BarkModel.from_pretrained(
        bark_model,
        torch_dtype=torch.float32).to(device)
    model = model.to_bettertransformer()

    script = narritive.replace("\n", " ").strip()

    sentences = nltk.sent_tokenize(script)

    # voice_preset = "v2/en_speaker_9"
    # voice_preset = "v2/en_speaker_6"
    # voice_preset = "v2/en_speaker_1"
    voice_preset = "v2/en_speaker_3"  # british male
    # voice_preset = "v2/fr_speaker_2"
    logger.info(f"Using speaker: {voice_preset}")

    sample_rate = model.generation_config.sample_rate

    silence = np.zeros(int(0.15 * sample_rate))  # quarter second of silence

    logger.info(
        f"Running voice synthesis for file: {file_name}, in {len(sentences)} chunks.")

    pieces = []
    i = 0
    for sentence in sentences:
        sentence_loop_start = time.time()
        inputs = processor(
            text=[
                sentence
            ],
            voice_preset=voice_preset,
            return_tensors="pt",
        )

        audio_array = model.generate(
            **inputs,
            pad_token_id=model.generation_config.eos_token_id)

        audio_array = audio_array.cpu().numpy().squeeze()
        pieces += [audio_array, silence.copy()]
        i = i + 1
        sentence_loop_end = time.time()

        logger.info(
            f"Finished sentence {i}/{len(sentences)} in {(sentence_loop_end - sentence_loop_start) / 60} minutes")

    out = np.concatenate(pieces)
    # save audio to disk, but first take the sample rate from the model config
    write_wav(waw_file_name, sample_rate, out)

    time_end = time.time()
    elapsed = (time_end - time_start) / 60
    logger.info(f"Audio generation: {elapsed} minutes\n")

final_time_end = time.time()
total_elapsed = (final_time_end - final_time_start) / 60
logger.info(f"Finished in: {total_elapsed} minutes\n")
