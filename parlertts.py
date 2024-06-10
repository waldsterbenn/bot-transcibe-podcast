import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler_tts_mini_v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

prompt = """
In the latest episode of Millionærklubben, hosts and guests discuss the current state of the stock market, focusing on GN Store Nord. They share thoughts on capital allocation, inflation rates, and forecasts for American banks. A notable discussion point is the recent purchase of ALCC by Millionaire Club members, described as a high-risk investment with potential significant gains or losses.
The group also discusses expectations for the European Central Bank (ECB) interest rate meeting, suggesting that while no change is expected tomorrow, a reduction could come in the near future. The American inflation rates are seen as crucial in determining whether the US Federal Reserve will lower interest rates.
"""
description = "A British female speaker with an animated normal-pitched voice, in studio environment with clear audio quality. She speaks fast."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(
    input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("sensual_parler_tts_out.wav", audio_arr, model.config.sampling_rate)
