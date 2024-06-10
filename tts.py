import time
import numpy as np
from transformers import BarkModel
import torch
from transformers import AutoProcessor, AutoModel
from transformers import pipeline
import scipy
from scipy.io.wavfile import write as write_wav
import nltk
# synthesiser = pipeline("text-to-speech", "suno/bark")

# speech = synthesiser("Hello, my dog is cooler than you!",
#                    forward_params={"do_sample": True})

# scipy.io.wavfile.write(
#    "bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])

time_start = time.time()

processor = AutoProcessor.from_pretrained("suno/bark")
# model = AutoModel.from_pretrained("suno/bark")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BarkModel.from_pretrained(
    "suno/bark",
    torch_dtype=torch.float32).to(device)
model = model.to_bettertransformer()

script = """
In this episode of "The Horrible Man", titled "Episode 1: The Political Mass Murders", the host delves into the minds of those responsible for some of the most heinous acts of political mass murder throughout history. The podcast features interviews with historian and professor Mikkel Thorp, who provides insights into the historical context and ideologies behind these atrocities, as well as psychiatrist Christian Liegen, who offers expert analysis on the psychological and physiological aspects of human violence.
The discussion begins by examining the role that men have played in perpetrating political mass murders throughout history, with a focus on the motivations and justifications behind their actions. The hosts explore the concept of "ordinary men" and how they are often driven to commit unimaginable acts of brutality due to their adherence to a particular ideology or political system.
Throughout the episode, the hosts examine various case studies, including the actions of Adolf Hitler, Joseph Stalin, and other notorious leaders who have been responsible for the deaths of millions of people. The podcast delves into the ways in which these individuals exploited fear and violence to maintain power and control over their respective nations, as well as how they manipulated the perceptions and beliefs of their followers to gain support for their actions.
The discussion also touches upon the role of the bystander and the complex dynamics that can lead ordinary people to become complicit in acts of mass murder. The hosts explore the concept of "structural evil" and how it is often rooted in a deep-seated conviction that one's actions are justified in the name of a greater good or moral cause.
Throughout the episode, the hosts emphasize the importance of understanding the motivations and experiences of those who have committed political mass murders in order to gain a deeper insight into the nature of human violence and the factors that can drive individuals to commit unspeakable acts.
Key Points.
First point: The podcast explores the historical context and ideologies behind political mass murders, with a focus on the role of men in perpetrating these atrocities.
Second point: The hosts examine the motivations and justifications behind the actions of notorious leaders such as Adolf Hitler and Joseph Stalin.
Third point: The discussion touches upon the ways in which fear and violence are exploited to maintain power and control over nations, as well as how individuals can be manipulated into becoming complicit in acts of mass murder.
Fourth point: The podcast highlights the importance of understanding the experiences and motivations of those who have committed political mass murders in order to gain a deeper insight into human violence and the factors that drive individuals to commit such acts.
Fifth point: The role of ordinary men in perpetrating political mass murders is explored, as well as the concept of "structural evil" and how it is often rooted in a deep-seated conviction that one's actions are justified in the name of a greater good or moral cause.
""".replace("\n", " ").strip()

sentences = nltk.sent_tokenize(script)

voice_preset = "v2/en_speaker_9"
voice_preset = "v2/fr_speaker_2"

sample_rate = model.generation_config.sample_rate

silence = np.zeros(int(0.25 * sample_rate))  # quarter second of silence

pieces = []
i = 0
for sentence in sentences:
    inputs = processor(
        text=[
            sentence
        ],
        voice_preset=voice_preset,
        return_tensors="pt",
    )
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    pieces += [audio_array, silence.copy()]
    i = i + 1
    print(f"Finished sentence {i}/{len(sentences)}")

out = np.concatenate(pieces)
# save audio to disk, but first take the sample rate from the model config
write_wav("fr_horrible_bark_generation.wav", sample_rate, out)

time_end = time.time()
total_elapsed = (time_end - time_start) / 60
print(f"Total: {total_elapsed} minutes\n")
