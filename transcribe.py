from transformers import AutoTokenizer
from openai import OpenAI
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModel, AutoModelForCausalLM
# from datasets import load_dataset
from urllib.request import urlretrieve
import os
import time

skip_sts = True
useSystemPrompt = False
audiofile_path = "audio.mp3"
language = "english"
summarizer_temp = 0.3

time_start = time.time()

files = [os.path.join(".", f)
         for f in os.listdir(".") if f.endswith('.mp3')]

if (len(files) == 0):
    #  Supertanker menneske jobmølle "https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:11032411011/312bc5e3ff8870e58217597ca8630c292f7996405c2ac23dd4e296c0711996e8.mp3"
    # Brinkmans briks AI TEASER "https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:11032498006/a152dbe562557bd9668911b762664bcb14cd83b97a7fb416aaac20b8e997ffa0.mp3"
    # DR Orientering "https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:11802410111/1672abbd9f51c47e64d50206c42bf772b1bbeb5556be4973f57c35db3c45544a.mp3"
    # Found my fit 3t "https://traffic.libsyn.com/secure/foundmyfitness/gibala-public.mp3?dest-id=184296"
    # LEX Yann le cun 3t "https://media.blubrry.com/takeituneasy/content.blubrry.com/takeituneasy/lex_ai_yann_lecun_3.mp3"
    # AI Marketing 25m "https://anchor.fm/s/73b430c0/podcast/play/82659313/https%3A%2F%2Fd3ctxlq1ktw2nl.cloudfront.net%2Fstaging%2F2024-1-14%2Fdf1a5d6e-6329-b023-4bf9-71efa93573be.mp3"
    url = "https://media.blubrry.com/takeituneasy/content.blubrry.com/takeituneasy/lex_ai_yann_lecun_3.mp3"
    print("No audio found. Downloading audio.")
    urlretrieve(url, audiofile_path)


time_pre_sts = time.time()
text = "Error no text found"

if skip_sts:
    print("Reading transcribtion from file")
    f = open("result-text.txt", "r")
    text = f.readlines()
    f.close()
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Transcribing audio STS")
    if language == "english":
        args = generate_kwargs = {"language": language}
    else:
        args = generate_kwargs = {"language": language, "task": "translate"}

    # Way to segment file and find lang

    # from pydub import AudioSegment

    # song = AudioSegment.from_mp3("good_morning.mp3")

    # # PyDub handles time in milliseconds
    # ten_minutes = 10 * 60 * 1000

    # first_10_minutes = song[:ten_minutes]

    # first_10_minutes.export("good_morning_10.mp3", format="mp3")
    # wisper that and determine language
    # language = "danish"

    result = pipe(audiofile_path, generate_kwargs=args)
    text = result["text"]
    print(f"Transcribed len: {text}")
    text_file = open("result-text.txt", "w")
    text_file.writelines(text)
    text_file.close()

time_post_sts = time.time()

# Point to the local server
client = OpenAI(base_url="http://localhost:1236/v1", api_key="not-needed")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
max_tokens = 10000  # Maximum tokens for a single chunk
overlap = 500  # Tokens to overlap between chunks to ensure continuity


# Function to chunk the text
def chunk_text(text, max_tokens, overlap):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks


# Chunk the text
chunks = chunk_text(text[0], max_tokens, overlap)

# Summarize each chunk and merge the summaries
merged_summary = ""
for chunk in chunks:
    prompt = f"<s>[INST] Summarize the text:\n{chunk}[/INST]"
    response = client.chat.completions.create(
        model="local-model",  # Adjust according to your setup
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=summarizer_temp,  # Adjust as needed
    )
    merged_summary += response.choices[0].message.content + "\n"

# Save the merged summary chunks to a file
with open("merged-summary-chunks.txt", "w") as file:
    file.write(merged_summary)

# LLM Summarize
system_prompt = "<s>[INST] You are an expert writer, that summarizes transcibed podcast text. \
    You follow these rules: \
    1. The summary should be elaborate and include sections about the important discussions in the podcast. \
    2. Aim for around 700 words for the finished text. \
    3. You output the summary in correct Markdown format, that can be copied into an .md file, whithout need for further modification. \
    4. Use standard Markdown syntax. Reference the provided example of layout. \
    5. Use '<br/>' syntax instead of '\n', for newline. \
    6. You may insert quotes from interessting passages. Use '>' syntax for quotes. \
    7. Arrange the text in sections: Title, Participants, Elaborate summary, Key points, References. \
    8. Finish the text with a bulletpoint list of important points, topics or subjects, form the podcast, that can be used in a word cloud. \
    "

example_layout = " Example of layout:\
<br /># Title 'Podcast'  \
<br /> \
<br />## Participants \
<br /> \
<br />- **Gregg**, Host \
<br />- **Simon Susann**, job title, experiences \
<br /> \
<br />## Elaborate Summary \
<br /> \
<br />Elaborate 500 words summary of the conversation. \
<br />They discuss a lot of interresting things and have a lengthy debate.\
<br />It's a very long conversation, but it is sumarized here.\
<br /> \
<br />## Key Points \
<br /> \
<br />1. **First thing**: Description of first thing. \
<br />2. **Second thing**: Description of another thing. \
<br /> \
<br />## References \
<br /> \
<br />1. First reference \
<br />2. John Doe's book  - 'My book' \
<br /> \
<br />## Word cloud \
<br />- Topic one \
<br />- Essencial knowledge \
<br />- More things that where discussed \
"

user_prompt = f"\n Summarize the text and adhere strictly to the rules: \n{merged_summary} [/INST]"

print("Summarizing")
time_pre_summary = time.time()

history = []
if (useSystemPrompt):
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": example_layout},
        {"role": "user", "content": user_prompt}
    ]
else:
    history = [
        {"role": "user", "content": f"{system_prompt}\n{example_layout}\n{user_prompt}"}]

completion = client.chat.completions.create(
    model="local-model",  # this field is currently unused
    messages=history,
    temperature=summarizer_temp,
)

time_post_summary = time.time()

print(completion.choices[0].message)
text_file = open("result-summary.md", "w")
text_file.writelines(completion.choices[0].message.content)
text_file.close()

time_end = time.time()
# Calculate elapsed times
sts_total = (time_post_sts - time_pre_sts) / 60
summary_total = (time_post_summary - time_pre_summary) / 60
total_elapsed = (time_end - time_start) / 60

# Open a text file to write the performance summary
with open("performance-summary.txt", "w") as text_file:
    text_file.write(f"Start: {time.ctime(time_start)}\n")
    text_file.write(f"STS start: {time.ctime(time_pre_sts)}\n")
    text_file.write(f"STS end: {time.ctime(time_post_sts)}\n")
    text_file.write(f"STS total: {sts_total} minutes\n")
    text_file.write(f"Sum start: {time.ctime(time_pre_summary)}\n")
    text_file.write(f"Sum end: {time.ctime(time_post_summary)}\n")
    text_file.write(f"Sum total: {summary_total} minutes\n")
    text_file.write(f"End: {time.ctime(time_end)}\n")
    text_file.write(f"Total: {total_elapsed} minutes\n")
