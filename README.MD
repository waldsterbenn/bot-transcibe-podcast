# Automated Podcast Note Taking

System for automatically taking notes from podcasts.
It uses a locally hosted LLM to summarize a transcribed conversation, and use that as a basis to create a report ready for dropping into your notes. Output is a Markdown file for each podcast episode.

The scripts will:

- Take an RSS feed and parse the metadata.
- Download an MP3 file.
- Transcribe audio using Wisper3.
- Summarize the conversation using local a OLLAMA model (chunked if needed). The model used is configurable (`current_llm_model_key`)
- Generate a report as a Markdown file.
- Copy the reports to a desired location on disk. Fx note system or storage (I use Obsidian).

# Local LLM

Lets local OLLAMA model handle heavy lifting of the summarization. It will likely use GPU - both AMD and Nvidia.

The wisper model only runs on Nvidia/CPU. Beacuse it uses Huggingface transformers.
Script will flip to use DistilWisper if the audio is in english. Because it's a lot faster.

It's also possible to point at LMStudio (which i previously did) and use those models instead (if you have your own GGUF file or whatever). Just use OpenAi interface on local host/port. However, you will have to change the client and the code where infrence is invoked.

## Please note

The quality of the output relies on the quality of the audio. If you have a good audio file, it should be fine. But if you have poor audio, wisper will probably output garbage.
Summarization can be flaky and relies on the quality of the model used. LLAMA3-8B or Mistral-7B are pretty good and lightweight. LLAMA3-70B is pretty good, but i have to scrape by on the q4 version on a RX6900XT (16GB), which is slow but gives decent result.

# Example output

[Supertrends example](example-supertrend.md)

# Setup

1.  Install Python 3.11.8
2.  Install conda (i use anaconda) for handling virtual environments.
3.  Run `pip install -r requirements.txt`
4.  Run `python main.py`
5.  Or run `python multi_transcribe.py` to transcribe multiple files defined in podcast_data.json.
