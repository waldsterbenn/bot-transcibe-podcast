# Automated Podcast Note Taking

System for automatically taking notes from podcasts.
It will:

- Take an RSS feed and parse the metadata.
- Download an MP3 file.
- Transcribe using Wisper3.
- Summarize using local OLLAMA model (chunked if needed).
- Generate a report as a Markdown file.
- Copy it to a desired location. Fx note system or storage. (I use Obsidian).

## Local LLM

We let local OLLAMA model handle heavy lifting of the summarization. It should use GPU, both AMD and Nvidia.
The wisper model only runs on Nvidia/CPU. Beacuse it uses Huggingface transformers.
It will flip to use DistilWisper if the audio is in english.
It's also possible to point at LMStudio (which i previously did). Just use OpenAi interface on local host/port.

# Example

[Supertrends example](example-supertrend.md)

# Setup

1.  Install Python 3.11.8
2.  Install conda (i use anaconda) for handling virtual environments.
3.  Run `pip install -r requirements.txt`
4.  Run `python main.py`
5.  Or run `python multi_transcribe.py` to transcribe multiple files defined in podcast_data.json.
