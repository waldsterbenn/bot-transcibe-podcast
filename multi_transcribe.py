from datetime import datetime
from transformers import AutoTokenizer
from openai import OpenAI
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModel, AutoModelForCausalLM
from urllib.request import urlretrieve
import os
import time
import re
import shutil
from enum import Enum
import json
from llama_index.llms.ollama import Ollama
import logging
import logging.config

# Load the logging configuration
logging.config.fileConfig('logging.config')

# Get the logger specified in the configuration file
logger = logging.getLogger('sampleLogger')

# Brug: Kræver at der hostes en model i LM studio eller Ollama.
# Find url til mp3 som skal transcribes og summarizes.
# Chunker transcribe text hvis den er over 10000 tokens (configurable)


class LlmModelConfig:
    def __init__(self, name, context_window):
        self.name = name
        self.context_window = context_window


class SupportedModels(Enum):
    Mistral_7B_v1 = 1
    Mistral_7B_v2 = 2
    Dolphin_2_8_mistral_v02 = 3
    Hermes_2_Pro_Mistral = 4
    Gemma_1_1_it = 5
    Mixtral_8x_7b = 6
    WizardLm2_7b = 7
    Llama3_8b = 8
    Llama3_70b = 9
    Llama3_gradient = 10
    Phi3_medium = 11


# Før man kan bruge andre modeller skal man hente dem via "ollama pull <modelname>"
llm_models_config = {
    SupportedModels.Mistral_7B_v1:  LlmModelConfig("mistral7b", 8192),
    SupportedModels.Mistral_7B_v2:  LlmModelConfig("mistral7b", 32000),
    SupportedModels.Dolphin_2_8_mistral_v02:  LlmModelConfig("dolphin-mistral", 32000),
    SupportedModels.Hermes_2_Pro_Mistral:  LlmModelConfig("hermes-2-mistral", 32000),
    SupportedModels.Mixtral_8x_7b:  LlmModelConfig("mixtral", 32000),
    SupportedModels.WizardLm2_7b:  LlmModelConfig("wizardlm2:7b", 8192),

    # https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md
    SupportedModels.Llama3_8b:  LlmModelConfig("llama3:8b-instruct-q8_0", 8192),
    SupportedModels.Llama3_70b:  LlmModelConfig("llama3:70b-instruct-q4_K_M", 8192),
    SupportedModels.Llama3_gradient:  LlmModelConfig("llama3-gradient:8b-instruct-1048k-q6_K", 32000),
    SupportedModels.Phi3_medium:  LlmModelConfig("phi3:14b", 32000),
}

current_llm_model_key = SupportedModels.Llama3_8b

delete_merged_summaries = True
language = "danish"
summarizer_temp = 0.3

prompt_token_length = 1700
# The llm context window configured in LM Studio or default for model
llm_context_window_size = llm_models_config[current_llm_model_key].context_window - \
    prompt_token_length

# Maximum tokens for a single chunk
chunker_overlap = 100  # Tokens to overlap between chunks to ensure continuity

logger.info(
    f"Starting summarizer. LLM: {llm_models_config[current_llm_model_key].name}. Context Window: {llm_models_config[current_llm_model_key].context_window}. Temperature: {summarizer_temp}")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased-finetuned-sst-2-english')


# Supertanker menneske jobmølle "https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:11032411011/312bc5e3ff8870e58217597ca8630c292f7996405c2ac23dd4e296c0711996e8.mp3"
# Brinkmans briks AI TEASER "https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:11032498006/a152dbe562557bd9668911b762664bcb14cd83b97a7fb416aaac20b8e997ffa0.mp3"
# DR Orientering "https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:11802410111/1672abbd9f51c47e64d50206c42bf772b1bbeb5556be4973f57c35db3c45544a.mp3"
# Found my fit 3t "https://traffic.libsyn.com/secure/foundmyfitness/gibala-public.mp3?dest-id=184296"
# LEX Yann le cun 3t "https://media.blubrry.com/takeituneasy/content.blubrry.com/takeituneasy/lex_ai_yann_lecun_3.mp3"
# AI Marketing 25m "https://anchor.fm/s/73b430c0/podcast/play/82659313/https%3A%2F%2Fd3ctxlq1ktw2nl.cloudfront.net%2Fstaging%2F2024-1-14%2Fdf1a5d6e-6329-b023-4bf9-71efa93573be.mp3"
# Supertrends ufo "https://api.spreaker.com/download/episode/58706293/supertrends_ufo_spreaker.mp3",
# Supertrends AI "https://api.spreaker.com/download/episode/58863815/supertrends_mennesker_og_k_d.mp3",
# "[https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:16122011705/9a3f1a4c93a985779b62eb7b7d6b1b5b2e773272c12fd01ec1754989e9d1255f.mp3" (5: 5 - Den forkrøblede mand)
# "[https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:16122011704/cfca2952896d30c66355c57f02a95f2293adf229332629c4bb9b0530618e3144.mp3" (4: 5 - Fra dyrplagere til drabsmænd)
# "[https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:16122011703/daf809adb64a948a280e64049c67b3d111fe634c781aca86b60cc2965aa2eaeb.mp3" (3: 5 - Den voldelige mand)
# "[https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:16122011702/bf2382684e39efea2332bb19c5ae4554834c9bdc187cbcdbc9dc54e856ce4d6f.mp3" (2: 5 - De små mordere)
# "[https://api.dr.dk/podcasts/v1/assets/urn:dr:podcast:item:16122011701/5a0edb9fac3694c0ab176113039b2df8bac47f4f1417f04985f7eb75e089832d.mp3" (1: 5 - Den politiske massemorder)


podcast_data_collection = [
    # {
    #    "title": "Invester i fremtiden",
    #    "subtitle": "Liselotte og Lars samler op p\u00e5 de vigtigste game-changers og megatrends i forhold til demografi, arbejdsliv, uddannelse, sundhed, mobilitet og teknologi. Endelig f\u00e5r dem, der t\u00f8r stole p\u00e5 at verden bliver bedre mod 2050 konkrete investeringstips....",
    #    "description": "Liselotte og Lars samler op p\u00e5 de vigtigste game-changers og megatrends i forhold til demografi, arbejdsliv, uddannelse, sundhed, mobilitet og teknologi. Endelig f\u00e5r dem, der t\u00f8r stole p\u00e5 at verden bliver bedre mod 2050 konkrete investeringstips.<br /><br /><b>V\u00e6rter:</b><br />Liselotte Lyngs\u00f8 - <a href=\"https://www.linkedin.com/in/liselotte-lyngs%C3%B8-03a205/\" target=\"_blank\" rel=\"noreferrer noopener\">LinkedIn</a><br /><br />Iv\u00e6rks\u00e6tter og forfatter Lars Tvede<b> </b>- <a href=\"https://www.linkedin.com/in/lars-tvede/\" target=\"_blank\" rel=\"noreferrer noopener\">LinkedIn</a><br /><b><br />Tilrettel\u00e6gger:</b><br />Liselotte Lyngs\u00f8 og Denis Rivin - <a href=\"https://www.linkedin.com/in/denisrivin/\" target=\"_blank\" rel=\"noreferrer noopener\">LinkedIn</a><br /><br /><b>Producent:</b><br />Kim Pihl-Vester - <a href=\"https://www.linkedin.com/in/kimpihlvester/\" target=\"_blank\" rel=\"noreferrer noopener\">LinkedIn</a><br /><br /><b>Klipper:<br /></b>Anne Lund Larsen - <a href=\"https://www.linkedin.com/in/anne-lund-larsen-a21058295/\" target=\"_blank\" rel=\"noreferrer noopener\">Linkdin</a>",
    #    "link": "https://www.spreaker.com/episode/invester-i-fremtiden--59111180",
    #    "pubdate": "Sat, 23 Mar 2024 05:00:00 +0000",
    #    "author": "24syv",
    #    "duration": "3247",
    #    "mp3_url": "https://api.spreaker.com/download/episode/59111180/finale_rettet.mp3",
    #    "language": "da",
    #    "categorys": "Science",
    #    "podcasttitle": "Supertrends"
    # },
]

if (len(podcast_data_collection) == 0 and os.path.exists("podcast_data.json")):
    with open("podcast_data.json", "r", encoding='utf-8') as json_file:
        podcast_data_collection = json.load(json_file)

loop_time_start = time.time()


def sanitize_filename(filename):
    # Define a regular expression pattern to match invalid characters
    invalid_chars_pattern = r'[<>:"/\\|?*\x00-\x1F\x7F]'

    # Replace invalid characters with an underscore
    sanitized_filename = re.sub(invalid_chars_pattern, '', filename)

    # Remove leading and trailing whitespaces and dots
    sanitized_filename = sanitized_filename.strip(' .')

    return sanitized_filename


def format_duration(duration):
    # Check if duration is already in tt:mm:ss format
    if ':' in duration:
        return duration  # Assume it's already in the correct format
    else:
        # Convert from total seconds to hh:mm:ss
        total_seconds = int(duration)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"


def reduce_overall_size(ollama_client, tokenizer, chunker_max_tokens, chunker_overlap, partials):

    # Function to chunk the text
    def chunk_text(text, max_tokens, overlap):
        tokens = tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk = tokens[i:i + max_tokens]
            chunks.append(tokenizer.convert_tokens_to_string(chunk))
        return chunks

    merged_summary = "\n".join(partials)
    tokens_in_full_text = len(tokenizer.tokenize(merged_summary))
    while (tokens_in_full_text > llm_context_window_size):
        logger.info(
            f"Will have to chunk text, since context window is {llm_context_window_size} tokens and text is {tokens_in_full_text}. Overshoot = {tokens_in_full_text-llm_context_window_size} tokens")

        # Chunk the text
        chunks = chunk_text(
            merged_summary, chunker_max_tokens, chunker_overlap)

        number_of_chunks = len(chunks)
        max_tokens_pr_chunk = int(chunker_max_tokens / number_of_chunks)
        merged_summary = ""
        chunk_counter = 0
        # Summarize each chunk and merge the summaries
        for chunk in chunks:
            chunk_counter += 1
            logger.info(
                f"Summarizing chunk {chunk_counter}/{number_of_chunks}")

            prompt = f"""¤¤¤{chunk}¤¤¤\n\n
                Make a detailed recap of this part of the podcast (delimited by ¤¤¤), it can be less than, but no more than {max_tokens_pr_chunk} words.
                Ignore advertisements, sponsers and other promotion.
                """

            ollama_completion = ollama_client.complete(prompt)
            summarytext = ollama_completion.text
            merged_summary += summarytext + "\n"
        tokens_in_full_text = len(tokenizer.tokenize(merged_summary))
    return merged_summary

# Chunk into 30 mins of text. 15min audio = ~2100 tokens @ 140 words/min


def process_minute_chunks(ollama_client, tokenizer, input_text, original_description):

    # Function to chunk the text
    def chunk_text(text, max_tokens, overlap):
        tokens = tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk = tokens[i:i + max_tokens]
            chunks.append(tokenizer.convert_tokens_to_string(chunk))
        return chunks

    # Chunk the text
    total_text_tokens = len(tokenizer.tokenize(input_text))
    if llm_context_window_size > total_text_tokens:
        return [input_text]
    else:
        chunks = chunk_text(input_text, 4000, 100)

    number_of_chunks = len(chunks)
    logger.debug(f"Total text tokens: {total_text_tokens}")
    recap_peices = []
    chunk_counter = 0
    for chunk in chunks:
        chunk_counter += 1
        logger.info(
            f"Summarizing chunk {chunk_counter}/{number_of_chunks}: {len(tokenizer.tokenize(chunk))} tokens")

        prompt = f"""
           You are an expert techical writer, who is provided with a section of a transcribed podcast (delimited by ¤¤¤).
           Here is you're task:
            1. Describe in 1000 words, what is talked about in details.
            2. It's important that details, opinions and ideas are not lost to the reader,
            because this will be used for creating a report of the full podcast later.
            3. Keep a list of non-participtant's names names and places, so we don't mix them up.
            4. Do NOT include and do not mention advertisements, sponsers and other promotion.
            5. Use the original description for reference.
           \n
           Original description:
           ¤¤¤
                {original_description}
           ¤¤¤
           Podcast text:
           ¤¤¤
               {chunk}
           ¤¤¤
           """

        ollama_completion = ollama_client.complete(prompt)
        summarytext = ollama_completion.text
        logger.debug(
            f"Partial summary tokens: {len(tokenizer.tokenize(summarytext))}")
        recap_peices.append(summarytext)
    return recap_peices


def remove_html_tags_except_links(text):
    """
    Remove all HTML tags from the text except for links.
    """
    if text is None:
        return text
    # Remove all HTML tags except <a> tags
    text_no_tags = re.sub(r"(?!<\/?a(?=>|\s.*>))<\/?.*?>", "", text)
    return text_no_tags


def episode_to_markdown(episode):

    subtitle = episode.get("subtitle", "")
    if subtitle is not None:
        subtitle = remove_html_tags_except_links(subtitle)
        txt = "\n".join([f" >{subtitle}"])
        subtitle = f"**Subtitle**\n{txt}<br/>\n"
    else:
        subtitle = ""
    duration = episode.get('duration', '0')
    formatted_duration = format_duration(duration)
    # Construct the markdown content
    markdown = (
        f"**Podcast**: {episode.get('podcasttitle', 'N/A')}\n"
        f"**Episode**: {episode.get('title', 'N/A')}\n"
        f"**Link**: [Episode Link]({episode.get('link', '#')})\n"
        f"**MP3 URL**: [Download]({episode.get('mp3_url', '#')})\n"
        f"**Duration**: {formatted_duration}\n"
        f"**Publication Date**: {episode.get('pubdate', 'N/A')}\n"
        f"**Author**: {episode.get('author', 'N/A')}\n"
        f"**Language**: {episode.get('language', 'N/A')}\n"
        f"**Category**: {episode.get('categories', 'N/A')}\n"
        f"**Type**: {episode.get('type', 'N/A')}\n"
        ""+subtitle
    )

    return markdown


def format_meta_description(episode):
    # Preprocess the description
    description = episode.get("description", "")
    if description == "":
        return ""

    # Remove HTML styling but retain links
    description = remove_html_tags_except_links(description)
    # Split the description into lines for further processing
    description_lines = description.split('\n')
    # Prepare the formatted description with a callout for the first line
    formatted_description = "\n".join(
        [f">{description_lines[0]}"] + [f">{line}" for line in description_lines[1:]])
    return f"{formatted_description}"


def strip_chars(text_array):
    return text_array


# Point to the local server
ollama_client = Ollama(
    model=llm_models_config[current_llm_model_key].name,
    request_timeout=15000.0,
    temperature=summarizer_temp,
    stream=False,
    context_window=llm_models_config[current_llm_model_key].context_window
)

base_data_dir = "podcasts"
current_podcast_folder = ""
podcast_episode_name = ""


def get_file_location(file_name):
    if not current_podcast_folder:
        raise ValueError("Current podcast folder cannot be empty")
    if not podcast_episode_name:
        raise ValueError("Current podcast must have a name, cannot be empty")
    location = os.path.join(
        base_data_dir, current_podcast_folder, podcast_episode_name)
    os.makedirs(location, exist_ok=True)
    return os.path.join(location, file_name)


current_progress_count = 0
for podcast_data in podcast_data_collection:
    logger.info("--------------------------------------------------------------")

    time_start = time.time()
    language = podcast_data['language']
    if (language == 'da'):
        language = 'danish'
    if (language == 'en' or language == 'en-US'):
        language = 'english'

    episode_stamp = ""
    if (podcast_data["type"] == "episodic"):
        fmt = "%Y-%m-%d"
        fancy = "%a, %d %b %Y %H:%M:%S %z"
        date_str = podcast_data['pubdate']
        short_date = datetime.strptime(date_str, fancy).strftime("%Y-%m-%d")
        episode_stamp = f" - {podcast_data['podcastnumber']} - {short_date}"
    else:
        episode_stamp = f" - {podcast_data['podcastnumber']}"

    podcast_episode_name = f"{podcast_data['podcasttitle']}{episode_stamp} - {podcast_data['title']}"
    podcast_episode_name = sanitize_filename(podcast_episode_name)

    current_progress_count += 1
    logger.info(
        f"Processing podcast {current_progress_count}/{len(podcast_data_collection)} - {podcast_episode_name}")

    if os.path.exists(f"./summaries/{podcast_episode_name}.md"):
        logger.info(
            f"Output file already exists in summaries folder, skipping {podcast_episode_name}.md")
        continue  # skip if already created

    current_podcast_folder = podcast_data['podcasttitle']
    audio_file_dest = get_file_location(f"{podcast_episode_name}.mp3")

    if not os.path.exists(audio_file_dest):
        logger.info("Downloading audio to: " + podcast_episode_name)
        urlretrieve(podcast_data['mp3_url'], audio_file_dest)

    # Save json info for later
    if not os.path.exists(get_file_location("podcast_info.json")):
        logger.info("Saving podcast info json")
        with open(get_file_location(f"podcast_info.json"), "w", encoding='utf-8') as f:
            f.writelines(json.dumps(podcast_data))
    # continue
    time_pre_sts = time.time()
    text = "Error no text found"
    if os.path.exists(get_file_location("merged-summary-chunks.txt")) and delete_merged_summaries:
        os.remove(get_file_location("merged-summary-chunks.txt"))
    if os.path.exists(get_file_location(f"{podcast_episode_name}.txt")):
        logger.info("Reading transcribtion from existing file")
        with open(get_file_location(f"{podcast_episode_name}.txt"), "r", encoding='utf-8') as f:
            text = strip_chars(f.readlines())
    else:

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Distil is much quicker 18 min pr 50 min audio vs wisper3 40min pr 50 min audio. But only works for English
        model_id = "openai/whisper-large-v3"
        if (language == "english"):
            model_id = "distil-whisper/distil-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=25,
            batch_size=16,
            return_timestamps=False,
            torch_dtype=torch_dtype,
            device=device,
        )
        logger.info(f"Transcribing audio text-to-speech using {model_id}")
        if language == "english":
            args = generate_kwargs = {"language": language}
        else:
            args = generate_kwargs = {
                "language": language, "task": "translate"}

        result = pipe(audio_file_dest, generate_kwargs=args)
        text = strip_chars([result["text"]])
        logger.info(
            f"Tokens in transcribed text: {len(tokenizer.tokenize(text[0]))}")
        with open(get_file_location(f"{podcast_episode_name}.txt"), "w", encoding='utf-8') as f:
            f.writelines(text)

    time_post_sts = time.time()

    metadata = episode_to_markdown(podcast_data)
    original_description = format_meta_description(podcast_data)

    recap_minutes_peices = []

    if os.path.exists(get_file_location("merged-summary-chunks.txt")):
        logger.info("Reading partials transcribtion from existing file")
        f = open(get_file_location("merged-summary-chunks.txt"),
                 "r", encoding='utf-8')
        data = f.readlines()
        f.close()
        if (len(data) > 1):
            recap_minutes_peices = data
    else:
        recap_minutes_peices = process_minute_chunks(
            ollama_client, tokenizer, text[0], original_description)

        # Save the merged summary chunks to a file
        with open(get_file_location("merged-summary-chunks.txt"), "w", encoding='utf-8') as file:
            file.writelines(recap_minutes_peices)

    merged_summary = reduce_overall_size(
        ollama_client, tokenizer, llm_context_window_size, chunker_overlap, recap_minutes_peices)

    user_prompt = f"""
            You are an analyst and commentator and expert secratary. Your task is to write a report on a podcast.
            You'll find all the information need in the data section, delimited by '¤¤¤'.
            Useful information:
                - 'Original title'.
                - 'Original shownotes': names and information about participants and topics.
                - 'Transcribed text': transcription text from the podcast audio.
            Please DO NOT include advertisements, sponsers and other promotion.

            ¤¤¤
                Original Title:
                {podcast_episode_name}

                Original shownotes:
                {original_description}

                Transcribed text:
                {merged_summary}
            ¤¤¤

            Structure the report like the following example.

            ## Participants
            List the names and details, of the people contributing to the conversation (i.e. name of job, country of origin, field of expertise).
            You might find the names in the 'Original shownotes'. If not, just write you don't know the name.

            ## List of key topics
            Start by porviding a short bullet list of important points, topics and subjects.

            ## Insightful podcast notes
            If there are some things that are supprising, remarkable, stand out or worth remembering, add some insightful notes about it here.

            ## Report
            Make a comprehensive recap of the podcast. It's important that you go into detail and mention all of the topics from the coversation.
            The idea is that the reader should get a good understanding of the information.
        """

    time_pre_summary = time.time()

    token_check = len(tokenizer.tokenize(user_prompt))
    if token_check > llm_context_window_size:
        logger.warn(
            f"Prompt is larger ({token_check}) than LLM context window ({llm_context_window_size}). It may cause problems.")

    logger.info(
        f"Summarizing with LLM query. Prompt {token_check} tokens.")
    try:
        ollama_completion = ollama_client.complete(user_prompt)
        summarytext = ollama_completion.text
    except Exception as ex:
        logger.error(ex)
        continue
    time_post_summary = time.time()

    time_end = time.time()
    # Calculate elapsed times
    sts_total = (time_post_sts - time_pre_sts) / 60
    summary_total = (time_post_summary - time_pre_summary) / 60
    total_elapsed = (time_end - time_start) / 60

    # Open a text file to write the performance summary
    with open(get_file_location("performance-summary.txt"), "w", encoding='utf-8') as text_file:
        text_file.write(f"Start: {time.ctime(time_start)}\n")
        text_file.write(f"STS start: {time.ctime(time_pre_sts)}\n")
        text_file.write(f"STS end: {time.ctime(time_post_sts)}\n")
        text_file.write(f"STS total: {sts_total} minutes\n")
        text_file.write(f"Sum start: {time.ctime(time_pre_summary)}\n")
        text_file.write(f"Sum end: {time.ctime(time_post_summary)}\n")
        text_file.write(f"Sum total: {summary_total} minutes\n")
        text_file.write(f"End: {time.ctime(time_end)}\n")
        text_file.write(f"Total: {total_elapsed} minutes\n")

    lines = [
        "# Original description",
        f"\n{original_description}\n",

        "\n# Summary\n",
        summarytext,

        # "\n\n# By the minute\n",
        # "*Here follows the 30 min recaps, that the above report is based on.*\n\n",
        # "".join(recap_minutes_peices),

        "\n\n# Metadata\n",
        metadata,

        "\n\n# Technical info\n",
        f"- Summary created on: {time.ctime(time_end)}\n",
        f"- Audio transcription time: {sts_total} minutes \n",
        f"- Processing time: {total_elapsed} minutes \n",
        "> [!NOTE] LLM backend system: ollama\n",
        f"<br/> LLM model used for summary: {ollama_completion.raw['model']}\n",
        f"<br/> Context window size: {llm_models_config[current_llm_model_key].context_window}\n",
        f"<br/> Summarizer temperature: {summarizer_temp}\n",
        f"<br/> Token/sec: {ollama_completion.raw['eval_count'] / ollama_completion.raw['eval_duration']}\n",

        "\n\n# Tags\n",
        "#podcast\n",
        f"#podcast-{str.lower(current_podcast_folder)}\n"

    ]

    output_filename = get_file_location(f"{podcast_episode_name}.md")

    try:
        with open(output_filename, "w", encoding='utf-8') as of:
            of.writelines(lines)
        # Create the destination folder if it doesn't exist
        if not os.path.exists('summaries'):
            os.makedirs('summaries')

        shutil.copy(output_filename, 'summaries')
    except IOError:
        logger.error(f"Error invalid filename: {output_filename}")
        try:
            output_filename = get_file_location("result-summary.md")
            with open(output_filename, "w", encoding='utf-8') as of:
                of.writelines(lines)
        except IOError as e:
            logger.exception(e)

    logger.info(f"Performance: Speech-to-Text elapsed: {sts_total} minutes\n")
    logger.info(f"Performance: LLM Summary elapsed: {summary_total} minutes\n")
    logger.info(
        f"Performance: Total podcast processing elapsed: {total_elapsed} minutes\n")
    logger.info(f"Finished processing podcast: {podcast_episode_name}")
    logger.info("--------------------------------------------------------------")


loop_total_elapsed = (time.time() - loop_time_start) / 60
logger.info(
    f"Finished summarizing {len(podcast_data_collection)} podcasts, in {loop_total_elapsed}")
