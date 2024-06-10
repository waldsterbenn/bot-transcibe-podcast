import glob
import os
import shutil
import subprocess
import logging
import logging.config

# Load the logging configuration
logging.config.fileConfig('logging.config')
report_output_destination_folder = "C:/Users/ls/Documents/Obsidian/Noter/Podcast notes"

# Get the logger specified in the configuration file
log = logging.getLogger('sampleLogger')
log.info("***************************************")
log.info("*** Starting transcription pipeline ***")
log.info("***************************************")

# Paths to the Python scripts
script1_path = 'find_mp3s.py'
script2_path = 'multi_transcribe.py'
script3_path = 'audio_report_generator.py'

podcast_feeds = {
    # Supertrends
    "1": "https://www.spreaker.com/show/5768334/episodes/feed",
    # Flyvende tallerken
    "2": "https://api.dr.dk/podcasts/v1/feeds/flyvende-tallerken",
    # Supertanker
    "3": "https://api.dr.dk/podcasts/v1/feeds/supertanker",
    # Lex Firedman
    "4": "https://lexfridman.com/feed/podcast/",
    # Found my fitness
    "5": "https://podcast.foundmyfitness.com/rss.xml",
    # The Public Philosopher
    "6": "https://podcasts.files.bbci.co.uk/b01nmlh2.rss",
    # den-grusomme-mand DR
    "7": "https://api.dr.dk/podcasts/v1/feeds/den-grusomme-mand",
    # Bakspejl DR
    "8": "https://api.dr.dk/podcasts/v1/feeds/bakspejl",
    # Millionærklubben gennem mærkelig rss
    "9": "https://www.omnycontent.com/d/playlist/1283f5f4-2508-4981-a99f-acb500e64dcf/27dfeb66-f61a-4fcc-aa6d-ad0800b05139/dc61232c-7e07-438e-981a-ad0800b05142/podcast.rss",
    # Body IO Keifer
    "10": "https://feeds.soundcloud.com/users/soundcloud:users:75058403/sounds.rss"
}

# String parameter to pass to the first script
param_to_first_script = podcast_feeds["1"]


def run_script(script_path, args=None):
    if args is None:
        args = []
    try:
        # Command includes the script path and any arguments
        command = ['python', script_path] + args
        # Run the script
        subprocess.run(command, check=True,
                       text=True, capture_output=True)
        log.info(f"Script {script_path} finished successfully.")
    except subprocess.CalledProcessError as e:
        log.error(f"Script {script_path} failed.")
        log.exception("Error:", e.stderr)


# Run the first script with the string parameter
if os.path.exists("podcast_data.json") == False:
    run_script(script1_path, [param_to_first_script])

# Run the second script without any parameters
run_script(script2_path)

# run_script(script3_path)

log.info("Copying summary files")

# Source directory path
source_dir = './summaries'

# Destination directory path
dest_dir = os.path.join(
    report_output_destination_folder, "unsorted")
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)
# Get a list of all Markdown files in the source directory
md_files = glob.glob(os.path.join(source_dir, '**', '*.md'), recursive=True)

# Copy each Markdown file to the destination directory
for md_file in md_files:
    # Extract the filename without the path and extension
    filename = os.path.basename(md_file)
    # Construct the destination file path
    dest_file = os.path.join(dest_dir, filename)
    # Copy the file to the destination directory
    shutil.copy2(md_file, dest_file)
    log.info(f'Copied {md_file} to {dest_file}')

log.info('All Markdown files have been copied.')
log.info("***************************************")
log.info("*** Transcription pipeline finished ***")
log.info("***************************************")
