import requests
import xml.etree.ElementTree as ET
import json
import sys
import logging
import logging.config

# Load the logging configuration
logging.config.fileConfig('logging.config')

# Get the logger specified in the configuration file
logger = logging.getLogger('sampleLogger')

# URL of the RSS feed
rss_url = "https://www.omnycontent.com/d/playlist/1283f5f4-2508-4981-a99f-acb500e64dcf/27dfeb66-f61a-4fcc-aa6d-ad0800b05139/dc61232c-7e07-438e-981a-ad0800b05142/podcast.rss"
rss_url = "https://www.spreaker.com/show/5768334/episodes/feed"

if len(sys.argv) > 1:
    # The first argument after the script name
    rss_url = sys.argv[1]

# Download the RSS feed XML document
response = requests.get(rss_url)
response.encoding = 'utf-8'
rss_xml = response.text

# Parse the XML document
root = ET.fromstring(rss_xml)

logger.info(
    f"Found RSS feed with {len(root.findall('./channel/item'))} podcasts")


# Namespace to parse elements properly
namespaces = {'itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd'}

title = ""
language = ""
categories = []  # List to store all categories
for item in root.findall('./channel'):
    title = item.find('title').text if item.find(
        'title') is not None else 'No title'
    language = item.find('language').text if item.find(
        'language') is not None else 'en'

    episode_type = None
    type = item.find('./itunes:type', namespaces)
    if type is not None:
        episode_type = type.text

    # Collecting all categories and itunes categories
    category_elements = item.findall('category')
    itunes_category_elements = item.findall('.//itunes:category', namespaces)

    for cat in category_elements:
        if cat.text is not None:
            categories.append(cat.text)

    for itunes_cat in itunes_category_elements:
        text = itunes_cat.get('text')
        if text is not None:
            categories.append(text)
        # Handle nested itunes categories
        nested_cats = itunes_cat.findall('.//itunes:category', namespaces)
        for nested_cat in nested_cats:
            nested_text = nested_cat.get('text')
            if nested_text is not None:
                categories.append(nested_text)

categories_text = ", ".join(list(set(categories)))

# Collect metadata and MP3 URLs for each item
podcast_data = []
items = root.findall('./channel/item')
number_of_items = len(items)
for item in items:
    try:
        # Extract the desired metadata and the MP3 URL
        data = {
            'title': item.find('title').text,
            'subtitle': item.find('./itunes:subtitle', namespaces).text if item.find('./itunes:subtitle', namespaces) is not None else None,
            'description': item.find('description').text if item.find('description') is not None else None,
            'link': item.find('link').text if item.find('link') is not None else None,
            'pubdate': item.find('pubDate').text if item.find('pubDate') is not None else None,
            'author': item.find('./itunes:author', namespaces).text if item.find('./itunes:author', namespaces) is not None else None,
            'duration': item.find('./itunes:duration', namespaces).text if item.find('./itunes:duration', namespaces) is not None else None,
            'type': episode_type,
            'mp3_url': item.find('enclosure').get('url') if item.find('enclosure') is not None else None,
            'language': language,
            # Ensures no duplicate categories
            'categories': categories_text,
            'podcasttitle': title,
            # last in list is expected to be episode 1
            'podcastnumber': number_of_items - len(podcast_data)
        }
        podcast_data.append(data)
    except RuntimeError as e:
        logger.error(f"Error in item: {item}")
        logger.exception(e)

# Write the data object to a JSON file
with open('podcast_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(podcast_data, json_file, indent=4)

logger.info(f"Podcast data has been written to podcast_data.json")
