import requests
from bs4 import BeautifulSoup
import time
import os
import logging
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
from tqdm import tqdm 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set the base URL
BASE_URL = "https://iiitd.ac.in/"
TEXT_DIR = "extracted_texts"
DOCUMENTS_DIR = "downloaded_docs"

# Ensure output directories exist
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

def get_robots_txt(base_url):
    robots_url = "https://iiitd.ac.in/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        logging.info("Successfully fetched robots.txt")
    except Exception as e:
        logging.warning(f"Couldn't fetch robots.txt: {e}")
    return rp

def is_allowed(url, rp):
    return rp.can_fetch("*", url)

def download_file(url):
    filename = os.path.join(DOCUMENTS_DIR, os.path.basename(urlparse(url).path))
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            logging.info(f"Downloaded: {filename}")
        else:
            logging.warning(f"Failed to download: {url}")
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")

def extract_and_save_text(soup, url):
    text_content = []
    for tag in ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]:
        for element in soup.find_all(tag):
            clean_text = element.get_text(strip=True)
            text_content.append(clean_text)
    
    if text_content:
        file_name = os.path.join(TEXT_DIR, f"{urlparse(url).netloc}_{urlparse(url).path.replace('/', '_')}.txt")
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(f"Extracted from: {url}\n\n")
            file.write("\n".join(text_content))
        logging.info(f"Saved text from {url} -> {file_name}")

def crawl_website(base_url):
    robots_parser = get_robots_txt(base_url)
    queue = [base_url]
    visited = set()
    
    with tqdm(total=100, desc="Crawling Progress", unit="page") as pbar:
        while queue:
            url = queue.pop(0)
            if url in visited or not is_allowed(url, robots_parser):
                continue
            visited.add(url)
            logging.info(f"Crawling: {url}")
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.text, "html.parser")
                
                extract_and_save_text(soup, url)
                
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(base_url, link["href"])
                    if full_url.endswith((".pdf", ".docx", ".txt")):
                        if is_allowed(full_url, robots_parser):
                            download_file(full_url)
                    elif full_url.startswith(base_url) and full_url not in visited:
                        queue.append(full_url)
                
                pbar.update(1)  # Update progress bar
                time.sleep(10)  # Crawl delay
            except Exception as e:
                logging.error(f"Error crawling {url}: {e}")

crawl_website(BASE_URL)
