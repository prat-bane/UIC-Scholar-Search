import shutil
import fitz  # PyMuPDF
from pathlib import Path
from sentence_transformers import SentenceTransformer
import glob
from bs4 import BeautifulSoup, Tag
from grobid_client.grobid_client import GrobidClient
import json
from google import genai
import tqdm
import time
from concurrent.futures import ThreadPoolExecutor 
from threading import Lock# Import needed for parallelism
from typing import cast, Union, List, Dict, Tuple
import os, math
from utils import get_author_for_pdf, get_uic_staff_details, get_year, get_paper_link
from dotenv import load_dotenv
load_dotenv()

import logging
# Suppress INFO logs from the Gemini client, only show WARNING or higher.
# The library uses both 'google.generativeai' and 'google_genai' for logging.
logging.getLogger('google_genai').setLevel(logging.WARNING)


from models.KeywordResponse import KeywordResponse

# --- CONSTANTS ---
MAX_GEMINI_CONTEXT_LENGTH = 6000
GEMINI_SYSTEM_INSTRUCTION = '''Extract specific technical tools, algorithms, and domain concepts.
Exclude generic words like 'paper', 'method'. Focus on Topics related to the research (e.g., "Natural Language Processing", "Computer Vision", "Human-Computer Interaction", etc).
Return clean list.'''
TAGS_TO_REMOVE_FROM_SECTIONS = ["figure", "table", "note", "formula", "biblStruct"]
UNIQUE_PAPERS_CSV = 'research_paper_unique.csv' 

# --- 1. GLOBAL SETUP ---
# Clear cache if needed (Keep your existing logic)
cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "jinaai"
if cache_dir.exists():
    try:
        shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception:
        pass

model_lock= Lock()
# Load models globally so threads share them (Saving RAM)
print("Loading Models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

uic_authors_details: List[Dict] = get_uic_staff_details(UNIQUE_PAPERS_CSV)

def get_uic_author_details(authors: list[str]) -> List[Dict]:
    author_details: List[Dict] = []
    for author_detail in uic_authors_details:
        if author_detail['name'] in authors:
            author_details.append(author_detail)
    return author_details

def get_author_list(tei_header: Union[Tag, None], paper_id: str) -> Tuple[List[Dict], List[Dict]]:
    authors = []
    uic_authors = []
    if not tei_header:
        return [], []

    for author in tei_header.find_all("author"):
        persName = author.find("persName")
        if not persName:
            continue

        forename_tag = persName.find("forename")
        surname_tag = persName.find("surname")
        forename = forename_tag.get_text().strip() if forename_tag else ""
        surname = surname_tag.get_text().strip() if surname_tag else ""

        authors.append({
            "forename": forename,
            "surname": surname
        })

    uic_authors = get_author_for_pdf(paper_id) 
    author_details = get_uic_author_details(uic_authors)
    return authors, author_details

# --- 2. WORKER FUNCTION (Runs inside a thread) ---
def process_single_xml_file(xml_file, json_output_path) -> str:
    try:
        with open(xml_file, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml-xml")

        filename = Path(xml_file).name
        paper_id = filename.split(' ')[0]

        year = get_year(paper_id)
        paper_link = get_paper_link(paper_id)

        # A. Extract Title
        title = "Unknown Title"
        title_tag = soup.find("title", {"level": "a", "type": "main"})
        if title_tag:
            title = title_tag.get_text(strip=True)

        sections = []
        
        # B. Extract Abstract
        abstract = ""
        abstract_tag = soup.find("abstract")
        if abstract_tag:
            abstract = " ".join(abstract_tag.stripped_strings)
            if abstract:
                # Generate Embedding
                with model_lock:
                    abstract_embedding = embedding_model.encode(abstract, show_progress_bar=False).tolist()
                    sections.append({
                        "name": "ABSTRACT", 
                        "text": abstract, 
                        "embedding": abstract_embedding
                    })

        # C. Extract Body
        body = soup.find("body")
        if body:
            for div in body.find_all("div"):
                # Header
                head = div.find("head")
                header_text = head.get_text(strip=True) if head else "Body Section"

                # Cleaning
                for tag_name in TAGS_TO_REMOVE_FROM_SECTIONS:
                    for junk in div.find_all(tag_name):
                        junk.decompose() 

                # Text Extraction
                paragraph_texts = []
                for p in div.find_all("p"):
                    text = " ".join(p.stripped_strings)
                    if len(text) > 20: 
                        paragraph_texts.append(text)

                full_section_text = "\n".join(paragraph_texts)

                if full_section_text:
                    # Generate Embedding
                    with model_lock:
                        section_embedding = embedding_model.encode(full_section_text, show_progress_bar=False).tolist()
                        sections.append({
                            "name": header_text,
                            "text": full_section_text,
                            "embedding": section_embedding
                        })

        # D. Extract Authors
        tei_header = soup.find("teiHeader")
        authors, uic_authors = get_author_list(tei_header, paper_id)  # Pass list_of_staff if available

        # E. Extract Keywords (Gemini)
        keywords_list = []
        
        # Build Context: Abstract + First Section (Intro)
        context_text = ""
        if len(sections) > 0: context_text += sections[0]["text"]
        if len(sections) > 1: context_text += "\n" + sections[1]["text"]

        if context_text:
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[context_text[:MAX_GEMINI_CONTEXT_LENGTH]], # Limit context
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": KeywordResponse,
                        "system_instruction": GEMINI_SYSTEM_INSTRUCTION
                    }
                )
                if response.parsed:
                    parsed = cast(KeywordResponse, response.parsed)
                    keywords_list = parsed.keywords
            except Exception as e:
                print(f"Gemini Error {Path(xml_file).name}: {e}")

        # F. Save Data
        data = {
            "id": paper_id,
            "link": paper_link,
            "filename": Path(xml_file).name,
            "title": title,
            "year": year,
            "authors": authors,
            "uic_authors": uic_authors,
            "keywords": keywords_list,
            "sections": sections
        }

        # Derive the base filename (e.g., "ray-chowdhury23b" from "ray-chowdhury23b.grobid.tei.xml")
        base_name = Path(xml_file).name.split('.')[0]
        json_output_path = Path(json_output_path)
        out_json = json_output_path / f"{base_name}.json"
        with open(out_json, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
            
        return f"{Path(xml_file).name}"

    except Exception as e:
        return f"Error {Path(xml_file).name}: {e}"


# --- 3. MANAGER FUNCTIONS ---

def extract_info_parallel(xml_path, json_output_path):
    # 1. Get the list of files
    xml_path = Path(xml_path)
    xml_files = glob.glob(str(xml_path / "*.tei.xml"))
    print(f"Found {len(xml_files)} XML files. Processing with 5 threads...")
    
    # Create the output directory for JSON files
    print(f"Saving JSON files to: {json_output_path}")
    # 2. Map the list of files to the Worker Function
    # We use ThreadPoolExecutor because Gemini (Network) and Embeddings (Torch) release GIL
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Pass both the xml_file and the output path to the worker
        results = list(tqdm.tqdm(executor.map(lambda f: process_single_xml_file(f, json_output_path), xml_files), total=len(xml_files)))


def batch_list(data: list, batch_size: int):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def extract_text_using_grobid_client(pdf_path, outputxml_path, json_output_path='paper_json'):
    from time import perf_counter 
    client = GrobidClient(config_path="config.json")
    
    pdf_files = glob.glob(str(Path(pdf_path) / "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process.")

    # Process in smaller batches to avoid overwhelming the client/server
    batch_size = 20 
    batches = list(batch_list(pdf_files, batch_size))
    outputxml_path = Path(outputxml_path)
    
    # --- Performance Timers ---
    total_start_time = perf_counter()
    total_grobid_time = 0
    total_xml_processing_time = 0
    
    with tqdm.tqdm(total=len(pdf_files), desc="Overall Processing") as pbar:
        for i, batch in enumerate(batches):
            pbar.set_description(f"Batch {i+1}/{len(batches)}")
            
            # --- Step 1: Run GROBID for the batch (PDF -> XML) ---
            grobid_start = perf_counter()
            batch_input_dir = outputxml_path / "temp_batch_input"
            batch_input_dir.mkdir(exist_ok=True)
            for pdf_file in batch:
                shutil.copy(pdf_file, batch_input_dir)
            
            client.process("processFulltextDocument", str(batch_input_dir), outputxml_path, n=5)
            shutil.rmtree(batch_input_dir) # Clean up the temp PDF dir
            grobid_end = perf_counter()
            total_grobid_time += (grobid_end - grobid_start)

            # --- Step 2: Process the newly created XMLs for this batch ---
            xml_start = perf_counter()
            extract_info_parallel(outputxml_path, json_output_path)
            xml_end = perf_counter()
            total_xml_processing_time += (xml_end - xml_start)
            
            # --- Step 3: Clean up the processed XML files ---
            for xml_file in glob.glob(str(outputxml_path / "*.tei.xml")):
                os.remove(xml_file)

            pbar.update(len(batch))
            
    total_end_time = perf_counter()
    
    # --- Print Performance Summary ---
    print("\n--- Processing Complete ---")
    print(f"Total GROBID (PDF->XML) time: {total_grobid_time:.2f} seconds")
    print(f"Total XML Parsing & Embedding time: {total_xml_processing_time:.2f} seconds")
    print(f"Total pipeline time: {total_end_time - total_start_time:.2f} seconds")