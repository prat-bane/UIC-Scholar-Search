import shutil
import fitz  # PyMuPDF
from pathlib import Path
from sentence_transformers import SentenceTransformer
import glob
from bs4 import BeautifulSoup
from grobid_client.grobid_client import GrobidClient
import json
from google import genai
import tqdm
import time
from concurrent.futures import ThreadPoolExecutor 
from threading import Lock# Import needed for parallelism

from models.KeywordResponse import KeywordResponse

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
client = genai.Client(api_key="AIzaSyBx0z6zP252ms_iZrYKLXRNMOxQQYt1OZA")


# --- 2. WORKER FUNCTION (Runs inside a thread) ---
def process_single_xml_file(xml_file):
    try:
        with open(xml_file, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml-xml")

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
                tags_to_remove = ["figure", "table", "note", "formula", "biblStruct"]
                for tag_name in tags_to_remove:
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
        authors = []
        tei_header = soup.find("teiHeader")
        if tei_header:
            for author in tei_header.find_all("author"):
                persName = author.find("persName")
                if persName:
                    forename = persName.find("forename")
                    surname = persName.find("surname")
                    authors.append({
                        "forename": forename.get_text() if forename else "",
                        "surname": surname.get_text() if surname else ""
                    })

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
                    contents=[context_text[:6000]], # Limit context
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": KeywordResponse,
                        "system_instruction": "Extract specific technical tools, algorithms, and domain concepts. Exclude generic words like 'paper', 'method'. Return clean list."
                    }
                )
                if response.parsed:
                    keywords_list = response.parsed.keywords
            except Exception as e:
                print(f"Gemini Error {Path(xml_file).name}: {e}")

        # F. Save Data
        data = {
            "filename": Path(xml_file).name,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "keywords": keywords_list,
            "sections": sections
        }

        out_json = Path(xml_file).with_name(Path(xml_file).stem + "_sections.json")
        with open(out_json, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, ensure_ascii=False)
            
        return f"{Path(xml_file).name}"

    except Exception as e:
        return f"Error {Path(xml_file).name}: {e}"


# --- 3. MANAGER FUNCTIONS ---

def extract_info_parallel(xml_path):
    # 1. Get the list of files
    xml_files = glob.glob(str(Path(xml_path) / "*.tei.xml"))
    print(f"Found {len(xml_files)} XML files. Processing with 5 threads...")

    # 2. Map the list of files to the Worker Function
    # We use ThreadPoolExecutor because Gemini (Network) and Embeddings (Torch) release GIL
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(tqdm.tqdm(executor.map(process_single_xml_file, xml_files), total=len(xml_files)))


def extract_text_using_grobid_client(pdf_path, outputxml_path):
    from time import perf_counter
    client = GrobidClient(config_path="config.json")
    print("Step 1: Running GROBID (PDF → XML)...")
    t0 = perf_counter()
    client.process("processFulltextDocument", pdf_path, outputxml_path, n=5)
    t1 = perf_counter()
    print(f" GROBID time: {t1 - t0:.2f} seconds")

    print("Step 2: Parsing XML + embeddings + keywords...")
    t2_start = perf_counter()
    extract_info_parallel(outputxml_path)  # or extract_info_parallel(...)
    t2_end = perf_counter()
    print(f" XML + embedding phase time: {t2_end - t2_start:.2f} seconds")


# --- 4. EXECUTION ---
if __name__ == "__main__":
    pdf_path = r"D:\VSCodeProjects\UIC-Scholar-Search\input"
    outputxml_path = r"output_xml"
    
    # Create output dir if not exists
    Path(outputxml_path).mkdir(parents=True, exist_ok=True)
    
    extract_text_using_grobid_client(pdf_path, outputxml_path)