from concurrent.futures import ThreadPoolExecutor
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pytextrank
import spacy
import glob
from bs4 import BeautifulSoup
from grobid_client.grobid_client import GrobidClient
import json
from keybert import KeyBERT
from google import genai
import time


from models.KeywordResponse import KeywordResponse


cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "jinaai"
if cache_dir.exists():
    print(f"Clearing corrupted cache: {cache_dir}")
    shutil.rmtree(cache_dir, ignore_errors=True)
    print("Cache cleared")
# # Load spaCy model and add PyTextRank
# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("textrank")
kw_model = KeyBERT("all-MiniLM-L6-v2")
embedding_model=SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
client= genai.Client(api_key="")

def extract_text_using_grobid_client(pdf_path, outputxml_path):
    from time import perf_counter
    client = GrobidClient(config_path="config.json")
    print("Step 1: Running GROBID (PDF → XML)...")
    t0 = perf_counter()
    client.process("processFulltextDocument", pdf_path, outputxml_path, n=3)
    t1 = perf_counter()
    print(f"GROBID time: {t1 - t0:.2f} seconds")

    print("Step 2: Parsing XML + embeddings + keywords...")
    t2_start = perf_counter()
    extract_info_from_xml(outputxml_path)  # or extract_info_parallel(...)
    t2_end = perf_counter()
    print(f" XML + embedding phase time: {t2_end - t2_start:.2f} seconds")
    


def extract_info_from_xml(xml_path):
   xml_files= glob.glob(str(Path(xml_path) / "*.tei.xml"))
   for xml_file in xml_files:
       with open(xml_file,encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml-xml")  # use XML parser

            title = ""
            title_tag=soup.find("title", {"level": "a", "type": "main"})
            if title_tag:
                title= title_tag.get_text(strip=True)
            sections = []
            abstract = ""
            if soup.find("abstract"):
                abstract = " ".join(soup.find("abstract").stripped_strings)  
                print("Generating embedding for ABSTRACT...")
                abstract_embedding = embedding_model.encode(abstract, show_progress_bar=False).tolist()  
                sections.append({"name": "ABSTRACT", "text": abstract, "embedding": abstract_embedding})


           
            body = soup.find("body")
        
            if not body:
                return []

            # Iterate over every section (div) in the body
            for div in body.find_all("div"):
                
                # --- STEP 1: Get the Exact Section Header ---
                head = div.find("head")
                if head:
                    # We take the text exactly as it is (e.g., "1 INTRODUCTION")
                    header_text = head.get_text(strip=True)
                else:
                    # Fallback if a div has no header (rare but possible)
                    header_text = "No Header"

                # --- STEP 2: The Cleaning (Remove Junk) ---
                # We look for specific tags inside THIS div and delete them.
                tags_to_remove = ["figure", "table", "note", "formula", "biblStruct"]
                
                for tag_name in tags_to_remove:
                    for junk in div.find_all(tag_name):
                        junk.decompose() 

                # --- STEP 3: Extract Remaining Text ---
                paragraph_texts = []
                for p in div.find_all("p"):
                    text = " ".join(p.stripped_strings)
                    if len(text) > 10: 
                        paragraph_texts.append(text)

                full_section_text = "\n".join(paragraph_texts)

                # Only add if there is actual text
                if full_section_text:
                    print(f"Generating embedding for {header_text}...")
                    section_embedding = embedding_model.encode(full_section_text, show_progress_bar=False).tolist()
                    
                    sections.append({
                        "name": header_text,
                        "text": full_section_text,
                        "embedding": section_embedding
                    })      

            # # ---- Extract Introduction ----
            introduction = ""
            for div in soup.find_all("div"):
                head = div.find("head")
                if head and "INTRODUCTION" in head.get_text().upper():
                    introduction += " ".join(div.stripped_strings) + "\n"

            # Extract Authors
            authors = []
            tei_header = soup.find("teiHeader")
            if tei_header:
                for author in tei_header.find_all("author"):
                    persName = author.find("persName")
                    if persName:
                        forename = persName.find("forename")
                        surname = persName.find("surname")
                        name = {
                            "forename": forename.get_text() if forename else "",
                            "surname": surname.get_text() if surname else ""
                        }
                    
                        authors.append(name)
            
            keywords=[]
            combined_text = (abstract + "\n\n" + introduction).strip()
            # if combined_text:
            #     doc = nlp(combined_text)
            #     keywords = [
            #         {"text": phrase.text, "rank": round(phrase.rank, 6)}
            #         for phrase in doc._.phrases[:]   
            #     ]
           
            keywords_list = []
            if combined_text:
                try:
                    # We use the global 'client' here. 
                    # defining 'config' forces the model to follow our KeywordResponse schema
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[combined_text],
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": KeywordResponse,
                            "system_instruction": """
                                You are a technical keyword extractor.
                                Rules:
                                1. Extract specific tools, algorithms, and domain concepts.
                                2. EXCLUDE generic words: 'study', 'paper', 'method', 'results', 'introduction'.
                                3. Return a clean list of strings.
                            """
                        }
                    )
                    # The SDK parses the JSON automatically into our Pydantic object
                    if response.parsed:
                        keywords_list = response.parsed.keywords
                        print(f"Extracted {len(keywords_list)} keywords")
                    else:
                        print(" Empty response from Gemini")

                except Exception as e:
                    print(f" Gemini Error: {e}")

            # ---- Save Data ----
            data = {
                "filename": Path(xml_file).name,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "keywords": keywords_list  ,
                "sections": sections
            }

            out_json = Path(xml_file).with_name(Path(xml_file).stem + "_sections.json")
            with open(out_json, "w", encoding="utf-8") as out:
                json.dump(data, out, indent=2, ensure_ascii=False)

            # out_txt = Path(xml_file).with_name(Path(xml_file).stem + "_sections.txt")
            # with open(out_txt, "w", encoding="utf-8") as out:
            #     out.write("ABSTRACT:\n" + abstract + "\n\nINTRODUCTION:\n" + introduction)

            print(f" Extracted sections from: {Path(xml_file).name}")
        


pdf_path = r"D:\VSCodeProjects\UIC-Scholar-Search\input"
outputxml_path = r"output_xml"
extract_text_using_grobid_client(pdf_path, outputxml_path)
#extract_text_and_all_keywords(pdf_path)
