from text_extractor import extract_text_using_grobid_client
from pathlib import Path
from utils import refresh_directories, get_staff_list
import glob

PDF_INPUT_PATH = 'input'
OUTPUT_XML_PATH = 'output_xml'
JSON_OUTPUT_PATH = 'paper_json'    

if __name__ == "__main__":

    # 1. Clean up previous output directories
    print("--- Cleaning up old output directories ---")
    dirs_to_refresh = [OUTPUT_XML_PATH, JSON_OUTPUT_PATH]
    refresh_directories(dirs_to_refresh)

    # 3. Run the pipeline
    print("--- Starting processing pipeline ---")
    extract_text_using_grobid_client(
        PDF_INPUT_PATH,
        OUTPUT_XML_PATH, 
        JSON_OUTPUT_PATH
    )
    json_files = glob.glob(str(Path(JSON_OUTPUT_PATH) / "*.json"))
    print(f"\n--- Processing complete! ---")
    print(f"Successfully created {len(json_files)} JSON files in '{JSON_OUTPUT_PATH}'.")    