from neo4j import GraphDatabase
from pathlib import Path
import json
import glob

# --- Configuration ---
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "test1234")
JSON_DIR = r"D:\VSCodeProjects\UIC-Scholar-Search\output_xml"

driver = GraphDatabase.driver(URI, auth=AUTH)

def load_paper_optimized(tx, doc):
    """
    Loads a single paper JSON into Neo4j using your specific constraints.
    """
    query = """
    // 1. Merge the Paper 
    // Uses your 'paper_id_unique' constraint (Paper -> paperId)
    MERGE (p:Paper {paperId: $paper_id})
    SET p.title = $title,
        p.abstract = $abstract,
        p.introduction = $introduction,
        p.filename = $filename
    
    // 2. Handle Authors
    // Uses your 'author_name_unique' constraint (Author -> name)
    WITH p
    UNWIND $authors as auth_data
    // Combine names safely
    WITH p, trim(coalesce(auth_data.forename, "") + " " + coalesce(auth_data.surname, "")) as full_name
    WHERE full_name <> "" 
    MERGE (a:Author {name: full_name}) 
    MERGE (a)-[:AUTHORED]->(p)

    // 3. Handle Topics (Keywords)
    // Uses your 'topic_name_unique' constraint (Topic -> name)
    WITH p
    UNWIND $keywords as kw
    WITH p, toLower(trim(kw)) as clean_kw
    WHERE clean_kw <> ""
    MERGE (t:Topic {name: clean_kw})
    MERGE (p)-[:HAS_TOPIC]->(t)
    """
    
    # Create the ID cleanly in Python
    paper_id = doc["filename"].replace(".grobid.tei.xml", "")
    
    tx.run(query, 
           paper_id=paper_id,
           filename=doc["filename"],
           title=doc.get("title", "Unknown Title"),
           abstract=doc.get("abstract", ""),
           introduction=doc.get("introduction", ""),
           # Pass the raw lists directly to Cypher
           authors=doc.get("authors", []),   
           keywords=doc.get("keywords", [])
    )

def load_all(json_dir):
    paths = glob.glob(str(Path(json_dir) / "*.json"))
    print(f"Found {len(paths)} JSON files. Starting import...")

    with driver.session() as session:
        count = 0
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                
                # Execute the transaction
                session.execute_write(load_paper_optimized, doc)
                
                count += 1
                if count % 50 == 0:
                    print(f"   Processed {count} papers...")
            except Exception as e:
                print(f"Error processing {Path(path).name}: {e}")

    print(f"Finished! Total papers loaded: {count}")

if __name__ == "__main__":
    try:
        load_all(JSON_DIR)
    finally:
        driver.close()