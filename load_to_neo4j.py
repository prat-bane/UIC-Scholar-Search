from neo4j import GraphDatabase
from pathlib import Path
import json
import glob

# --- Configuration ---
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "test1234")
JSON_DIR = "paper_json" # Or the correct relative/absolute path

driver = GraphDatabase.driver(URI, auth=AUTH)

def setup_database(driver):
    """
    Run this once to ensure indexes exist.
    REMOVED try/except to expose errors immediately.
    """
    queries = [
        "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paperId IS UNIQUE",
        "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
        "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
        
        # VECTOR INDEX
        """
        CREATE VECTOR INDEX section_embedding_index IF NOT EXISTS
        FOR (s:Section) ON (s.embedding)
        OPTIONS {indexConfig: {
         `vector.dimensions`: 384,
         `vector.similarity_function`: 'cosine'
        }}
        """,
        
        # FULLTEXT INDEXES
        # Note: This syntax works for Neo4j 5.x. 
        """
        CREATE FULLTEXT INDEX section_text_index IF NOT EXISTS
        FOR (s:Section) ON EACH [s.text]
        """,
        """
        CREATE FULLTEXT INDEX topic_name_index IF NOT EXISTS
        FOR (t:Topic) ON EACH [t.name]
        """,
        """
        CREATE FULLTEXT INDEX author_name_index IF NOT EXISTS
        FOR (a:Author) ON EACH [a.name]
        """
    ]
    
    print("⚙️ Configuring Database...")
    with driver.session() as session:
        for q in queries:
            # RUN WITHOUT TRY/EXCEPT so errors crash the script visibly
            print(f"   Running: {q[:50]}...") 
            session.run(q)
            
    print("✅ Database configured successfully.")

def load_paper_optimized(tx, doc, paper_id):
    """
    Loads a single paper JSON into Neo4j.
    UPDATED: Abstract removed from Parent Node.
    """
    query = """
    // 1. Merge the Paper (Metadata Only)
    MERGE (p:Paper {paperId: $id})
    SET p.title = $title,
        p.filename = $filename,
        p.year = $year,
        p.link = $link
    
    // 2. Handle UIC Authors
    WITH p
    UNWIND $uic_authors as uic_auth
    WITH p, uic_auth, split(uic_auth.name, ',') AS name_parts
    WITH p, uic_auth, trim(name_parts[1]) + " " + trim(name_parts[0]) AS full_name
    WHERE full_name <> "" 
    MERGE (a:Author {name: full_name})
    SET a:UIC_Author, a.department = uic_auth.department, a.title = uic_auth.title
    MERGE (a)-[:AUTHORED]->(p)

    // 3. Handle Topics (Keywords)
    WITH p
    UNWIND $keywords as kw
    WITH p, toLower(trim(kw)) as clean_kw
    WHERE clean_kw <> ""
    MERGE (t:Topic {name: clean_kw})
    MERGE (p)-[:HAS_TOPIC]->(t)

    // 4. Handle Sections (Searchable Content)
    WITH p
    UNWIND $sections as sec
    // Create a unique Section ID
    WITH p, sec, $id + "_" + sec.name as sec_id
    
    MERGE (s:Section {id: sec_id})
    SET s.name = sec.name,
        s.text = sec.text,
        s.embedding = sec.embedding
    
    MERGE (p)-[:HAS_SECTION]->(s)
    """
    
    tx.run(query, 
           id=paper_id,
           filename=doc["filename"],
           year=doc.get("year"),
           link=doc.get("link"),
           title=doc.get("title", "Unknown Title"),
           uic_authors=doc.get("uic_authors", []),
           keywords=doc.get("keywords", []),
           sections=doc.get("sections", [])
    )

def load_all(json_dir):
    setup_database(driver)

    paths = glob.glob(str(Path(json_dir) / "*.json"))
    print(f"Found {len(paths)} JSON files. Starting import...")

    with driver.session() as session:
        count = 0
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                
                # ID Generation
                paper_id = doc["id"]
                
                session.execute_write(load_paper_optimized, doc, paper_id)
                
                count += 1
                if count % 50 == 0:
                    print(f"   Processed {count} papers...")
            except Exception as e:
                print(f"Error processing {Path(path).name}: {e}")

    print(f"🎉 Finished! Total papers loaded: {count}")

if __name__ == "__main__":
    try:
        load_all(JSON_DIR)
    finally:
        driver.close()