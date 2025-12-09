from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import uvicorn
import os

# --- CONFIGURATION ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "test1234") 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

# --- DATA MODELS ---
class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class AuthorSearchRequest(BaseModel):
    query: str
    author_name: str
    limit: int = 10

class SearchResult(BaseModel):
    professor: str
    paper_title: str
    matched_section: str
    snippet: str
    score: float
    source_type: str 

# --- GLOBAL VARIABLES ---
driver = None
model = None
kw_model = None

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Search API...")
    global driver, model, kw_model
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        print("Neo4j Connected")
    except Exception as e:
        print(f"Neo4j Connection Failed: {e}")

    print("⏳ Loading Models (runs once)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    kw_model = KeyBERT(model=model)
    print("Models Loaded")
    
    yield
    
    if driver:
        driver.close()
        print("Neo4j Connection Closed")

# --- INITIALIZE APP ---
app = FastAPI(title="UIC Scholar Search Engine", lifespan=lifespan)

# ==========================================
# API 1: GLOBAL TOPIC SEARCH (Deduplicated)
# ==========================================
@app.post("/search", response_model=list[SearchResult])
def search_endpoint(request: SearchRequest):
    if not driver or not model: raise HTTPException(503, "Service not ready")

    try:
        # 1. Prepare Data
        query_vec = model.encode(request.query).tolist()
        keywords = kw_model.extract_keywords(request.query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=2)
        clean_query = " OR ".join([f'"{kw[0]}"' for kw in keywords]) if keywords else request.query
        
        print(f"🔎 [Topic] Query: '{request.query}' -> Filter: '{clean_query}'")

        # 2. Hybrid Query with Deduplication
        cypher_query = """
        CALL {
            // A. Topic Match
            CALL db.index.fulltext.queryNodes('topic_name_index', $clean_query, {limit: 50})
            YIELD node as topic
            MATCH (topic)<-[:HAS_TOPIC]-(p:Paper)-[:HAS_SECTION]->(s:Section)
            RETURN s as candidate, 2.0 as boost
            
            UNION
            
            // B. Keyword Match
            CALL db.index.fulltext.queryNodes('section_text_index', $clean_query, {limit: 50})
            YIELD node as s, score
            RETURN s as candidate, score as boost
            
            UNION

            // C. Vector Match
            CALL db.index.vector.queryNodes('section_embedding_index', 50, $embedding)
            YIELD node as s, score
            RETURN s as candidate, score as boost
        }

        // 3. Score Fusion
        WITH DISTINCT candidate, max(boost) as keyword_boost
        WITH candidate, keyword_boost, 
             vector.similarity.cosine(candidate.embedding, $embedding) as vector_score
        
        WITH candidate, keyword_boost, (vector_score + (keyword_boost * 0.5)) as final_score
        
        // 4. Traverse to Context
        MATCH (candidate)<-[:HAS_SECTION]-(p:Paper)<-[:AUTHORED]-(a:Author)

        // 5. DEDUPLICATION (Group by Paper)
        WITH a, p, candidate, final_score, keyword_boost
        ORDER BY final_score DESC
        
        // Keep only the best section per paper
        WITH a, p, head(collect({
            section: candidate.name,
            text: candidate.text,
            score: final_score,
            boost: keyword_boost
        })) as best_match
        
        // 6. Return Result
        RETURN 
            a.name AS Professor,
            p.title AS Paper,
            best_match.section AS SectionName,
            best_match.text AS Snippet,
            best_match.score AS Score,
            CASE 
                WHEN best_match.boost > 1.5 THEN 'Topic Match'
                WHEN best_match.boost > 0.5 THEN 'Keyword Match'
                ELSE 'Vector Match'
            END AS SourceType
        ORDER BY Score DESC
        LIMIT $limit
        """

        with driver.session() as session:
            result = session.run(cypher_query, embedding=query_vec, clean_query=clean_query, limit=request.limit)
            return [SearchResult(
                professor=r["Professor"], paper_title=r["Paper"], matched_section=r["SectionName"],
                snippet=r["Snippet"][:300] + "...", score=r["Score"], source_type=r["SourceType"]
            ) for r in result]

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))


# ==========================================
# API 2: AUTHOR FILTER SEARCH (New)
# ==========================================
@app.post("/search/author", response_model=list[SearchResult])
def search_author_endpoint(request: AuthorSearchRequest):
    """
    Finds papers by a SPECIFIC author that match the query.
    Strategy: Filter Author -> Get Papers -> Score Sections.
    """
    if not driver or not model: raise HTTPException(503, "Service not ready")

    try:
        # 1. Prepare Data
        query_vec = model.encode(request.query).tolist()
        keywords = kw_model.extract_keywords(request.query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=1)
        clean_keyword = keywords[0][0] if keywords else request.query
        
        print(f"🔎 [Author] Who: '{request.author_name}' | Query: '{clean_keyword}'")

        # 2. Author-First Query
        cypher_query = """
        // A. Match Author (Fastest Filter)
        MATCH (a:Author) 
        WHERE toLower(a.name) CONTAINS toLower($author_name)
        
        // B. Find their Papers & Sections
        MATCH (a)-[:AUTHORED]->(p:Paper)-[:HAS_SECTION]->(s:Section)
        
        // C. Calculate Vector Score (Semantic)
        WITH a, p, s, vector.similarity.cosine(s.embedding, $embedding) as vector_score
        
        // D. Calculate Keyword Boost (Manual check on text)
        WITH a, p, s, vector_score,
             CASE WHEN toLower(s.text) CONTAINS toLower($clean_keyword) THEN 1.0 ELSE 0.0 END as text_boost
        
        // E. Threshold & Fusion
        WITH a, p, s, (vector_score + text_boost) as final_score, text_boost
        WHERE final_score > 0.35
        
        // F. Deduplication (Best section per paper)
        ORDER BY final_score DESC
        WITH a, p, head(collect({
            section: s.name,
            text: s.text,
            score: final_score,
            source: CASE WHEN text_boost > 0 THEN 'Keyword + Vector' ELSE 'Vector Match' END
        })) as best_match
        
        // G. Return
        RETURN 
            a.name AS Professor,
            p.title AS Paper,
            best_match.section AS SectionName,
            best_match.text AS Snippet,
            best_match.score AS Score,
            best_match.source AS SourceType
        ORDER BY Score DESC
        LIMIT $limit
        """

        with driver.session() as session:
            result = session.run(
                cypher_query, 
                embedding=query_vec, 
                author_name=request.author_name,
                clean_keyword=clean_keyword, 
                limit=request.limit
            )
            return [SearchResult(
                professor=r["Professor"], paper_title=r["Paper"], matched_section=r["SectionName"],
                snippet=r["Snippet"][:300] + "...", score=r["Score"], source_type=r["SourceType"]
            ) for r in result]

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)