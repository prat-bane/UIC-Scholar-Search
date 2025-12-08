
from models.SearchRequest import SearchRequest
from models.SearchResult import SearchResult

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import os

# --- CONFIGURATION ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "test1234") # <--- Change this
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 

# --- GLOBAL VARIABLES ---
# We use globals so we load the heavy model only ONCE when the server starts
driver = None
model = None

# --- LIFESPAN MANAGER (Startup/Shutdown logic) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Load resources
    print("Starting Search API...")
    global driver, model
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    print("Neo4j Connected")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model Loaded")
    
    yield # API runs here
    
    # 2. Shutdown: Clean up
    if driver:
        driver.close()
        print("🛑 Neo4j Connection Closed")

# --- INITIALIZE APP ---
app = FastAPI(title="UIC Scholar Search Engine", lifespan=lifespan)
@app.post("/search", response_model=list[SearchResult])
def search_endpoint(request: SearchRequest):
    """
    Hybrid Search Endpoint:
    1. Vector Search (Semantic meaning)
    2. Full-Text Search (Exact keywords)
    3. Returns Professors & Papers
    """
    if not driver or not model:
        raise HTTPException(status_code=503, detail="Database or Model not ready")

    try:
        # 1. Generate Embedding for the User's Query
        query_embedding = model.encode(request.query).tolist()

        # 2. Run Hybrid Cypher Query
        # This combines Vector Search + Keyword Search using Reciprocal Rank Fusion
        cypher_query = """
        CALL {
            // A. Vector Search (Semantic)
            CALL db.index.vector.queryNodes('section_embedding_index', 20, $embedding)
            YIELD node, score
            RETURN node, score, "vector" as source
            
            UNION
            
            // B. Keyword Search (Exact Match)
            // We use the query text directly against the FullText index
            CALL db.index.fulltext.queryNodes('section_text_index', $query_text, {limit: 20})
            YIELD node, score
            RETURN node, score, "keyword" as source
        }

        // C. Fusion: Combine scores if a node appears in both
        WITH node, collect(source) as sources, max(score) as best_score
        
        // D. Traverse to find Context (Paper & Author)
        MATCH (node)<-[:HAS_SECTION]-(p:Paper)<-[:AUTHORED]-(a:Author)

        // E. Format Output
        RETURN 
            a.name AS Professor,
            p.title AS Paper,
            node.name AS SectionName,
            node.text AS Snippet,
            best_score AS Score,
            sources AS FoundBy
        ORDER BY Score DESC
        LIMIT $limit
        """

        with driver.session() as session:
            result = session.run(
                cypher_query, 
                embedding=query_embedding, 
                query_text=request.query,
                limit=request.limit
            )
            
            # 3. Format Response for API
            response_data = []
            for record in result:
                # Determine source type string
                sources = record["FoundBy"]
                source_type = "hybrid" if len(sources) > 1 else sources[0]

                response_data.append(SearchResult(
                    professor=record["Professor"],
                    paper_title=record["Paper"],
                    matched_section=record["SectionName"],
                    # Truncate snippet to 200 chars for clean display
                    snippet=record["Snippet"][:200] + "...", 
                    score=record["Score"],
                    source_type=source_type
                ))
            
            return response_data

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    # Run server: host 0.0.0.0 allows access from other machines
    uvicorn.run(app, host="0.0.0.0", port=8000)