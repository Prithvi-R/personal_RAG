import os
from dotenv import load_dotenv
import glob
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# For Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredImageLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# For Vector Store
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# For Graph Store
from langchain_neo4j import Neo4jGraph

# For LLM and Prompt
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 

load_dotenv()
print("Environment variables loaded.")

# ==============================================================================
#  1. SETTING UP CONFIGURATION
# ==============================================================================
VECTOR_COLLECTION_NAME = "ai_documents"
DOCS_DIRECTORY = "docs"
PERSIST_DIRECTORY = "db_chroma"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ==============================================================================
#  2. PYDANTIC MODELS
# ==============================================================================
class Node(BaseModel):
    id: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class Relationship(BaseModel):
    source: Node
    target: Node
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

# ==============================================================================
#  3. THE HIGH-DETAIL GRAPH EXTRACTION PROMPT
# ==============================================================================
GRAPH_EXTRACTION_PROMPT = """
You are an expert in knowledge graph generation from text. Your task is to identify entities and their relationships based on a strict schema.

**Schema:**
- **Entities**: Must have an `id` (the entity's name) and a `type`. Permitted types are: `Company`, `Person`, `Technology`, `Product`, `Feature`, `VentureCapital`, `Concept`.
- **Relationships**: Must have a `source` (entity id), a `target` (entity id), and a `type`. Permitted types are: `FOUNDED_BY`, `WORKS_AT`, `INVESTED_IN`, `COMPETES_WITH`, `USES`, `HAS_PRODUCT`, `HAS_FEATURE`, `BLOCKED_BY`, `PIONEERED`.

**Instructions:**
1.  Carefully read the text and identify all entities and relationships that match the schema.
2.  Construct a valid JSON object adhering to the `KnowledgeGraph` Pydantic schema.
3.  Do not extract generic entities. Be specific (e.g., "Dr. Aris Thorne" is a good entity, "a founder" is not).
4.  If an entity or relationship type is not in the schema, do not extract it.

**High-Quality Example:**
Text: "Photonics AI, founded by Dr. Aris Thorne, specializes in Optical Computing. Dr. Thorne previously worked at Google. Their main competitor is QuantumLeap."
Output (as a JSON object):
{{
  "nodes": [
    {{"id": "Photonics AI", "type": "Company"}},
    {{"id": "Dr. Aris Thorne", "type": "Person"}},
    {{"id": "Optical Computing", "type": "Technology"}},
    {{"id": "Google", "type": "Company"}},
    {{"id": "QuantumLeap", "type": "Company"}}
  ],
  "relationships": [
     {{"source": {{"id": "Photonics AI", "type": "Company"}}, "target": {{"id": "Dr. Aris Thorne", "type": "Person"}}, "type": "FOUNDED_BY"}},
    {{"source": {{"id": "Photonics AI", "type": "Company"}}, "target": {{"id": "Optical Computing", "type": "Technology"}}, "type": "USES"}},
    {{"source": {{"id": "Dr. Aris Thorne", "type": "Person"}}, "target": {{"id": "Google", "type": "Company"}}, "type": "WORKS_AT"}},
    {{"source": {{"id": "Photonics AI", "type": "Company"}}, "target": {{"id": "QuantumLeap", "type": "Company"}}, "type": "COMPETES_WITH"}}
  ]
}}
"""
# Create the structured output parser with the KnowledgeGraph model
extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRAPH_EXTRACTION_PROMPT),
        ("human", "Now, based on the schema and example, process the following text. Do not add any explanation.\n\nText: {text_chunk}"),
    ]
)

# ==============================================================================
#  4. DOCUMENT LOADING & PROCESSING LOGIC
# ==============================================================================
def get_loader_for_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf": return PyPDFLoader(file_path)
    elif ext == ".md": return UnstructuredMarkdownLoader(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]: return UnstructuredImageLoader(file_path, mode="single")
    elif ext == ".txt": return TextLoader(file_path)
    return None

def embedding_documents():
    print("\n--- STARTING DOCUMENT INGESTION ---")

    print(f"Scanning for documents in '{DOCS_DIRECTORY}'...")
    supported_files = [f for f in glob.glob(os.path.join(DOCS_DIRECTORY, "**/*.*"), recursive=True) if os.path.splitext(f)[1].lower() in [".pdf", ".md", ".png", ".jpg", ".jpeg", ".txt"]]
    if not supported_files: print(f"No supported documents found."); return

    documents = []
    for file_path in supported_files:
        loader = get_loader_for_file(file_path)
        if loader:
            try:
                print(f"  - Loading: {os.path.basename(file_path)}")
                documents.extend(loader.load())
            except Exception as e: print(f"    ERROR loading file {file_path}: {e}")

    if not documents: print("Could not load any documents successfully. Exiting."); return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"-> Loaded and split {len(documents)} document(s) into {len(chunks)} chunks.")

    print(f"\n-> Upserting documents into Vector Store {PERSIST_DIRECTORY}...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save the FAISS index to disk
    vector_store.save_local(PERSIST_DIRECTORY)    
    print("---> Vector Store Ingestion Completed.")

    print("\n-> Upserting data into Graph Store (Neo4j)...")
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    # For a full reset of prev graph, run this manually in the Neo4j Browser---------> MATCH (n) DETACH DELETE n
    #or uncomment the line below to reset everytime
    # graph.query("MATCH (n) DETACH DELETE n")
    
    
    llm_for_extraction = ChatGroq(model="llama3-70b-8192", temperature=0).with_structured_output(KnowledgeGraph)
    llm_chain = extraction_prompt | llm_for_extraction
    
    total_nodes, total_rels = 0, 0
    for i, chunk in enumerate(chunks):
        if not chunk.page_content.strip(): continue
        print(f"--> Processing chunk {i+1}/{len(chunks)} from source: {chunk.metadata.get('source', 'Unknown')}...")
        try:
            # We invoke the full chain, which includes the detailed prompt.
            kg_object = llm_chain.invoke({"text_chunk": chunk.page_content})
            
            source_filename = os.path.basename(chunk.metadata.get('source', 'Unknown'))
            for node in kg_object.nodes:
                node.properties["source_document"] = source_filename
            
            graph.add_graph_documents([kg_object])
            
            total_nodes += len(kg_object.nodes)
            total_rels += len(kg_object.relationships)
        except Exception as e:
            print(f"    ERROR processing chunk for graph: {e}")

    print(f"\n-> Upserted a total of {total_nodes} nodes and {total_rels} relationships to the graph.")
    print("-> Graph Store Ingestion Complete.")

if __name__ == "__main__":
    embedding_documents()