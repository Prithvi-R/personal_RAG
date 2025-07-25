import os
from dotenv import load_dotenv
import glob
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import shutil

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
    properties: Dict[str, Any] = {}
class Relationship(BaseModel):
    source: Node
    target: Node
    type: str
    properties: Dict[str, Any] = {}

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

# ==============================================================================
#  3. THE HIGH-DETAIL GRAPH EXTRACTION PROMPT
# ==============================================================================
GRAPH_EXTRACTION_PROMPT = """
You are an expert in extracting important structured knowledge graphs from diverse unstructured documents with relevance entites and their relation.

Your task is to:
1. Identify important **entities** in the text.
2. Extract relationships between those entities.
3. Organize the output in a **JSON format** that matches the KnowledgeGraph schema.

tool_params = {
    "nodes": nodes_list,
    "relationships": relationships_list,
}

**Schema:**
- Each **Entity** must have:
  - `id`: the name of the entity.
  - `type`: e.g. Person, Course, Institution, Degree, Subject, Role, Organization, Technology, etc.
  - `properties`: type dict ,for any useful attributes (e.g., grade, roll number, e-mail, contact, cpi, spi enrollment year, etc.)

- Each **Relationship** must have:
  - `source`: an entity object.
  - `target`: another entity object.
  - `type`: the type of relationship (e.g., ENROLLED_IN, HAS_COURSE, GRADES_IN, WORKS_AT, FOUNDED, STUDIED, etc.)
  - `properties`: type dict ,for relative info. 

**Instructions:**
1.  Carefully read the text and identify all entities and relationships that match the schema.
2.  Construct a valid JSON object adhering to the `KnowledgeGraph` Pydantic schema.
3.  Do not extract generic entities. Be specific (e.g., "Dr. Aris Thorne" is a good entity, "a founder" is not).
4.  If an entity or relationship type is not in the schema, do not extract it.
5. Extract only meaningful entities (e.g., "Prithvi Raj", "Mathematics I", "IIIT Guwahati").
6. Be specific. Avoid vague entities like "a student" or "the course".
7. Use only relationships relevant to the document.
8. Include `properties` if useful details are present (e.g., Grade, Credits, etc.)
9. If the document doesn't follow a known format, still extract what fits logically into the schema.

**High-Quality Example:**
Text: "Photonics AI, founded by Dr. Aris Thorne, specializes in Optical Computing. Dr. Thorne previously worked at Google. Their main competitor is QuantumLeap."
Output (as a JSON object):
{{
  "nodes": [
    {{"id": "Photonics AI", "type": "Company", "properties": {{}}}},
    {{"id": "Dr. Aris Thorne", "type": "Person", "properties": {{}}}},
    {{"id": "Optical Computing", "type": "Technology", "properties": {{}}}},
    {{"id": "Google", "type": "Company", "properties": {{}}}},
    {{"id": "QuantumLeap", "type": "Company", "properties": {{}}}}
  ],
  "relationships": [
    {{
      "type": "FOUNDED_BY",
      "source": {{"id": "Photonics AI", "type": "Company", "properties": {{}}}},
      "target": {{"id": "Dr. Aris Thorne", "type": "Person", "properties": {{}}}}
    }},
    {{
      "type": "USES",
      "source": {{"id": "Photonics AI", "type": "Company", "properties": {{}}}},
      "target": {{"id": "Optical Computing", "type": "Technology", "properties": {{}}}}
    }},
    {{
      "type": "WORKS_AT",
      "source": {{"id": "Dr. Aris Thorne", "type": "Person", "properties": {{}}}},
      "target": {{"id": "Google", "type": "Company", "properties": {{}}}}
    }},
    {{
      "type": "COMPETES_WITH",
      "source": {{"id": "Photonics AI", "type": "Company", "properties": {{}}}},
      "target": {{"id": "QuantumLeap", "type": "Company", "properties": {{}}}}
    }}
  ]
}}
**Another Example:**
Text:  
"Raj (Roll No: 2310321) is pursuing B.Tech in Electronics and Communication Engineering(ECE) at XYZ College. email:xyz@gmail.com male 923943053"

Output (as a JSON object):
{{
  "nodes": [
    {{"id": "raj", "type": "Person", "properties": {{"gender": "Male", "phone": "923943053", "roll_number": "2310321", "program": "B.Tech", "branch": "ECE", "email": "rprithvi939@gmail.com"}}}},
    {{"id": "IIIT Guwahati", "type": "Institution"}}
  ],
  "relationships": [
    {{"source": {{"id": "raj", "type": "Person"}}, "target": {{"id": "XYZ College", "type": "Institution"}}, "type": "STUDIES_AT"}}
    {{"source": {{"id": "raj", "type": "Person"}}, "target": {{"id": "ECE", "type": "Course"}}, "type": "STUDIES"}}
  ]
}}
"""
escaped_example = GRAPH_EXTRACTION_PROMPT.replace("{", "{{").replace("}", "}}")
# Create the structured output parser with the KnowledgeGraph model
extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", escaped_example),
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

    if os.path.isdir(PERSIST_DIRECTORY):
        print(f"Clearing previous vector store memory at '{PERSIST_DIRECTORY}'...")
        shutil.rmtree(PERSIST_DIRECTORY)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
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
    graph.query("MATCH (n) DETACH DELETE n")
    
    
    llm_for_extraction = ChatGroq(model="llama3-70b-8192", temperature=0).with_structured_output(KnowledgeGraph)
    llm_chain = extraction_prompt | llm_for_extraction
    
    total_nodes, total_rels = 0, 0
    for i, chunk in enumerate(chunks):
        if not chunk.page_content.strip(): continue
        print(f"--> Processing chunk {i+1}/{len(chunks)} from source: {chunk.metadata.get('source', 'Unknown')}...")
        try:
            kg_object = llm_chain.invoke({"text_chunk": chunk.page_content})
            
            source_filename = os.path.basename(chunk.metadata.get('source', 'Unknown'))
            for node in kg_object.nodes:
                # Normalize id and type to lowercase
                node.id = node.id.lower()
                node.type = node.type.lower()

                # Normalize property keys and string values to lowercase
                normalized_props = {}
                for key, value in node.properties.items():
                    new_key = key.lower()
                    new_value = value.lower() if isinstance(value, str) else value
                    normalized_props[new_key] = new_value

                # Add source document
                normalized_props["source_document"] = source_filename.lower()
                node.properties = normalized_props

            # Normalize relationships as well
            for rel in kg_object.relationships:
                rel.type = rel.type.lower()
                rel.source.id = rel.source.id.lower()
                rel.source.type = rel.source.type.lower()
                rel.target.id = rel.target.id.lower()
                rel.target.type = rel.target.type.lower()
                if rel.properties:
                    rel.properties = {
                        k.lower(): v.lower() if isinstance(v, str) else v
                        for k, v in rel.properties.items()
                    }
            graph.add_graph_documents([kg_object])
            
            total_nodes += len(kg_object.nodes)
            total_rels += len(kg_object.relationships)
        except Exception as e:
            print(f"    ERROR processing chunk for graph: {e}")

    print(f"\n-> Upserted a total of {total_nodes} nodes and {total_rels} relationships to the graph.")
    print("-> Graph Store Ingestion Complete.")

if __name__ == "__main__":
    embedding_documents()