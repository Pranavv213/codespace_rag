import os
import zipfile
import ast
from io import BytesIO
import time
import re
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec

# Core & Integrations
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langsmith import traceable
import uvicorn

# Load environment keys from .env file automatically
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Codebase RAG API", description="API for codebase indexing and querying")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    index_name: str
    k: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    query: str
    index_name: str

class UploadResponse(BaseModel):
    message: str
    index_name: str
    chunks_uploaded: int
    files_processed: int
    status: str

# Global dictionary to store vectorstores for different indices
vectorstores = {}

# 1. Initialize Pinecone
# ------------------------------------------------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
dimension = 1024  # Required dimension size for multilingual-e5-large

# 2. Setup Components
# ------------------------------------------------------------------
embeddings = PineconeEmbeddings(model="multilingual-e5-large")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)

@traceable
def sanitize_text(text: str) -> str:
    """
    Masks potential API keys, tokens, or common PII patterns.
    Add more patterns to the list as needed for your codebase.
    """
    patterns = [
        # Matches generic API keys (16-64 chars hex/alphanumeric)
        (r"(?i)(api[_-]?key|token|secret|password|passwd|pwd)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{16,64}['\"]?", r"\1: [MASKED]"),
        # Matches common email patterns
        (r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", r"\1 [MASKED]")
    ]
    
    sanitized = text
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized)
    
    return sanitized

# 3. Code Parser (Extracts Functions & Classes as Child Chunks)
# ------------------------------------------------------------------
@traceable
def parse_python_file_to_nodes(file_content: str, filename: str):
    """
    Parses a python file's source code using AST to extract 
    individual functions and classes as discrete child chunks.
    """
    child_chunks = []
    try:
        tree = ast.parse(file_content)
        lines = file_content.splitlines()
        
        for node in ast.walk(tree):
            # Target functions, async functions, and classes
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                # Find end line safely
                end_line = getattr(node, "end_lineno", start_line + 5)
                if end_line > len(lines):
                    end_line = len(lines)
                
                node_code = "\n".join(lines[start_line:end_line])
                node_type = "class" if isinstance(node, ast.ClassDef) else "function"
                
                child_chunks.append({
                    "text": f"[{node_type.upper()}] {node.name} in {filename}:\n{node_code}",
                    "name": node.name,
                    "type": node_type
                })
    except SyntaxError:
        # Fallback for non-python or malformed syntax: treat file as one chunk
        child_chunks.append({
            "text": file_content[:1000],
            "name": "raw_content",
            "type": "raw"
        })
    except Exception:
        child_chunks.append({
            "text": file_content[:1000],
            "name": "raw_content",
            "type": "raw"
        })
        
    return child_chunks

def get_or_create_index(index_name: str):
    """
    Get or create a Pinecone index with the given name.
    Returns the index object and a boolean indicating if it was newly created.
    """
    # Check if index exists
    index_exists = index_name in pc.list_indexes().names()
    
    if not index_exists:
        print(f"Creating Pinecone Index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index '{index_name}' created successfully.")
        return pc.Index(index_name), True
    else:
        print(f"Index '{index_name}' already exists.")
        return pc.Index(index_name), False

def get_vectorstore(index_name: str):
    """
    Get or create a vectorstore for the given index name.
    """
    if index_name not in vectorstores:
        index, _ = get_or_create_index(index_name)
        vectorstores[index_name] = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
    return vectorstores[index_name]

# 4. Zip File Processor & Ingestion Pipeline
# ------------------------------------------------------------------
@traceable
def ingest_codebase_zip(zip_path: str, index_name: str):
    """
    Extracts code files, sanitizes them, runs hierarchical parsing, 
    and uploads child chunks with sanitized parent metadata.
    """
    # Get or create the index
    index, is_new = get_or_create_index(index_name)
    
    if not is_new:
        return {
            "message": f"Index '{index_name}' already exists. Use a different index name for new uploads.",
            "chunks": 0,
            "files": 0,
            "status": "skipped"
        }
    
    # Get the vectorstore for this index
    vectorstore = get_vectorstore(index_name)
    
    child_docs_to_upload = []
    parent_counter = 0
    files_processed = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for file_info in archive.infolist():
                if file_info.is_dir() or file_info.filename.startswith('__MACOSX'):
                    continue
                    
                if file_info.filename.endswith(('.py', '.js', '.ts')):
                    with archive.open(file_info.filename) as file:
                        try:
                            raw_content = file.read().decode('utf-8')
                            
                            # --- SECURITY STEP: Sanitize immediately upon extraction ---
                            safe_content = sanitize_text(raw_content)
                            
                        except UnicodeDecodeError:
                            continue 
                        
                        filename = file_info.filename
                        # Use the sanitized content for the parent text
                        parent_text = f"--- FILE: {filename} ---\n{safe_content}"
                        parent_id = f"file_parent_{parent_counter}"
                        parent_counter += 1
                        files_processed += 1
                        
                        # Parse nodes using the sanitized content
                        extracted_nodes = parse_python_file_to_nodes(safe_content, filename)
                        
                        for node in extracted_nodes:
                            child_metadata = {
                                "source_file": filename,
                                "entity_name": node["name"],
                                "entity_type": node["type"],
                                "parent_id": parent_id,
                                "parent_text": parent_text
                            }
                            
                            child_docs_to_upload.append(
                                Document(page_content=node["text"], metadata=child_metadata)
                            )

        if child_docs_to_upload:
            print(f"Uploading {len(child_docs_to_upload)} sanitized code chunks to Pinecone index '{index_name}'...")
            try:
                vectorstore.add_documents(child_docs_to_upload)
                print("Upload completed successfully!")
                return {
                    "message": f"Upload completed successfully to index '{index_name}'!",
                    "chunks": len(child_docs_to_upload),
                    "files": files_processed,
                    "status": "success"
                }
            except Exception as e:
                print(f"Upload error: {e}")
                return {
                    "message": f"Upload error: {e}",
                    "chunks": 0,
                    "files": 0,
                    "status": "error"
                }
        else:
            print("No compatible code files found in the zip archive.")
            return {
                "message": "No compatible code files found in the zip archive.",
                "chunks": 0,
                "files": 0,
                "status": "no_files"
            }
            
    except FileNotFoundError:
        error_msg = f"Error: Zip file '{zip_path}' not found."
        print(error_msg)
        return {
            "message": error_msg,
            "chunks": 0,
            "files": 0,
            "status": "error"
        }
    except zipfile.BadZipFile:
        error_msg = f"Error: '{zip_path}' is not a valid zip file."
        print(error_msg)
        return {
            "message": error_msg,
            "chunks": 0,
            "files": 0,
            "status": "error"
        }
    except Exception as e:
        error_msg = f"Error processing zip file: {e}"
        print(error_msg)
        return {
            "message": error_msg,
            "chunks": 0,
            "files": 0,
            "status": "error"
        }

# 5. Metadata Parent Retriever (Your Custom Logical Retriever)
# ------------------------------------------------------------------
@traceable
def metadata_parent_retriever(query: str, index_name: str, k: int = 5):
    """
    Queries Pinecone for child matches, extracts their parent texts,
    and deduplicates them to present clean context to Gemini.
    """
    try:
        # Get the vectorstore for this index
        vectorstore = get_vectorstore(index_name)
        
        # 1. Search Pinecone for the top child vectors
        results = vectorstore.similarity_search(query, k=k)
        
        seen_parent_ids = set()
        parent_contexts = []
        
        # 2. Loop through child matches and grab the parent text
        for doc in results:
            parent_id = doc.metadata.get("parent_id")
            parent_text = doc.metadata.get("parent_text")
            
            # Deduplicate so we don't repeat the same parent block
            if parent_id not in seen_parent_ids and parent_text:
                seen_parent_ids.add(parent_id)
                parent_contexts.append(parent_text)
                
        return "\n\n".join([sanitize_text(ctx) for ctx in parent_contexts])
    except Exception as e:
        print(f"Retriever error: {e}")
        return ""

# 6. Construct RAG Chain
# ------------------------------------------------------------------
prompt_template = """You are an expert developer assistant. Answer the user's question about the codebase using ONLY the provided code files.

Context (Relevant Source Code Files):
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def get_rag_chain(index_name: str):
    """
    Creates a RAG chain for a specific index.
    """
    return (
        {
            "context": lambda q: metadata_parent_retriever(q, index_name, k=5), 
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# 7. FastAPI Endpoints
# ------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "Codebase RAG API is running", "status": "active"}

@app.post("/upload", response_model=UploadResponse)
async def upload_codebase(
    file: UploadFile = File(...),
    index_name: str = Form(...)
):
    """
    Upload a zip file containing codebase to be indexed.
    The file will be extracted, chunked, and embeddings will be stored in Pinecone.
    A new index will be created with the provided index_name.
    """
    # Check if file is a zip
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")
    
    # Validate index name
    if not index_name or not index_name.strip():
        raise HTTPException(status_code=400, detail="Index name is required")
    
    # Sanitize index name (replace spaces and special chars)
    index_name = re.sub(r'[^a-zA-Z0-9-]', '-', index_name.strip())
    
    # Save the uploaded zip file temporarily
    temp_zip_path = f"temp_{file.filename}"
    try:
        # Read and save the file
        content = await file.read()
        with open(temp_zip_path, 'wb') as f:
            f.write(content)
        
        # Process the zip file with the provided index name
        result = ingest_codebase_zip(temp_zip_path, index_name)
        
        # Clean up temp file
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        
        return UploadResponse(
            message=result["message"],
            index_name=index_name,
            chunks_uploaded=result["chunks"],
            files_processed=result["files"],
            status=result["status"]
        )
        
    except Exception as e:
        # Clean up temp file if exists
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a query about the codebase and get an answer based on the indexed content.
    """
    try:
        # Validate index name
        if not request.index_name or not request.index_name.strip():
            raise HTTPException(status_code=400, detail="Index name is required")
        
        # Sanitize index name
        index_name = re.sub(r'[^a-zA-Z0-9-]', '-', request.index_name.strip())
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            raise HTTPException(
                status_code=404, 
                detail=f"Index '{index_name}' not found. Please upload a codebase with this index name first."
            )
        
        # Get the RAG chain for this index
        rag_chain = get_rag_chain(index_name)
        
        # Update the k value in the retriever if needed
        if request.k != 5:
            # Create a new chain with custom k
            custom_chain = (
                {
                    "context": lambda q: metadata_parent_retriever(q, index_name, k=request.k), 
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            response = custom_chain.invoke(request.query)
        else:
            response = rag_chain.invoke(request.query)
        
        return ChatResponse(
            answer=response,
            query=request.query,
            index_name=index_name
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/indices")
async def list_indices():
    """List all available indices in Pinecone"""
    try:
        indices = pc.list_indexes().names()
        return {
            "indices": list(indices),
            "count": len(indices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing indices: {str(e)}")

@app.delete("/index/{index_name}")
async def delete_index(index_name: str):
    """
    Delete a specific index from Pinecone.
    """
    try:
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            # Remove from vectorstores cache if present
            if index_name in vectorstores:
                del vectorstores[index_name]
            return {"message": f"Index '{index_name}' deleted successfully", "status": "success"}
        else:
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting index: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "available_indices": list(pc.list_indexes().names())
    }

# 8. Run the FastAPI application
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)