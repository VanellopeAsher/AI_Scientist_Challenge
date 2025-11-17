import os
import sys
import json
import asyncio
from typing import Optional
from datetime import datetime
from collections import namedtuple
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.integrated_workflow import IntegratedWorkflow, parse_user_prompt

app = FastAPI(title="Literature Review API")

# Token usage log file
TOKEN_LOG_FILE = "token_use.log"

def log_token_usage(endpoint: str, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
    """
    Log token usage to file
    """
    try:
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "endpoint": endpoint,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
        with open(TOKEN_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print(f"[token_log] Logged token usage: {total_tokens} tokens (prompt: {prompt_tokens}, completion: {completion_tokens})")
    except Exception as e:
        print(f"[token_log] Error logging token usage: {e}")

def get_config():
    """
    Create configuration from environment variables
    """
    Config = namedtuple("Config", [
        "n_candidates",
        "n_keywords",
        "gen_engine",
        "reranking_prompt_type",
        "temperature",
        "max_tokens",
        "seed",
        "rerank_method",
        "search_engine",
        "skip_extractive_check",
        "use_pdf",
        "use_full_text",
        "n_papers_for_generation",
        "skip_rerank",
        "search_sources",
    ])
    
    # Get configuration from environment variables with defaults
    n_candidates = int(os.getenv("SCI_N_CANDIDATES", "50"))
    n_keywords = max(int(os.getenv("SCI_N_KEYWORDS", "3")), 3)
    gen_engine = os.getenv("SCI_LLM_MODEL", "deepseek-chat")
    max_tokens = min(int(os.getenv("SCI_MAX_TOKENS", "8192")), 8192)
    temperature = float(os.getenv("SCI_TEMPERATURE", "0.2"))
    n_papers_for_generation = int(os.getenv("SCI_N_PAPERS_FOR_GENERATION", "40"))
    
    # Parse search sources
    search_sources_str = os.getenv("SCI_SEARCH_SOURCES", "arxiv,openalex")
    search_sources = [s.strip() for s in search_sources_str.split(",") if s.strip()]
    
    config = Config(
        n_candidates=n_candidates,
        n_keywords=n_keywords,
        gen_engine=gen_engine,
        reranking_prompt_type="basic_ranking",
        temperature=temperature,
        max_tokens=max_tokens,
        seed=42,
        rerank_method="llm",
        search_engine="arxiv",
        skip_extractive_check=False,
        use_pdf=False,
        use_full_text=False,
        n_papers_for_generation=n_papers_for_generation,
        skip_rerank=False,
        search_sources=search_sources,
    )
    
    return config

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )

async def stream_literature_review(query: str, config):
    """
    Generate literature review and stream the response
    """
    try:
        # Parse user prompt
        parsed_prompt = parse_user_prompt(query)
        core_topic = parsed_prompt.get('core_topic', query)
        
        # Create workflow
        workflow = IntegratedWorkflow(config)
        
        # Step 1: Search papers (no status messages to match README format)
        # Search papers
        if hasattr(config, 'search_sources') and config.search_sources:
            papers = workflow.search_multiple_sources(core_topic, config.search_sources)
        else:
            from src.single_query_retrieval import SingleQueryRetrieval
            retrieval = SingleQueryRetrieval(config)
            if config.skip_rerank:
                arxiv_query = retrieval.generate_arxiv_query(core_topic)
                papers = retrieval.search_arxiv(arxiv_query)
            else:
                papers = retrieval.retrieve(core_topic)
        
        if not papers:
            # Return error in standard format
            error_data = {
                "object": "error",
                "error": {
                    "message": "No papers found"
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        
        # Step 2: Select papers for generation
        n_papers = min(config.n_papers_for_generation, len(papers))
        selected_papers = papers[:n_papers]
        
        # Step 3: Generate plan and review
        plan, formatted_papers, plan_usage_info, plan_cost = workflow.generate_plan(
            core_topic, selected_papers, parsed_prompt
        )
        
        # For streaming, we need to modify the review generation to support streaming
        # For now, we'll generate the full review and then stream it
        # In a production system, you'd want to stream during generation
        
        review, references, _, review_usage_info, review_cost = workflow.generate_review(
            core_topic, selected_papers, plan, parsed_prompt
        )
        
        # Normalize citations
        review_normalized, citation_map, original_nums = workflow.normalize_citations(
            review, formatted_papers
        )
        review_normalized = workflow.clean_multiple_right_brackets(review_normalized)
        
        # Format references
        if not references:
            references = workflow.format_references_standard(
                citation_map, original_nums, formatted_papers
            )
        else:
            references = workflow.format_references_standard(
                citation_map, original_nums, formatted_papers
            )
        
        # Stream the review content
        full_content = review_normalized
        if references:
            full_content += "\n\n## References\n\n" + references
        
        # Stream content in chunks (matching README format exactly)
        # Use smaller chunks for better streaming experience
        chunk_size = 100  # characters per chunk
        for i in range(0, len(full_content), chunk_size):
            chunk = full_content[i:i+chunk_size]
            response_data = {
                "object": "chat.completion.chunk",
                "choices": [{
                    "delta": {
                        "content": chunk
                    }
                }]
            }
            yield f"data: {json.dumps(response_data)}\n\n"
            # Small delay to simulate streaming (optional, can be removed for faster response)
            await asyncio.sleep(0.005)
        
        # Log token usage
        total_prompt_tokens = 0
        total_completion_tokens = 0
        if plan_usage_info:
            total_prompt_tokens += plan_usage_info.get('prompt_tokens', 0)
            total_completion_tokens += plan_usage_info.get('completion_tokens', 0)
        if review_usage_info:
            total_prompt_tokens += review_usage_info.get('prompt_tokens', 0)
            total_completion_tokens += review_usage_info.get('completion_tokens', 0)
        
        if total_prompt_tokens > 0 or total_completion_tokens > 0:
            log_token_usage(
                endpoint="/literature_review",
                model=config.gen_engine,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens
            )
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_data = {
            "object": "error",
            "error": {
                "message": str(e)
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"

@app.post("/literature_review")
async def literature_review(request: Request):
    """
    Literature review endpoint - uses integrated workflow with retrieval and generation
    
    Request body:
    {
        "query": "What are the latest advances in transformer models?"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")

        if not query:
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": "Query is required"}
            )

        print(f"[literature_review] Received query: {query}")
        
        # Get configuration
        config = get_config()
        print(f"[literature_review] Using model: {config.gen_engine}")

        return StreamingResponse(
            stream_literature_review(query, config),
            media_type="text/event-stream"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)

