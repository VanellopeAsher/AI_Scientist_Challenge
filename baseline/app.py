import os
import json
import base64
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(title="Science Arena Challenge Example Submission")

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

# Initialize AsyncOpenAI client for LLM models
client = AsyncOpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for non-streaming endpoints
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc)
        }
    )


@app.post("/literature_review")
async def literature_review(request: Request):
    """
    Literature review endpoint - uses standard LLM model

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
        print(f"[literature_review] Using model: {os.getenv('SCI_LLM_MODEL')}")

        async def generate():
            # Prepare prompt for literature review
            prompt = f"""Conduct a literature review on the following topic:{query}"""
            model_name = os.getenv("SCI_LLM_MODEL")

            # Call LLM model with streaming
            stream = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.2,
                stream=True
            )

            # Variables to track token usage and accumulated content
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            accumulated_content = ""

            # Stream back results
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        accumulated_content += delta_content
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": delta_content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"
                
                # Check for usage information in chunk (usually in the last chunk)
                if hasattr(chunk, 'usage') and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0
                    total_tokens = chunk.usage.total_tokens or 0

            # If usage not found in chunks, estimate tokens using simple approximation
            # (rough estimate: 1 token ≈ 4 characters for English, 1.5 for Chinese)
            if total_tokens == 0:
                # Estimate prompt tokens (rough approximation)
                prompt_tokens = len(prompt) // 3  # Rough estimate
                # Estimate completion tokens
                completion_tokens = len(accumulated_content) // 3  # Rough estimate
                total_tokens = prompt_tokens + completion_tokens
                print(f"[token_usage] Usage not available from API, using estimation: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

            # Log token usage
            log_token_usage(
                endpoint="/literature_review",
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

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
