"""
LitLLMs FastAPI Application
提供文献综述生成服务的后端API
"""

import os
import sys
import json
import asyncio
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
from collections import namedtuple

# 加载环境变量
load_dotenv()

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)

from integrated_workflow import IntegratedWorkflow, parse_user_prompt

app = FastAPI(title="LitLLMs Literature Review API")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Token使用日志文件
TOKEN_LOG_FILE = "token_use.log"


def log_token_usage(endpoint: str, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
    """记录token使用情况到文件"""
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


def create_config():
    """创建配置对象"""
    Config = namedtuple("Config", [
        "n_candidates",
        "n_papers_for_generation",
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
        "skip_rerank",
        "search_sources",
    ])
    
    # 从环境变量读取配置，如果没有则使用默认值
    gen_engine = os.getenv("SCI_LLM_MODEL", "deepseek-chat")
    
    config = Config(
        n_candidates=150,  # 初始检索150篇
        n_papers_for_generation=40,  # 使用40篇生成综述
        n_keywords=3,
        gen_engine=gen_engine,
        reranking_prompt_type="basic_ranking",
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
        seed=42,
        rerank_method="llm",
        search_engine="arxiv",
        skip_extractive_check=False,
        use_pdf=False,
        use_full_text=False,
        skip_rerank=False,  # 使用4因子评分系统
        search_sources=None,  # 可以设置为['arxiv', 'openalex']等
    )
    
    return config


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    import traceback
    traceback.print_exc()
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
    文献综述生成端点
    
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
        
        # 创建配置
        config = create_config()
        print(f"[literature_review] Using model: {config.gen_engine}")

        async def generate():
            """生成文献综述的异步生成器"""
            try:
                # 解析用户prompt
                parsed_prompt = parse_user_prompt(query)
                core_topic = parsed_prompt.get('core_topic', query)
                
                # 创建工作流
                workflow = IntegratedWorkflow(config)
                
                # 运行工作流（在后台线程中执行，因为它是同步的）
                loop = asyncio.get_event_loop()
                
                # 在后台线程中运行同步工作流
                def run_workflow():
                    return workflow.run(core_topic, parsed_prompt)
                
                # 运行工作流
                result = await loop.run_in_executor(None, run_workflow)
                
                if not result or result[0] is None:
                    error_data = {
                        "object": "error",
                        "error": {
                            "message": "Failed to generate literature review"
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                # result是一个tuple: (result_dict, plan, review, references)
                if isinstance(result, tuple) and len(result) >= 3:
                    result_dict, plan, review = result[0], result[1], result[2]
                else:
                    # 如果返回格式不同，尝试提取review
                    review = result.get('review', '') if isinstance(result, dict) else str(result)
                
                if not review:
                    error_data = {
                        "object": "error",
                        "error": {
                            "message": "No review content generated"
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
                
                # 流式输出review内容
                # 将review分成小块发送
                chunk_size = 50  # 每次发送50个字符
                review_text = review if isinstance(review, str) else str(review)
                
                for i in range(0, len(review_text), chunk_size):
                    chunk = review_text[i:i + chunk_size]
                    response_data = {
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "delta": {
                                "content": chunk
                            }
                        }]
                    }
                    yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"
                    # 添加小延迟以模拟流式输出
                    await asyncio.sleep(0.01)
                
                # 发送完成标记
                yield "data: [DONE]\n\n"
                
                # 记录token使用（如果有的话）
                # 这里可以添加token使用统计
                log_token_usage(
                    endpoint="/literature_review",
                    model=config.gen_engine,
                    prompt_tokens=0,  # 可以从workflow中获取
                    completion_tokens=0,  # 可以从workflow中获取
                    total_tokens=0
                )
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_data = {
                    "object": "error",
                    "error": {
                        "message": str(e)
                    }
                }
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(e)}
        )


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "ok", "service": "LitLLMs Literature Review API"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)

