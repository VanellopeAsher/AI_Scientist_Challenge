"""
测试LitLLMs API端点
"""

import asyncio
import httpx
import json


async def test_literature_review():
    """测试文献综述生成端点"""
    print("=" * 80)
    print("测试 /literature_review 端点")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                "http://localhost:3000/literature_review",
                json={"query": "What are the latest advances in transformer models?"},
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("\n流式响应内容:")
                print("-" * 80)
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # 移除 "data: " 前缀
                        if data == "[DONE]":
                            print("\n[DONE]")
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                content = chunk["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            pass
                print("\n" + "-" * 80)
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")


async def test_health():
    """测试健康检查端点"""
    print("\n" + "=" * 80)
    print("测试 /health 端点")
    print("=" * 80)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:3000/health")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")


async def main():
    """主测试函数"""
    print("开始测试LitLLMs API...")
    
    # 先测试健康检查
    await test_health()
    
    # 测试文献综述生成（可能需要较长时间）
    print("\n注意: 文献综述生成可能需要几分钟时间...")
    await test_literature_review()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

