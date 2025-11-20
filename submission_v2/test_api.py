"""
测试LitLLMs API端点
"""

import asyncio
import httpx
import json
import traceback
import time


async def test_literature_review():
    """测试文献综述生成端点"""
    print("=" * 80)
    print("测试 /literature_review 端点")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # 记录开始时间
            start_time = time.time()
            print("发送请求...")
            response = await client.post(
                "http://localhost:3000/literature_review",
                json={"query": "What are the latest advances in transformer models?"},
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print("\n流式响应内容:")
                print("-" * 80)
                try:
                    review_started = False
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # 移除 "data: " 前缀
                            if data == "[DONE]":
                                # 记录结束时间并计算耗时
                                end_time = time.time()
                                total_time = end_time - start_time
                                
                                # 格式化时间输出
                                hours = int(total_time // 3600)
                                minutes = int((total_time % 3600) // 60)
                                seconds = int(total_time % 60)
                                
                                print("\n[DONE]")
                                print("\n" + "=" * 80)
                                print("请求完成时间统计:")
                                print("=" * 80)
                                if hours > 0:
                                    print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒 ({total_time:.2f}秒)")
                                elif minutes > 0:
                                    print(f"总耗时: {minutes}分钟 {seconds}秒 ({total_time:.2f}秒)")
                                else:
                                    print(f"总耗时: {seconds}秒 ({total_time:.2f}秒)")
                                print("=" * 80)
                                break
                            try:
                                chunk = json.loads(data)
                                
                                # 处理进度信息
                                if chunk.get("object") == "progress":
                                    step = chunk.get("step", "")
                                    message = chunk.get("message", "")
                                    print(f"\n[进度 {step}] {message}")
                                    continue
                                
                                # 处理错误响应
                                if "error" in chunk:
                                    print(f"\n错误响应: {chunk}")
                                    break
                                
                                # 处理内容块
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        if not review_started:
                                            print("\n开始生成文献综述内容:\n")
                                            review_started = True
                                        print(content, end="", flush=True)
                            except json.JSONDecodeError as e:
                                print(f"\nJSON解析错误: {e}, 原始数据: {data}")
                        elif line.strip():
                            print(f"\n非数据行: {line}")
                except Exception as stream_error:
                    print(f"\n流式读取错误: {stream_error}")
                    print(f"错误类型: {type(stream_error).__name__}")
                    traceback.print_exc()
                print("\n" + "-" * 80)
            else:
                print(f"HTTP错误状态码: {response.status_code}")
                try:
                    error_text = response.text
                    print(f"错误响应内容: {error_text}")
                    if error_text:
                        try:
                            error_json = response.json()
                            print(f"错误JSON: {json.dumps(error_json, indent=2, ensure_ascii=False)}")
                        except:
                            pass
                except Exception as e:
                    print(f"读取错误响应时出错: {e}")
                
        except httpx.TimeoutException as e:
            print(f"请求超时: {e}")
            print(f"超时类型: {type(e).__name__}")
        except httpx.ConnectError as e:
            print(f"连接错误: {e}")
            print(f"错误类型: {type(e).__name__}")
            print("请确保API服务器正在运行 (python app.py 或 docker-compose up)")
        except httpx.HTTPStatusError as e:
            print(f"HTTP状态错误: {e}")
            print(f"状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        except Exception as e:
            print(f"未预期的错误: {e}")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误详情:")
            traceback.print_exc()


async def test_health():
    """测试健康检查端点"""
    print("\n" + "=" * 80)
    print("测试 /health 端点")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get("http://localhost:3000/health")
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.json()}")
            else:
                print(f"错误响应: {response.text}")
        except httpx.ConnectError as e:
            print(f"连接错误: {e}")
            print("请确保API服务器正在运行 (python app.py 或 docker-compose up)")
        except httpx.TimeoutException as e:
            print(f"请求超时: {e}")
        except Exception as e:
            print(f"错误: {e}")
            print(f"错误类型: {type(e).__name__}")
            traceback.print_exc()


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
