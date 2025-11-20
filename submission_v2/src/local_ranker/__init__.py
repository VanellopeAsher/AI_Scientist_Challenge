"""
本地重排序模块 - 基于4因子评分系统
实现sim(P,q), norm_impact(q), intent_weight(q), FICE_like(q)四个因子的评分
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_utils import run_llm_api


class LocalRanker:
    """基于4因子评分的本地重排序器"""
    
    def __init__(self, config):
        """
        初始化本地重排序器
        config: 包含配置参数的namedtuple，需要包含gen_engine, temperature, max_tokens等
        """
        self.config = config
        # 根据测试结果，最优系数为：α=0.45, β=0.60, γ=0.40, δ=0.10
        self.alpha = 0.45  # sim(P,q) - 语义匹配分
        self.beta = 0.60   # norm_impact(q) - 标准化影响分
        self.gamma = 0.40  # intent_weight(q) - 对提纲的贡献度
        self.delta = 0.10  # FICE_like(q) - 新颖性/信息量
        
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """从LLM响应中提取JSON"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取JSON块
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # 尝试提取```json块
            json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_block_match:
                try:
                    return json.loads(json_block_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 尝试提取```块
            code_block_match = re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
            if code_block_match:
                try:
                    return json.loads(code_block_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        return None
    
    def score_similarity(self, outline_paragraph: str, paper: Dict) -> Tuple[float, str]:
        """
        计算sim(P,q)：语义匹配分（0-1）
        
        Args:
            outline_paragraph: 研究提纲段落
            paper: 论文字典，包含title_paper和abstract字段
            
        Returns:
            (score, reason): 分数和理由
        """
        title = paper.get('title_paper', 'No title')
        abstract = paper.get('abstract', 'No abstract')
        
        prompt = f"""你是一名科学文献匹配系统。请严格按照要求输出 0 到 1 的相似度分数。

【任务】给定下面两段文本：
1. 研究提纲段落（用于写综述的目标段落）
2. 候选文献的标题和摘要

请你判断候选文献是否能直接支撑/回答该提纲段落的内容。

【要求输出】
只输出一个 JSON:
{{
  "similarity": <0到1的小数>,
  "reason": "一句话理由"
}}

【提纲 p】
{outline_paragraph}

【候选文献 q】
Title: {title}
Abstract: {abstract}"""
        
        system_prompt = "你是一名科学文献匹配系统。请严格按照要求输出JSON格式的相似度分数。"
        
        json_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
        }
        
        try:
            result = run_llm_api(
                json_data,
                gen_engine=self.config.gen_engine,
                max_tokens=500,
                temperature=self.config.temperature,
            )
            response = result["response"] if isinstance(result, dict) else result
            
            parsed = self._extract_json_from_response(response)
            if parsed and "similarity" in parsed:
                score = float(parsed["similarity"])
                # 确保分数在0-1范围内
                score = max(0.0, min(1.0, score))
                reason = parsed.get("reason", "无理由")
                return score, reason
            else:
                print(f"警告: 无法解析similarity响应: {response[:200]}")
                return 0.0, "解析失败"
        except Exception as e:
            print(f"计算similarity时出错: {e}")
            return 0.0, f"错误: {str(e)}"
    
    def score_norm_impact(self, paper: Dict) -> Tuple[float, str]:
        """
        计算norm_impact(q)：文章级标准化影响分（0-1）
        
        Args:
            paper: 论文字典，包含title_paper, abstract, publication_date, venue, citation_count等字段
            
        Returns:
            (score, reason): 分数和理由
        """
        title = paper.get('title_paper', 'No title')
        abstract = paper.get('abstract', 'No abstract')
        year = paper.get('publication_date', 'Unknown')
        if isinstance(year, str) and len(year) >= 4:
            year = year[:4]  # 提取年份
        venue = paper.get('venue', paper.get('journal', 'Unknown'))
        citations = paper.get('citation_count', 0)
        
        prompt = f"""你是一名科学文献评估系统。
请根据以下信息判断论文的学术影响力（normalized 0-1）：
- Title: {title}
- Abstract: {abstract}
- Publication Year: {year}
- Venue / Journal: {venue}
- Raw citation count: {citations}

要求：
1. 考虑论文发表年份和领域的平均引用水平
2. 输出一个浮点数 0-1，0 表示影响力极低，1 表示极高
3. 给出一句简短理由

输出 JSON 格式：
{{
  "norm_impact": <0-1>,
  "reason": "一句话解释"
}}"""
        
        system_prompt = "你是一名科学文献评估系统。请根据论文信息评估其学术影响力，输出JSON格式。"
        
        json_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
        }
        
        try:
            result = run_llm_api(
                json_data,
                gen_engine=self.config.gen_engine,
                max_tokens=500,
                temperature=self.config.temperature,
            )
            response = result["response"] if isinstance(result, dict) else result
            
            parsed = self._extract_json_from_response(response)
            if parsed and "norm_impact" in parsed:
                score = float(parsed["norm_impact"])
                # 确保分数在0-1范围内
                score = max(0.0, min(1.0, score))
                reason = parsed.get("reason", "无理由")
                return score, reason
            else:
                print(f"警告: 无法解析norm_impact响应: {response[:200]}")
                return 0.0, "解析失败"
        except Exception as e:
            print(f"计算norm_impact时出错: {e}")
            return 0.0, f"错误: {str(e)}"
    
    def score_intent_weight(self, outline_paragraph: str, paper: Dict) -> Tuple[float, str]:
        """
        计算intent_weight(q)：对提纲的贡献度（0-1）
        
        Args:
            outline_paragraph: 研究提纲段落
            paper: 论文字典，包含abstract字段
            
        Returns:
            (score, reason): 分数和理由
        """
        title = paper.get('title_paper', 'No title')
        abstract = paper.get('abstract', 'No abstract')
        q_summary = f"Title: {title}\nAbstract: {abstract}"
        
        prompt = f"""你是一名科研综述写作助手。
请根据下面的信息判断该文献对提纲段落的贡献程度：
- 提纲段落：{outline_paragraph}
- 文献摘要或引用片段：{q_summary}

要求：
1. 输出一个 0-1 的数值，表示文献对段落的支撑/贡献强度
2. 0 表示几乎无帮助，1 表示核心支撑
3. 给出一句简短理由
4. 输出 JSON 格式：
{{
  "intent_weight": <0-1>,
  "reason": "一句话说明贡献程度"
}}"""
        
        system_prompt = "你是一名科研综述写作助手。请评估文献对提纲段落的贡献程度，输出JSON格式。"
        
        json_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
        }
        
        try:
            result = run_llm_api(
                json_data,
                gen_engine=self.config.gen_engine,
                max_tokens=500,
                temperature=self.config.temperature,
            )
            response = result["response"] if isinstance(result, dict) else result
            
            parsed = self._extract_json_from_response(response)
            if parsed and "intent_weight" in parsed:
                score = float(parsed["intent_weight"])
                # 确保分数在0-1范围内
                score = max(0.0, min(1.0, score))
                reason = parsed.get("reason", "无理由")
                return score, reason
            else:
                print(f"警告: 无法解析intent_weight响应: {response[:200]}")
                return 0.0, "解析失败"
        except Exception as e:
            print(f"计算intent_weight时出错: {e}")
            return 0.0, f"错误: {str(e)}"
    
    def score_fice_like(self, paper: Dict) -> Tuple[float, str]:
        """
        计算FICE_like(q)：新颖性/信息量/主题覆盖（0-1）
        
        Args:
            paper: 论文字典，包含title_paper, abstract, publication_date字段
            
        Returns:
            (score, reason): 分数和理由
        """
        title = paper.get('title_paper', 'No title')
        abstract = paper.get('abstract', 'No abstract')
        year = paper.get('publication_date', 'Unknown')
        if isinstance(year, str) and len(year) >= 4:
            year = year[:4]  # 提取年份
        
        prompt = f"""你是一名科研文献分析专家。
请根据以下信息评估文献的新颖性和信息量：

- Title: {title}
- Abstract: {abstract}
- Publication Year: {year}

要求：
1. 给出文献的新颖性/信息量评分，范围 0-1
2. 0 表示内容平凡或已知，1 表示内容新颖且信息量大
3. 给出一句简短理由
4. 输出 JSON 格式：
{{
  "FICE_like": <0-1>,
  "reason": "一句话解释"
}}"""
        
        system_prompt = "你是一名科研文献分析专家。请评估文献的新颖性和信息量，输出JSON格式。"
        
        json_data = {
            "prompt": prompt,
            "system_prompt": system_prompt,
        }
        
        try:
            result = run_llm_api(
                json_data,
                gen_engine=self.config.gen_engine,
                max_tokens=500,
                temperature=self.config.temperature,
            )
            response = result["response"] if isinstance(result, dict) else result
            
            parsed = self._extract_json_from_response(response)
            if parsed and "FICE_like" in parsed:
                score = float(parsed["FICE_like"])
                # 确保分数在0-1范围内
                score = max(0.0, min(1.0, score))
                reason = parsed.get("reason", "无理由")
                return score, reason
            else:
                print(f"警告: 无法解析FICE_like响应: {response[:200]}")
                return 0.0, "解析失败"
        except Exception as e:
            print(f"计算FICE_like时出错: {e}")
            return 0.0, f"错误: {str(e)}"
    
    def score_paper(self, outline_paragraph: str, paper: Dict) -> Dict:
        """
        对单篇论文计算所有4个因子分数
        
        Args:
            outline_paragraph: 研究提纲段落
            paper: 论文字典
            
        Returns:
            包含所有分数和总分的字典
        """
        # 并行计算4个因子
        sim_score, sim_reason = self.score_similarity(outline_paragraph, paper)
        impact_score, impact_reason = self.score_norm_impact(paper)
        intent_score, intent_reason = self.score_intent_weight(outline_paragraph, paper)
        fice_score, fice_reason = self.score_fice_like(paper)
        
        # 计算总分：α*sim + β*impact + γ*intent + δ*fice
        total_score = (
            self.alpha * sim_score +
            self.beta * impact_score +
            self.gamma * intent_score +
            self.delta * fice_score
        )
        
        return {
            "paper": paper,
            "sim_score": sim_score,
            "sim_reason": sim_reason,
            "impact_score": impact_score,
            "impact_reason": impact_reason,
            "intent_score": intent_score,
            "intent_reason": intent_reason,
            "fice_score": fice_score,
            "fice_reason": fice_reason,
            "total_score": total_score,
        }
    
    def rerank_papers(self, outline_paragraph: str, candidate_papers: List[Dict], 
                     max_workers: int = 10, top_k: int = 40) -> List[Dict]:
        """
        对候选论文进行重排序
        
        Args:
            outline_paragraph: 研究提纲段落（如果为None，使用topic_description）
            candidate_papers: 候选论文列表
            max_workers: 并行处理的最大线程数
            top_k: 返回前k篇论文
            
        Returns:
            重排序后的论文列表（包含评分信息）
        """
        if not candidate_papers:
            print("没有候选论文需要重排序")
            return []
        
        print(f"开始对 {len(candidate_papers)} 篇论文进行4因子评分...")
        print(f"使用系数: α={self.alpha}, β={self.beta}, γ={self.gamma}, δ={self.delta}")
        
        scored_papers = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self.score_paper, outline_paragraph, paper): paper
                for paper in candidate_papers
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_paper), total=len(candidate_papers), desc="评分进度"):
                try:
                    result = future.result()
                    scored_papers.append(result)
                except Exception as e:
                    paper = future_to_paper[future]
                    print(f"评分论文 '{paper.get('title_paper', 'Unknown')}' 时出错: {e}")
                    # 添加默认分数
                    scored_papers.append({
                        "paper": paper,
                        "sim_score": 0.0,
                        "sim_reason": "评分失败",
                        "impact_score": 0.0,
                        "impact_reason": "评分失败",
                        "intent_score": 0.0,
                        "intent_reason": "评分失败",
                        "fice_score": 0.0,
                        "fice_reason": "评分失败",
                        "total_score": 0.0,
                    })
        
        # 按总分排序
        scored_papers.sort(key=lambda x: x["total_score"], reverse=True)
        
        # 返回top_k篇论文
        top_papers = scored_papers[:top_k]
        
        print(f"\n重排序完成，选择前 {len(top_papers)} 篇论文")
        print(f"最高分: {top_papers[0]['total_score']:.4f} (sim={top_papers[0]['sim_score']:.3f}, "
              f"impact={top_papers[0]['impact_score']:.3f}, intent={top_papers[0]['intent_score']:.3f}, "
              f"fice={top_papers[0]['fice_score']:.3f})")
        
        return top_papers
    
    def rerank_papers_simple(self, outline_paragraph: str, candidate_papers: List[Dict], 
                            top_k: int = 40) -> List[Dict]:
        """
        简化版重排序，返回原始论文对象（不包含评分信息），用于向后兼容
        
        Args:
            outline_paragraph: 研究提纲段落
            candidate_papers: 候选论文列表
            top_k: 返回前k篇论文
            
        Returns:
            重排序后的原始论文列表
        """
        scored_papers = self.rerank_papers(outline_paragraph, candidate_papers, top_k=top_k)
        # 只返回原始论文对象
        return [item["paper"] for item in scored_papers]

