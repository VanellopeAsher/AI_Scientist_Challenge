"""
单次主题检索脚本
输入：一段话描述的主题
输出：在arxiv上检索并经过LLM重排序的论文列表
"""

import os
import re
import json
import argparse
import sys
import arxiv
import pandas as pd
from collections import namedtuple
from tqdm import tqdm

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.paper_manager_utils import rerank_candidates_batch, get_keywords_for_s2_search


class SingleQueryRetrieval:
    """单次主题检索类"""
    
    def __init__(self, config):
        """
        初始化检索器
        config: 包含配置参数的namedtuple
        """
        self.config = config
        
    def generate_arxiv_query(self, topic_description):
        """
        使用LLM将主题描述转换为arxiv搜索查询
        """
        print(f"正在为主题生成搜索查询: {topic_description}")
        
        # 使用LLM生成关键词查询
        query_data = get_keywords_for_s2_search(topic_description, self.config)
        
        if not query_data or not query_data.get("queries"):
            # 如果LLM生成失败，直接使用原始描述作为查询
            print("LLM生成查询失败，使用原始描述作为查询")
            return topic_description
        
        # 使用第一个查询关键词，或组合多个关键词
        queries = query_data.get("queries", [])
        if queries:
            # 组合所有查询关键词
            arxiv_query = " OR ".join(queries)
            print(f"生成的arxiv查询: {arxiv_query}")
            return arxiv_query
        else:
            return topic_description
    
    def search_arxiv(self, query, max_results=None):
        """
        在arxiv上搜索论文
        """
        if max_results is None:
            max_results = self.config.n_candidates
        
        print(f"正在arxiv上搜索论文，查询: {query}, 最多返回: {max_results}")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,  # 多获取一些，后续会重排序
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        
        papers = []
        for result in tqdm(search.results(), desc="获取论文"):
            entry_id = result.entry_id
            arxiv_id = entry_id.split("/")[-1].replace("v1", "").replace("v2", "").replace("v3", "")
            
            # 提取作者信息
            authors = []
            if result.authors:
                for author in result.authors:
                    author_name = author.name
                    if author_name:
                        authors.append(author_name)
            
            paper = {
                "paper_id": arxiv_id,
                "title_paper": result.title,
                "abstract": result.summary,
                "citation_count": 0,  # arxiv API不提供引用数
                "publication_date": result.published.strftime("%Y-%m-%d") if result.published else "Unknown",
                "external_ids": {"arxiv": arxiv_id},
                "authors": authors,  # 添加作者信息
            }
            papers.append(paper)
        
        print(f"从arxiv获取了 {len(papers)} 篇论文")
        return papers[:self.config.n_candidates]
    
    def rerank_with_llm(self, topic_description, candidate_papers):
        """
        使用LLM对候选论文进行重排序
        """
        if not candidate_papers:
            print("没有候选论文需要重排序")
            return []
        
        print(f"正在使用LLM对 {len(candidate_papers)} 篇论文进行重排序...")
        
        # 使用现有的重排序函数
        ranking, reason, arxiv_ids = rerank_candidates_batch(
            topic_description,  # 使用主题描述作为查询摘要
            candidate_papers,
            self.config
        )
        
        # 解析排序结果
        extracted_order = [
            int(s) for s in re.findall(r"\d+", ranking)
            if 1 <= int(s) <= len(candidate_papers)
        ]
        
        print(f"提取的排序顺序: {extracted_order[:10]}...")  # 只显示前10个
        
        # 根据排序结果重新排列论文
        seen = set()
        reranked_papers = []
        
        # 首先按照LLM给出的顺序添加
        for idx in extracted_order:
            if idx <= len(candidate_papers) and idx not in seen:
                reranked_papers.append(candidate_papers[idx - 1])
                seen.add(idx)
        
        # 如果还有论文没有被包含，按原始顺序添加
        for i, paper in enumerate(candidate_papers):
            if i + 1 not in seen:
                reranked_papers.append(paper)
        
        return reranked_papers
    
    def retrieve(self, topic_description):
        """
        执行完整的检索流程
        """
        print("=" * 60)
        print(f"开始检索主题: {topic_description}")
        print("=" * 60)
        
        # 步骤1: 生成arxiv搜索查询
        arxiv_query = self.generate_arxiv_query(topic_description)
        
        # 步骤2: 在arxiv上搜索
        candidate_papers = self.search_arxiv(arxiv_query)
        
        if not candidate_papers:
            print("未找到任何论文")
            return []
        
        # 步骤3: 使用LLM重排序
        reranked_papers = self.rerank_with_llm(topic_description, candidate_papers)
        
        return reranked_papers
    
    def save_results(self, papers, output_file=None):
        """
        保存结果到CSV文件
        """
        if output_file is None:
            output_file = "single_query_results.csv"
        
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", output_file)
        
        df = pd.DataFrame(papers)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"结果已保存到: {output_path}")
        
        return output_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="单次主题检索工具")
    
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="要检索的主题描述（一段话）"
    )
    
    parser.add_argument(
        "-k",
        "--n_candidates",
        type=int,
        default=50,
        help="检索的候选论文数量（默认50）"
    )
    
    parser.add_argument(
        "-q",
        "--n_keywords",
        type=int,
        default=3,
        help="生成的关键词查询数量（默认3）"
    )
    
    parser.add_argument(
        "-m",
        "--gen_engine",
        type=str,
        default="deepseek-chat",
        help="LLM模型名称（默认deepseek-chat）"
    )
    
    parser.add_argument(
        "--reranking_prompt_type",
        type=str,
        default="basic_ranking",
        help="重排序提示类型（默认basic_ranking）"
    )
    
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.2,
        help="LLM温度参数（默认0.2）"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4000,
        help="LLM最大token数（默认4000）"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（可选）"
    )
    
    parser.add_argument(
        "--skip_rerank",
        action="store_true",
        help="跳过LLM重排序，直接返回检索结果"
    )
    
    args = parser.parse_args()
    
    # 创建配置namedtuple
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
    ])
    
    config = Config(
        n_candidates=args.n_candidates,
        n_keywords=args.n_keywords,
        gen_engine=args.gen_engine,
        reranking_prompt_type=args.reranking_prompt_type,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        rerank_method="llm",
        search_engine="arxiv",
        skip_extractive_check=False,
        use_pdf=False,
        use_full_text=False,
    )
    
    return args, config


def main():
    """主函数"""
    args, config = parse_args()
    
    # 设置随机种子
    import random
    random.seed(config.seed)
    
    # 创建检索器
    retrieval = SingleQueryRetrieval(config)
    
    # 执行检索
    if args.skip_rerank:
        # 跳过重排序，直接搜索
        arxiv_query = retrieval.generate_arxiv_query(args.topic)
        papers = retrieval.search_arxiv(arxiv_query)
    else:
        # 完整流程：搜索 + LLM重排序
        papers = retrieval.retrieve(args.topic)
    
    if papers:
        print(f"\n检索完成！找到 {len(papers)} 篇相关论文")
        print("\n前5篇论文：")
        for i, paper in enumerate(papers[:5], 1):
            print(f"\n{i}. {paper['title_paper']}")
            print(f"   arXiv ID: {paper['paper_id']}")
            print(f"   摘要: {paper['abstract'][:200]}...")
        
        # 保存结果
        output_file = args.output or f"topic_retrieval_results.csv"
        retrieval.save_results(papers, output_file)
    else:
        print("未找到相关论文")


if __name__ == "__main__":
    main()

