"""
集成的检索-生成工作流
输入：主题描述
输出：文献综述
"""

import os
import sys
import argparse
import pandas as pd
from collections import namedtuple
import json
import re

# 可选导入spacy（如果未安装也不影响运行）
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from single_query_retrieval import SingleQueryRetrieval
from utils.llm_utils import run_llm_api
from utils.api_utils import get_openalex_candidates
from utils.paper_manager_utils import get_keywords_for_s2_search
import requests
import time
from tqdm import tqdm


def parse_user_prompt(user_prompt):
    """
    解析用户prompt，提取关键信息
    返回：{
        'core_topic': 核心主题（用于检索）,
        'keywords': 关键词列表,
        'time_constraint': 时间限制（如"since 2024"）,
        'special_requirements': 特殊要求（如"comparison", "differences"等）,
        'original_prompt': 原始prompt
    }
    """
    original_prompt = user_prompt.strip()
    
    # 提取时间限制
    time_constraint = None
    time_patterns = [
        r'since\s+(\d{4})',
        r'after\s+(\d{4})',
        r'from\s+(\d{4})',
        r'(\d{4})\s+onwards',
        r'in\s+(\d{4})',
        r'during\s+(\d{4})',
    ]
    for pattern in time_patterns:
        match = re.search(pattern, original_prompt, re.IGNORECASE)
        if match:
            time_constraint = match.group(1) if len(match.groups()) > 0 else match.group(0)
            break
    
    # 提取特殊要求关键词
    special_requirements = []
    requirement_keywords = [
        'comparison', 'compare', 'comparing',
        'difference', 'differences', 'different',
        'highlight', 'highlighting',
        'detailed', 'comprehensive',
        'latest', 'recent', 'current',
        'method', 'methods', 'technique', 'techniques',
        'key technology', 'key technologies',
        'sort out', 'organize', 'summarize'
    ]
    prompt_lower = original_prompt.lower()
    for keyword in requirement_keywords:
        if keyword in prompt_lower:
            special_requirements.append(keyword)
    
    # 提取核心主题和关键词（使用LLM或简单的关键词提取）
    # 简单方法：移除常见指令词，提取核心名词短语
    # 复杂方法可以调用LLM提取
    
    # 先尝试简单提取：移除常见的指令词
    instruction_words = [
        'conduct', 'please', 'i would like', 'i need', 'i want',
        'sort out', 'organize', 'comprehensive', 'detailed',
        'search', 'find', 'retrieve', 'get', 'give', 'provide',
        'for me', 'with a focus on', 'focusing on', 'particularly',
        'cite sources', 'citing sources'
    ]
    
    core_topic = original_prompt.lower()
    for word in instruction_words:
        core_topic = re.sub(rf'\b{word}\b', '', core_topic, flags=re.IGNORECASE)
    
    # 清理多余的空白
    core_topic = re.sub(r'\s+', ' ', core_topic).strip()
    
    # 如果清理后太短，使用原始prompt
    if len(core_topic.split()) < 3:
        core_topic = original_prompt
    
    # 提取关键词（简单的名词短语提取）
    # 这里可以改进为使用更复杂的NLP方法
    keywords = []
    # 提取引号内的内容
    quoted = re.findall(r'["\']([^"\']+)["\']', original_prompt)
    keywords.extend(quoted)
    
    # 提取特定模式，如"key technologies such as X"
    tech_pattern = r'(?:such as|like|including|especially)\s+([^.,;]+)'
    tech_matches = re.findall(tech_pattern, original_prompt, re.IGNORECASE)
    keywords.extend([t.strip() for t in tech_matches])
    
    return {
        'core_topic': core_topic,
        'keywords': keywords if keywords else [],
        'time_constraint': time_constraint,
        'special_requirements': special_requirements,
        'original_prompt': original_prompt
    }


class IntegratedWorkflow:
    """集成的检索-生成工作流类"""
    
    def __init__(self, config):
        """
        初始化工作流
        config: 包含配置参数的namedtuple
        """
        self.config = config
        # 可选加载spacy模型（当前未使用，保留以备将来需要）
        self.nlp_spacy = None
        if SPACY_AVAILABLE:
            try:
                self.nlp_spacy = spacy.load('en_core_web_sm')
            except OSError:
                # spacy已安装但模型未下载，不影响运行
                pass
        
        # 加载prompts
        self.prompts = self.load_prompts()
    
    def load_prompts(self):
        """加载prompt模板"""
        prompts_path = os.path.join(
            os.path.dirname(__file__),
            "resources", "prompts.json"
        )
        
        # 如果prompts.json不存在，使用默认prompts
        if not os.path.exists(prompts_path):
            return {
                "plan_learned_template": """You will be provided with an abstract of a scientific document and other reference papers in triple quotes. Your task is to write the related work section of the document using only the content from provided abstract and all of the other reference papers. Please generate the related work creating a cohesive storyline by doing a critical analysis of related work comparing the strengths and weaknesses while also motivating the new work. You should cite all the other related documents as (@cite_#) whenever referring them in the related work. Do not cite abstract. Do not include any extra notes or newline characters at the end. Do not copy the abstracts of reference papers directly but compare and contrast to the main work concisely. Do not provide the output in bullet points. Do not provide references at the end. Provide the output in max 200 words. You could also group multiple citations in the same line. You should first generate a plan, mentioning the total number of lines, words and the citations to refer to in different lines. You should follow this plan when generating sentences. 

Example: 

Plan: Generate the related work in [number] lines using max [number] words. Cite @cite_# on line [number]. Cite @cite_# on line [number].

""",
                "suffix_plan_learned_template": "Remember, the output should be in the format:\n Plan: \n\n Related Work:"
            }
        
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def format_papers_for_prompt(self, papers, max_papers=40):
        """
        格式化论文为prompt格式
        """
        # 只使用前max_papers篇论文
        papers = papers[:max_papers]
        
        ref_text = ""
        for idx, paper in enumerate(papers, 1):
            title = paper.get("title_paper", "No title")
            abstract = paper.get("abstract", "Abstract not available")
            # 增加摘要长度限制，保留更多信息
            abstract_words = abstract.split()
            if len(abstract_words) > 300:
                abstract = " ".join(abstract_words[:300]) + "..."
            ref_text += f"Reference @cite_{idx}: {title}\n{abstract}\n\n"
        
        return ref_text, papers
    
    def search_crossref(self, query, max_results=50):
        """
        从CrossRef检索论文
        不需要API key，只需要在User-Agent中包含邮箱
        """
        papers = []
        try:
            email = os.getenv('CROSSREF_EMAIL', 'litreview@example.com')
            url = "https://api.crossref.org/works"
            params = {
                "query": query,
                "rows": min(max_results, 100),
                "filter": "type:journal-article"
            }
            headers = {
                'User-Agent': f'LiteratureReview/1.0 (mailto:{email})'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if data and 'message' in data and 'items' in data['message']:
                for item in data['message']['items']:
                    title = item.get('title', [''])[0] if item.get('title') else ''
                    if not title:
                        continue
                    
                    abstract = item.get('abstract', '')
                    if abstract and not isinstance(abstract, str):
                        abstract = ''
                    
                    # 提取作者
                    authors = []
                    if 'author' in item:
                        for author in item['author']:
                            given = author.get('given', '')
                            family = author.get('family', '')
                            if given or family:
                                authors.append(f"{given} {family}".strip())
                    
                    # 提取日期
                    pub_date = 'Unknown'
                    if 'published-print' in item:
                        date_parts = item['published-print'].get('date-parts', [[]])[0]
                        if len(date_parts) >= 3:
                            pub_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                        elif len(date_parts) >= 1:
                            pub_date = str(date_parts[0])
                    
                    # 提取DOI
                    doi = item.get('DOI', '')
                    paper_id = doi.replace('https://doi.org/', '').replace('doi:', '') if doi else f"crossref_{item.get('URL', '').split('/')[-1] if item.get('URL') else ''}"
                    
                    paper = {
                        "paper_id": paper_id,
                        "title_paper": title,
                        "abstract": abstract or "Abstract not available",
                        "citation_count": 0,
                        "publication_date": pub_date,
                        "external_ids": {"doi": doi} if doi else {},
                        "authors": authors,
                        "source": "crossref"
                    }
                    papers.append(paper)
        except Exception as e:
            print(f"CrossRef检索错误: {e}")
        
        return papers
    
    def search_openreview(self, query, max_results=50):
        """
        从OpenReview检索论文
        不需要API key，只需要在User-Agent中包含邮箱
        """
        papers = []
        try:
            email = os.getenv('OPENREVIEW_EMAIL', 'litreview@example.com')
            base_url = "https://api2.openreview.net"
            url = f"{base_url}/notes"
            
            # 使用关键词提取
            query_words = [w for w in query.lower().split() if len(w) >= 4][:5]
            if not query_words:
                query_words = [w for w in query.lower().split() if len(w) >= 3][:5]
            query_words_set = set(query_words)
            
            # 尝试多个会议
            invitations = [
                "ICLR.cc/2024/Conference/-/Blind_Submission",
                "ICLR.cc/2023/Conference/-/Blind_Submission",
                "NeurIPS.cc/2024/Conference/-/Submission",
                "NeurIPS.cc/2023/Conference/-/Submission",
                "ICML.cc/2024/Conference/-/Submission",
            ]
            
            for invitation in invitations:
                if len(papers) >= max_results:
                    break
                
                try:
                    payload = {
                        "invitation": invitation,
                        "limit": 200,
                        "offset": 0,
                    }
                    
                    headers = {
                        'Content-Type': 'application/json',
                        'User-Agent': f'LiteratureReview/1.0 (mailto:{email})'
                    }
                    
                    response = requests.post(url, json=payload, headers=headers, timeout=20)
                    if response.status_code == 200:
                        data = response.json()
                        if data and 'notes' in data:
                            for note in data['notes']:
                                if len(papers) >= max_results:
                                    break
                                
                                content = note.get('content', {})
                                title = content.get('title', '')
                                if isinstance(title, dict):
                                    title = title.get('value', '')
                                
                                if not title:
                                    continue
                                
                                # 关键词匹配
                                title_lower = title.lower()
                                if not any(word in title_lower for word in query_words_set):
                                    continue
                                
                                abstract = content.get('abstract', '')
                                if isinstance(abstract, dict):
                                    abstract = abstract.get('value', '')
                                
                                # 提取作者信息
                                authors = []
                                if 'authors' in content:
                                    authors_list = content.get('authors', [])
                                    if isinstance(authors_list, list):
                                        for author in authors_list:
                                            if isinstance(author, str):
                                                authors.append(author)
                                            elif isinstance(author, dict):
                                                author_name = author.get('name') or author.get('value', '')
                                                if author_name:
                                                    authors.append(author_name)
                                elif 'authorids' in content:
                                    authorids = content.get('authorids', [])
                                    if isinstance(authorids, list):
                                        # 如果有authorids但没有authors，使用authorids作为备用
                                        authors = [str(aid) for aid in authorids[:5]]  # 限制数量
                                
                                paper_id = note.get('id', '').replace('forum?id=', '')
                                
                                paper = {
                                    "paper_id": paper_id,
                                    "title_paper": title,
                                    "abstract": abstract or "Abstract not available",
                                    "citation_count": 0,
                                    "publication_date": str(content.get('year', 'Unknown')),
                                    "external_ids": {"openreview": paper_id},
                                    "authors": authors,  # 添加作者信息
                                    "source": "openreview"
                                }
                                papers.append(paper)
                    
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    continue
        except Exception as e:
            print(f"OpenReview检索错误: {e}")
        
        return papers
    
    def search_multiple_sources(self, topic_description, sources=['arxiv', 'openalex']):
        """
        从多个数据源检索论文
        sources: 数据源列表，可选 ['arxiv', 'openalex', 'crossref', 'openreview', 'unpaywall']
        """
        all_papers = []
        seen_titles = set()
        
        # 生成搜索查询
        retrieval = SingleQueryRetrieval(self.config)
        query = retrieval.generate_arxiv_query(topic_description)
        
        print(f"\n从多个数据源检索论文: {', '.join(sources)}")
        
        # arXiv检索
        if 'arxiv' in sources:
            print("\n[1/{}] 从arXiv检索...".format(len(sources)))
            try:
                arxiv_papers = retrieval.search_arxiv(query, max_results=self.config.n_candidates // len(sources) + 20)
                for paper in arxiv_papers:
                    title = paper.get('title_paper', '').lower()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_papers.append(paper)
                print(f"从arXiv获得 {len(arxiv_papers)} 篇论文")
            except Exception as e:
                print(f"arXiv检索失败: {e}")
        
        # OpenAlex检索
        source_idx = 1
        if 'openalex' in sources:
            source_idx += 1
            print(f"\n[{source_idx}/{len(sources)}] 从OpenAlex检索...")
            try:
                # 使用关键词生成查询
                query_data = get_keywords_for_s2_search(topic_description, self.config)
                if query_data and query_data.get("queries"):
                    queries = query_data.get("queries", [])
                    for q in queries[:2]:  # 使用前2个关键词查询
                        openalex_papers = get_openalex_candidates(
                            query=q,
                            n_candidates=self.config.n_candidates // (len(sources) * 2) + 10
                        )
                        for paper in openalex_papers:
                            title = paper.get('title_paper', '').lower()
                            if title and title not in seen_titles:
                                seen_titles.add(title)
                                all_papers.append(paper)
                print(f"从OpenAlex获得部分论文")
            except Exception as e:
                print(f"OpenAlex检索失败: {e}")
        
        # CrossRef检索
        if 'crossref' in sources:
            source_idx += 1
            print(f"\n[{source_idx}/{len(sources)}] 从CrossRef检索...")
            try:
                crossref_papers = self.search_crossref(
                    query, 
                    max_results=self.config.n_candidates // len(sources) + 10
                )
                crossref_count = 0
                for paper in crossref_papers:
                    title = paper.get("title_paper", '').lower()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_papers.append(paper)
                        crossref_count += 1
                print(f"从CrossRef获得 {crossref_count} 篇唯一论文")
            except Exception as e:
                print(f"CrossRef检索失败: {e}")
        
        # OpenReview检索
        if 'openreview' in sources:
            source_idx += 1
            print(f"\n[{source_idx}/{len(sources)}] 从OpenReview检索...")
            try:
                openreview_papers = self.search_openreview(
                    query,
                    max_results=self.config.n_candidates // len(sources) + 10
                )
                openreview_count = 0
                for paper in openreview_papers:
                    title = paper.get("title_paper", '').lower()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_papers.append(paper)
                        openreview_count += 1
                print(f"从OpenReview获得 {openreview_count} 篇唯一论文")
            except Exception as e:
                print(f"OpenReview检索失败: {e}")
        
        # Unpaywall不支持搜索API，只能通过DOI查询
        if 'unpaywall' in sources:
            source_idx += 1
            print(f"\n[{source_idx}/{len(sources)}] Unpaywall检索")
            print("   提示：Unpaywall不支持搜索API，只能通过DOI查询已有论文的open access信息")
            print("   因此Unpaywall不能用于初始搜索，建议在其他源检索后再使用Unpaywall查询OA信息")
        
        print(f"\n总计从 {len(sources)} 个数据源获得 {len(all_papers)} 篇唯一论文")
        return all_papers
    
    def generate_plan(self, topic_description, papers, parsed_prompt=None):
        """
        第一步：生成plan
        parsed_prompt: 解析后的用户prompt信息
        """
        print(f"\n[步骤1/2] 生成文献综述计划...")
        if parsed_prompt:
            print(f"用户原始请求: {parsed_prompt.get('original_prompt', topic_description)}")
            if parsed_prompt.get('time_constraint'):
                print(f"时间限制: {parsed_prompt['time_constraint']}")
            if parsed_prompt.get('special_requirements'):
                print(f"特殊要求: {', '.join(parsed_prompt['special_requirements'])}")
        print(f"核心主题: {topic_description}")
        print(f"使用 {len(papers)} 篇论文")
        
        # 格式化输入
        ref_text, formatted_papers = self.format_papers_for_prompt(papers, max_papers=self.config.n_papers_for_generation)
        
        # 构建特殊要求说明
        requirements_text = ""
        if parsed_prompt:
            if parsed_prompt.get('time_constraint'):
                requirements_text += f"\n- Time constraint: Focus on papers since {parsed_prompt['time_constraint']} (if applicable)\n"
            if parsed_prompt.get('special_requirements'):
                reqs = parsed_prompt['special_requirements']
                if any('compar' in r for r in reqs) or any('differ' in r for r in reqs):
                    requirements_text += f"- MUST include detailed comparison of different methods, highlighting their differences\n"
                if any('latest' in r or 'recent' in r for r in reqs):
                    requirements_text += f"- MUST emphasize the latest and most recent advances\n"
                if any('comprehensive' in r or 'detailed' in r for r in reqs):
                    requirements_text += f"- MUST be comprehensive and detailed\n"
        
        # 构建plan生成的prompt
        plan_prompt = f"""You are an expert researcher planning a concise literature review based on the following user request:

User Request: "{parsed_prompt.get('original_prompt', topic_description) if parsed_prompt else topic_description}"

CRITICAL REQUIREMENTS:
1. MUST strictly focus on the core topic: "{topic_description}"
2. MUST cite ALL {len(formatted_papers)} provided references (from @cite_1 to @cite_{len(formatted_papers)}) - this is MANDATORY
3. MUST organize the review into 3-4 main sections with brief subsections, approximately 2500-3000 words total
4. MUST create a concise narrative that covers key aspects, methods, and contributions related to the topic
5. MUST include critical analysis and comparisons where relevant
6. MUST ensure every single one of the {len(formatted_papers)} references is cited at least once
{requirements_text}

Generate a focused plan that outlines:
- The overall structure (3-4 main sections with brief subsections, total ~2500-3000 words)
- How you will cite ALL {len(formatted_papers)} references across sections (list which references go in which section)
- The key themes for each section (be concise and focused)
- The logical flow between sections
- A citation distribution plan ensuring all {len(formatted_papers)} references are used
- How to address any special requirements from the user request

Topic: {topic_description}

References:
```{ref_text}```

Plan:"""

        json_data = {
            "prompt": plan_prompt,
            "system_prompt": f"You are a professional researcher creating concise literature review plans. You MUST ensure all {len(formatted_papers)} references are cited. Create focused plans for 2500-3000 word reviews with 3-4 main sections."
        }
        
        print(f"Plan Prompt长度: {len(plan_prompt.split())} words")
        print(f"正在调用LLM生成plan...")
        
        # Plan生成使用较小的max_tokens（2000足够）
        plan_max_tokens = min(2000, 8192)
        
        result = run_llm_api(
            json_data,
            gen_engine=self.config.gen_engine,
            max_tokens=plan_max_tokens,
            temperature=self.config.temperature,
        )
        
        # 提取plan和usage信息
        plan_usage_info = None
        plan_cost = 0
        if isinstance(result, dict):
            plan = result["response"]
            plan_usage_info = result.get("usage", None)
            plan_cost = result.get("cost", 0)
        else:
            plan = result
        
        print(f"Plan生成完成")
        if plan_usage_info:
            print(f"  Plan生成 - 输入tokens: {plan_usage_info.get('prompt_tokens', 'N/A')}, 输出tokens: {plan_usage_info.get('completion_tokens', 'N/A')}")
        
        return plan, formatted_papers, plan_usage_info, plan_cost
    
    def generate_review(self, topic_description, papers, plan, parsed_prompt=None):
        """
        第二步：根据plan生成文献综述
        parsed_prompt: 解析后的用户prompt信息
        """
        print(f"\n[步骤2/2] 根据plan生成文献综述...")
        
        # 格式化输入
        ref_text, formatted_papers = self.format_papers_for_prompt(papers, max_papers=self.config.n_papers_for_generation)
        
        # 构建特殊要求说明
        requirements_text = ""
        if parsed_prompt:
            if parsed_prompt.get('time_constraint'):
                requirements_text += f"\n- Time focus: Emphasize papers and advances since {parsed_prompt['time_constraint']} (if applicable)\n"
            if parsed_prompt.get('special_requirements'):
                reqs = parsed_prompt['special_requirements']
                if any('compar' in r for r in reqs) or any('differ' in r for r in reqs):
                    requirements_text += f"- MUST include detailed comparison of different methods, explicitly highlighting their key differences\n"
                    requirements_text += f"- Compare methods side-by-side where relevant\n"
                if any('latest' in r or 'recent' in r for r in reqs):
                    requirements_text += f"- MUST emphasize the latest and most recent advances in the field\n"
                if any('comprehensive' in r or 'detailed' in r for r in reqs):
                    requirements_text += f"- MUST provide comprehensive coverage and detailed analysis\n"
        
        # 构建review生成的prompt
        user_request = parsed_prompt.get('original_prompt', topic_description) if parsed_prompt else topic_description
        review_prompt = f"""You are an expert researcher writing a concise literature review based on the following user request:

User Request: "{user_request}"

You have created a plan for this literature review. Now write the complete literature review following your plan exactly.

CRITICAL REQUIREMENTS:
1. START with a main title using markdown format: # Your Title Here
   - The title should be concise and descriptive
2. MUST strictly follow the provided plan structure
3. MUST strictly focus on the core topic: "{topic_description}"
4. MUST address the specific requirements mentioned in the user request{requirements_text}
5. **MANDATORY: MUST cite ALL {len(formatted_papers)} provided references (from @cite_1 to @cite_{len(formatted_papers)})**
   - **EVERY SINGLE REFERENCE MUST BE CITED AT LEAST ONCE**
   - **DO NOT skip any references - all {len(formatted_papers)} references must appear in your text**
   - **Count your citations: you must have exactly {len(formatted_papers)} different citations**
6. MUST write in academic style with proper paragraph structure
7. MUST be approximately 2500-3000 words (8-10 paragraphs, organized into 3-4 main sections with brief subsections)
8. MUST include critical analysis and comparisons where relevant
9. MUST use standard citation format: [@cite_X] within the text (where X is the reference number)
   - Use [@cite_1], [@cite_2], etc. as you cite them
   - Each citation will be automatically renumbered to [1], [2], [3] in order of first appearance
10. MUST end with a "References" section listing all {len(formatted_papers)} cited papers in the order they FIRST appear, using standard academic format:
    [1] Author, A., & Author, B. (Year). Title. Journal/Conference, Details.
    [2] Author, C. (Year). Title. Journal/Conference, Details.
    etc.

Topic: {topic_description}

Plan:
{plan}

References:
```{ref_text}```

CRITICAL REMINDERS: 
- **YOU MUST CITE ALL {len(formatted_papers)} REFERENCES - THIS IS MANDATORY**
- **Every single reference from @cite_1 to @cite_{len(formatted_papers)} MUST appear in your text at least once**
- **Before finishing, verify you have cited all {len(formatted_papers)} references**
- **The References section MUST contain exactly {len(formatted_papers)} entries**
- Keep the review concise: 2500-3000 words total, 3-4 main sections
- Be clear and focused - avoid unnecessary verbosity
- Each section should be 1-2 paragraphs with brief subsections
- Distribute all {len(formatted_papers)} references across sections - do not cluster them
- Use the format [@cite_1], [@cite_2], etc. within the text

Literature Review:

"""

        json_data = {
            "prompt": review_prompt,
            "system_prompt": f"You are a professional researcher writing concise literature reviews. You MUST cite all {len(formatted_papers)} references. Write 2500-3000 word reviews with 3-4 main sections. Be concise, clear, and ensure every reference is used."
        }
        
        print(f"Review Prompt长度: {len(review_prompt.split())} words")
        print(f"正在调用LLM生成文献综述（这可能需要几分钟）...")
        
        # Review生成使用较大的max_tokens
        review_max_tokens = min(self.config.max_tokens, 8192)
        if self.config.max_tokens > 8192:
            print(f"Warning: max_tokens ({self.config.max_tokens}) exceeds API limit (8192), using 8192 instead")
        
        result = run_llm_api(
            json_data,
            gen_engine=self.config.gen_engine,
            max_tokens=review_max_tokens,
            temperature=self.config.temperature,
        )
        
        # 提取review和usage信息
        review_usage_info = None
        review_cost = 0
        if isinstance(result, dict):
            review = result["response"]
            review_usage_info = result.get("usage", None)
            review_cost = result.get("cost", 0)
        else:
            review = result
        
        print(f"文献综述生成完成")
        if review_usage_info:
            print(f"  综述生成 - 输入tokens: {review_usage_info.get('prompt_tokens', 'N/A')}, 输出tokens: {review_usage_info.get('completion_tokens', 'N/A')}")
        
        # 解析review和references
        references = ""
        review_text = review
        
        # 检查是否有References部分
        if "References:" in review_text:
            parts = review_text.rsplit("References:", 1)
            review_text = parts[0].strip()
            references = parts[1].strip()
        elif "References\n" in review_text:
            parts = review_text.rsplit("References\n", 1)
            review_text = parts[0].strip()
            references = parts[1].strip()
        elif "\n\nReferences\n" in review_text:
            parts = review_text.rsplit("\n\nReferences\n", 1)
            review_text = parts[0].strip()
            references = parts[1].strip()
        
        return review_text, references, formatted_papers, review_usage_info, review_cost
    
    def format_references_standard(self, citation_map, original_nums, formatted_papers):
        """
        根据citation_map和原始编号顺序，生成规范的References列表
        格式：[编号] Author1, Author2, Author3 et al. (年份). Title. URL
        
        citation_map: 原始编号 -> 新顺序编号的映射
        original_nums: 按照出现顺序的原始编号列表
        """
        
        references_list = []
        # 按照原始编号在正文中出现的顺序生成References
        for idx, original_num in enumerate(original_nums, 1):
            if original_num <= len(formatted_papers) and original_num > 0:
                paper = formatted_papers[original_num - 1]
                title = paper.get('title_paper', 'No title')
                pub_date = paper.get('publication_date', 'Unknown')
                paper_id = paper.get('paper_id', '')
                external_ids = paper.get('external_ids', {})
                
                # 提取作者（如果有）
                authors = paper.get('authors', [])
                if not authors or (isinstance(authors, list) and len(authors) == 0):
                    authors = []
                
                # 格式化作者：最多显示3个作者，超过3个用et al.
                if authors and len(authors) > 0:
                    # 过滤空字符串
                    authors = [a.strip() for a in authors if a and a.strip()]
                    if authors:
                        if len(authors) <= 3:
                            # 1-3个作者：全部列出，用逗号分隔
                            author_str = ", ".join(authors)
                        else:
                            # 超过3个：前3个 + et al.
                            author_str = ", ".join(authors[:3]) + " et al."
                    else:
                        author_str = "Unknown Author"
                else:
                    author_str = "Unknown Author"
                
                # 提取年份
                year = "Unknown"
                if pub_date and pub_date != "Unknown":
                    year_match = re.search(r'(\d{4})', pub_date)
                    if year_match:
                        year = year_match.group(1)
                
                # 构建URL
                url = ""
                
                # 优先使用external_ids中的信息
                if external_ids and isinstance(external_ids, dict):
                    # 1. 优先尝试DOI
                    doi = external_ids.get('doi', '') or external_ids.get('DOI', '')
                    if doi:
                        # 清理DOI格式
                        if isinstance(doi, dict):
                            doi = doi.get('value', '') or doi.get('url', '')
                        doi_str = str(doi).replace('https://doi.org/', '').replace('http://doi.org/', '').replace('doi:', '').strip()
                        if doi_str and doi_str.startswith('10.'):
                            url = f"https://doi.org/{doi_str}"
                    
                    # 2. 如果没有DOI，尝试arXiv
                    if not url:
                        arxiv_id = external_ids.get('arxiv', '') or external_ids.get('arXiv', '') or external_ids.get('ARXIV', '')
                        if arxiv_id:
                            # 清理arXiv ID格式
                            if isinstance(arxiv_id, dict):
                                arxiv_id = arxiv_id.get('value', '') or arxiv_id.get('url', '')
                            arxiv_id_str = str(arxiv_id).replace('arxiv:', '').replace('arXiv:', '').replace('ARXIV:', '').strip()
                            # arXiv ID通常格式：YYMM.NNNNN 或 arXiv:YYMM.NNNNN
                            if arxiv_id_str:
                                # 移除可能的版本号（如v1, v2）
                                arxiv_id_str = re.sub(r'v\d+$', '', arxiv_id_str).strip()
                                url = f"http://arxiv.org/abs/{arxiv_id_str}"
                    
                    # 3. 尝试其他可能的URL字段
                    if not url:
                        for key in ['url', 'URL', 'link', 'Link']:
                            if key in external_ids:
                                url_val = external_ids[key]
                                if url_val and isinstance(url_val, str):
                                    if url_val.startswith('http'):
                                        url = url_val
                                        break
                
                # 如果还没有URL，尝试从paper_id构建
                if not url:
                    if paper_id:
                        # 检查是否是DOI格式（以10.开头）
                        if paper_id.startswith('10.'):
                            url = f"https://doi.org/{paper_id}"
                        # 检查是否是arXiv ID格式（通常是YYMM.NNNNN，如2511.10629）
                        elif re.match(r'^\d{4}\.\d{5}(v\d+)?$', paper_id) or re.match(r'^[a-z-]+/\d{7}v?\d+$', paper_id):
                            # arXiv ID格式：YYMM.NNNNN 或 old-style: category/YYMMNNN
                            arxiv_id_clean = re.sub(r'v\d+$', '', paper_id).strip()
                            url = f"http://arxiv.org/abs/{arxiv_id_clean}"
                        elif paper_id.startswith('crossref_'):
                            # Crossref ID，没有直接URL，跳过
                            pass
                        elif 'arxiv' in paper_id.lower():
                            # 包含arxiv关键字
                            arxiv_id = re.sub(r'.*arxiv[:\s]*', '', paper_id, flags=re.IGNORECASE)
                            arxiv_id = re.sub(r'v\d+$', '', arxiv_id).strip()
                            if arxiv_id:
                                url = f"http://arxiv.org/abs/{arxiv_id}"
                        elif len(paper_id) > 8 and '.' in paper_id:
                            # 可能是arXiv ID（包含点，长度合理）
                            # 尝试匹配YYMM.NNNNN格式
                            if re.match(r'^\d{4}\.\d+', paper_id):
                                arxiv_id_clean = re.sub(r'v\d+$', '', paper_id).strip()
                                url = f"http://arxiv.org/abs/{arxiv_id_clean}"
                        elif paper_id and external_ids and isinstance(external_ids, dict):
                            # 尝试从external_ids获取OpenReview URL
                            if 'openreview' in external_ids:
                                openreview_id = external_ids.get('openreview', '')
                                if openreview_id:
                                    url = f"https://openreview.net/forum?id={openreview_id}"
                
                # 如果没有URL，尝试从paper source或paper_id获取
                if not url:
                    source = paper.get('source', '')
                    if source == 'openreview' and paper_id:
                        url = f"https://openreview.net/forum?id={paper_id}"
                    elif source == 'crossref' and paper_id and not paper_id.startswith('crossref_'):
                        # 如果有DOI
                        if paper_id.startswith('10.'):
                            url = f"https://doi.org/{paper_id}"
                    elif source == 'openalex' or (external_ids and 'openalex' in str(external_ids).lower()):
                        # OpenAlex URL格式：https://openalex.org/W{paper_id}
                        if paper_id and not paper_id.startswith('crossref_'):
                            # OpenAlex ID通常是W开头的数字，或者完整的URL
                            if paper_id.startswith('W'):
                                url = f"https://openalex.org/{paper_id}"
                            elif 'openalex.org' in paper_id:
                                url = paper_id
                            # 也可以从external_ids中获取
                            if not url and external_ids and isinstance(external_ids, dict):
                                openalex_url = external_ids.get('openalex', '')
                                if openalex_url:
                                    url = str(openalex_url)
                    elif source == 'arxiv' or (paper_id and '.' in paper_id and re.match(r'^\d{4}\.\d+', paper_id)):
                        # arXiv论文
                        if paper_id:
                            arxiv_id_clean = re.sub(r'v\d+$', '', paper_id).strip()
                            url = f"http://arxiv.org/abs/{arxiv_id_clean}"
                
                # 生成标准格式引用：[编号] Authors (年份). Title. URL
                new_num = citation_map.get(original_num, idx)
                ref_format = f"[{new_num}] {author_str} ({year}). {title}"
                
                # 确保标题以点结尾（如果还没有）
                if not ref_format.endswith('.'):
                    ref_format += "."
                
                # 添加URL
                if url:
                    ref_format += f" {url}"
                
                references_list.append(ref_format)
        
        # 每行参考文献之间添加空行，使得在PDF中每篇参考文献单独成行显示
        return "\n\n".join(references_list)
    
    def ensure_all_papers_cited(self, review_text, formatted_papers):
        """
        检查是否所有论文都被引用了，如果没有，尝试在合适的位置添加引用
        返回更新后的review文本和未引用的论文列表
        """
        # 提取所有已引用的原始编号
        cited_nums = set()
        for match in re.finditer(r'[@\[\(]cite_(\d+)[\)\]]?', review_text):
            num = int(match.group(1))
            cited_nums.add(num)
        
        # 找出未引用的论文
        all_nums = set(range(1, len(formatted_papers) + 1))
        uncited_nums = sorted(all_nums - cited_nums)
        
        if uncited_nums:
            print(f"\n警告: 有 {len(uncited_nums)} 篇论文未被引用: {uncited_nums}")
            print(f"建议: 请确保所有{len(formatted_papers)}篇文献都被引用到文献综述中")
            
            # 尝试在合适的位置添加引用（在相关段落末尾）
            # 这里不自动添加，只给出警告
            # 可以手动在prompt中要求LLM确保所有文献都被引用
        
        return review_text, uncited_nums
    
    def normalize_citations(self, text, formatted_papers):
        """
        将引用格式标准化并重新编号：@cite_X -> [顺序编号]
        按照在正文中第一次出现的顺序重新编号（1, 2, 3, ...）
        """
        
        # 提取所有引用的原始编号，按照出现顺序
        original_nums = []
        for match in re.finditer(r'[@\[\(]cite_(\d+)[\)\]]?', text):
            num = int(match.group(1))
            if num not in original_nums:
                original_nums.append(num)
        
        # 创建映射：原始编号 -> 新顺序编号
        citation_map = {}
        for idx, original_num in enumerate(original_nums, 1):
            citation_map[original_num] = idx
        
        # 替换所有引用为新的顺序编号
        def replace_cite(match):
            original_num = int(match.group(1))
            new_num = citation_map.get(original_num, original_num)
            return f"[{new_num}]"
        
        # 先处理已经是 [@cite_X] 格式的（避免双重括号）
        # 匹配 [@cite_X] 或 (@cite_X) 或 @cite_X
        text = re.sub(r'\[@cite_(\d+)\]', replace_cite, text)
        text = re.sub(r'\(@cite_(\d+)\)', replace_cite, text)
        text = re.sub(r'@cite_(\d+)', replace_cite, text)
        
        # 处理其他可能的格式
        text = re.sub(r'\[cite_(\d+)\]', replace_cite, text)
        
        # 修复双重括号问题：[[数字] -> [数字]
        text = re.sub(r'\[\[(\d+)\]', r'[\1]', text)
        # 修复三重或更多括号：[[[数字]]] -> [数字]
        text = re.sub(r'\[{2,}(\d+)\]+\]', r'[\1]', text)
        
        # 修复多重右中括号问题（在分号、逗号或空格后的多重右括号）
        # 例如：[21]; [22]] -> [21]; [22]
        # 例如：[21], [22]] -> [21], [22]
        # 例如：[21] [22]] -> [21] [22]
        text = re.sub(r'(\[\d+\])\s*([,;])\s*(\[\d+\])\]', r'\1\2 \3', text)  # [21]; [22]] -> [21]; [22]
        text = re.sub(r'(\[\d+\])\s+(\[\d+\])\]', r'\1 \2', text)  # [21] [22]] -> [21] [22]
        
        # 修复所有剩余的多重右括号：[数字]] -> [数字]
        while re.search(r'\[(\d+)\]\]', text):
            text = re.sub(r'\[(\d+)\]\]', r'[\1]', text)
        
        # 修复连续的多重右括号：[数字]]] -> [数字]
        text = re.sub(r'\[(\d+)\]+', r'[\1]', text)
        
        # 返回重新编号后的文本和引用映射
        return text, citation_map, original_nums
    
    def clean_multiple_right_brackets(self, text):
        """
        清理文本中的多重右中括号问题
        例如：[21]; [22]] -> [21]; [22]
        """
        # 修复分号或逗号后的多重右括号：[21]; [22]] -> [21]; [22]
        # 匹配模式：[数字]; [数字]] 或 [数字], [数字]]
        text = re.sub(r'(\[\d+\])\s*([,;])\s*(\[\d+\])\]', r'\1\2 \3', text)
        
        # 修复空格分隔的多个引用后的多重右括号：[21] [22]] -> [21] [22]
        text = re.sub(r'(\[\d+\])\s+(\[\d+\])\]', r'\1 \2', text)
        
        # 修复单个引用后的多重右括号：[21]] -> [21]
        # 使用循环确保所有情况都被处理
        max_iterations = 10  # 防止无限循环
        iteration = 0
        while re.search(r'\[(\d+)\]\]', text) and iteration < max_iterations:
            text = re.sub(r'\[(\d+)\]\]', r'[\1]', text)
            iteration += 1
        
        # 修复连续的多重右括号：[数字]]] -> [数字]（处理三层或更多层）
        # 匹配一个或多个右括号
        text = re.sub(r'\[(\d+)\]+', r'[\1]', text)
        
        # 修复其他可能的模式：例如 [21]]; [22] -> [21]; [22]
        text = re.sub(r'\[(\d+)\]\]([;,\s])', r'[\1]\2', text)
        
        # 修复括号内的多重右括号：([21]]) -> ([21])
        text = re.sub(r'\(\[(\d+)\]\]\)', r'([\1])', text)
        
        # 最后再次检查并修复所有剩余的多重右括号
        text = re.sub(r'\[(\d+)\]+', r'[\1]', text)
        
        return text
    
    def generate_plan_and_review(self, topic_description, papers, parsed_prompt=None):
        """
        分两步生成plan和文献综述（分离API调用）
        parsed_prompt: 解析后的用户prompt信息
        """
        print(f"\n正在生成plan和文献综述（分两步）...")
        if parsed_prompt:
            print(f"用户原始请求: {parsed_prompt.get('original_prompt', topic_description)}")
        print(f"核心主题: {topic_description}")
        print(f"使用 {len(papers)} 篇论文")
        
        # 第一步：生成plan
        plan, formatted_papers, plan_usage_info, plan_cost = self.generate_plan(topic_description, papers, parsed_prompt)
        
        # 第二步：根据plan生成review
        review, references, _, review_usage_info, review_cost = self.generate_review(topic_description, papers, plan, parsed_prompt)
        
        # 合并usage信息
        total_usage_info = None
        total_cost = plan_cost + review_cost
        
        if plan_usage_info and review_usage_info:
            total_usage_info = {
                "plan": {
                    "prompt_tokens": plan_usage_info.get('prompt_tokens', 0),
                    "completion_tokens": plan_usage_info.get('completion_tokens', 0),
                    "total_tokens": plan_usage_info.get('total_tokens', 0)
                },
                "review": {
                    "prompt_tokens": review_usage_info.get('prompt_tokens', 0),
                    "completion_tokens": review_usage_info.get('completion_tokens', 0),
                    "total_tokens": review_usage_info.get('total_tokens', 0)
                },
                "total": {
                    "prompt_tokens": plan_usage_info.get('prompt_tokens', 0) + review_usage_info.get('prompt_tokens', 0),
                    "completion_tokens": plan_usage_info.get('completion_tokens', 0) + review_usage_info.get('completion_tokens', 0),
                    "total_tokens": plan_usage_info.get('total_tokens', 0) + review_usage_info.get('total_tokens', 0)
                }
            }
        elif plan_usage_info:
            total_usage_info = {"plan": plan_usage_info, "review": None}
        elif review_usage_info:
            total_usage_info = {"plan": None, "review": review_usage_info}
        
        # 检查是否所有论文都被引用了（在重新编号之前检查）
        _, uncited_nums = self.ensure_all_papers_cited(review, formatted_papers)
        if uncited_nums:
            print(f"\n注意: 以下论文编号未被引用（原始编号）: {uncited_nums}")
            print(f"      共 {len(uncited_nums)} 篇论文未被引用")
            print("      建议在文献综述中确保所有论文都被引用")
        
        # 规范化引用格式并重新编号（按照出现顺序）
        review_normalized, citation_map, original_nums = self.normalize_citations(review, formatted_papers)
        
        # 再次清理多重右中括号（确保完全清理）
        review_normalized = self.clean_multiple_right_brackets(review_normalized)
        
        # 如果没有生成references，根据citation_map和original_nums生成
        if not references:
            references = self.format_references_standard(citation_map, original_nums, formatted_papers)
        else:
            # 即使有references，也要按照新的顺序重新生成，确保编号连续
            references = self.format_references_standard(citation_map, original_nums, formatted_papers)
        
        return plan, review_normalized, references, total_usage_info, total_cost, formatted_papers
    
    def run(self, topic_description, parsed_prompt=None):
        """
        运行完整的检索-生成工作流
        parsed_prompt: 解析后的用户prompt信息（可选）
        """
        # 记录开始时间
        start_time = time.time()
        
        print("=" * 80)
        print("开始集成的检索-生成工作流")
        print("=" * 80)
        
        # 如果提供了parsed_prompt，使用core_topic进行检索
        search_topic = parsed_prompt.get('core_topic', topic_description) if parsed_prompt else topic_description
        
        # 步骤1: 检索论文
        print("\n[步骤1/3] 检索相关论文...")
        print(f"检索主题: {search_topic}")
        
        # 如果配置了多源检索，使用多源检索
        if hasattr(self.config, 'search_sources') and self.config.search_sources:
            papers = self.search_multiple_sources(search_topic, self.config.search_sources)
        else:
            retrieval = SingleQueryRetrieval(self.config)
            if self.config.skip_rerank:
                # 跳过重排序，直接搜索
                arxiv_query = retrieval.generate_arxiv_query(search_topic)
                papers = retrieval.search_arxiv(arxiv_query)
            else:
                # 完整流程：搜索 + LLM重排序
                papers = retrieval.retrieve(search_topic)
        
        if not papers:
            print("错误：未找到任何论文")
            return None, None, None
        
        print(f"检索完成，找到 {len(papers)} 篇论文")
        
        # 步骤2: 选择前N篇论文用于生成
        n_papers = min(self.config.n_papers_for_generation, len(papers))
        selected_papers = papers[:n_papers]
        print(f"\n[步骤2/3] 选择前 {n_papers} 篇论文用于生成（配置值: {self.config.n_papers_for_generation}，实际使用: {n_papers}）...")
        if n_papers < self.config.n_papers_for_generation:
            print(f"注意: 检索到的论文数量({len(papers)})少于配置值({self.config.n_papers_for_generation})，实际使用 {n_papers} 篇论文")
        
        # 步骤3: 生成plan和文献综述
        print(f"\n[步骤3/3] 生成plan和文献综述...")
        plan, review, references, usage_info, cost, formatted_papers = self.generate_plan_and_review(topic_description, selected_papers, parsed_prompt)
        
        # 如果没有生成references，根据review中的引用顺序生成
        # 这部分已经在generate_plan_and_review中处理，这里不需要重复处理
        
        # 计算总用时
        end_time = time.time()
        total_time = end_time - start_time
        
        # 输出token和cost信息
        print("\n" + "=" * 80)
        print("LLM使用统计:")
        print("=" * 80)
        if usage_info:
            print(f"输入tokens: {usage_info.get('prompt_tokens', 'N/A')}")
            print(f"输出tokens: {usage_info.get('completion_tokens', 'N/A')}")
            print(f"总计tokens: {usage_info.get('total_tokens', 'N/A')}")
        if cost > 0:
            print(f"费用 (USD): ${cost:.6f}")
        print("=" * 80)
        
        # 输出总用时
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        if hours > 0:
            print(f"总用时: {hours}小时 {minutes}分钟 {seconds}秒 ({total_time:.2f}秒)")
        elif minutes > 0:
            print(f"总用时: {minutes}分钟 {seconds}秒 ({total_time:.2f}秒)")
        else:
            print(f"总用时: {seconds}秒 ({total_time:.2f}秒)")
        
        # 保存结果
        result = {
            "topic": topic_description,
            "n_papers_retrieved": len(papers),
            "n_papers_used": n_papers,
            "plan": plan,
            "review": review,
            "references": references,
            "papers_used": formatted_papers,
            "usage_info": usage_info,
            "cost": cost
        }
        
        return result, plan, review, references
    
    def save_results(self, result, output_file=None):
        """
        保存结果
        包括完整报告和单独的文献综述文件
        """
        if output_file is None:
            output_file = "integrated_workflow_results.json"
        
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", output_file)
        
        # 保存JSON（完整结果）
        result_to_save = {k: v for k, v in result.items() if k != 'usage_info'}
        if result.get('usage_info'):
            usage_dict = {}
            if isinstance(result['usage_info'], dict):
                if 'plan' in result['usage_info']:
                    usage_dict['plan'] = result['usage_info']['plan']
                if 'review' in result['usage_info']:
                    usage_dict['review'] = result['usage_info']['review']
                if 'total' in result['usage_info']:
                    usage_dict['total'] = result['usage_info']['total']
            result_to_save['usage_info'] = usage_dict
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_to_save, f, indent=2, ensure_ascii=False)
        
        # 保存完整markdown报告（包含plan和统计信息）
        md_path = output_path.replace('.json', '.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Literature Review Report\n\n")
            f.write(f"## Topic\n\n{result['topic']}\n\n")
            
            # 添加统计信息
            if result.get('usage_info'):
                f.write(f"## Usage Statistics\n\n")
                usage_info = result['usage_info']
                if isinstance(usage_info, dict):
                    if 'plan' in usage_info and usage_info['plan']:
                        plan_usage = usage_info['plan']
                        f.write(f"### Plan Generation\n")
                        f.write(f"- Input tokens: {plan_usage.get('prompt_tokens', 'N/A')}\n")
                        f.write(f"- Output tokens: {plan_usage.get('completion_tokens', 'N/A')}\n")
                        f.write(f"- Total tokens: {plan_usage.get('total_tokens', 'N/A')}\n\n")
                    
                    if 'review' in usage_info and usage_info['review']:
                        review_usage = usage_info['review']
                        f.write(f"### Review Generation\n")
                        f.write(f"- Input tokens: {review_usage.get('prompt_tokens', 'N/A')}\n")
                        f.write(f"- Output tokens: {review_usage.get('completion_tokens', 'N/A')}\n")
                        f.write(f"- Total tokens: {review_usage.get('total_tokens', 'N/A')}\n\n")
                    
                    if 'total' in usage_info:
                        total_usage = usage_info['total']
                        f.write(f"### Total\n")
                        f.write(f"- Total input tokens: {total_usage.get('prompt_tokens', 'N/A')}\n")
                        f.write(f"- Total output tokens: {total_usage.get('completion_tokens', 'N/A')}\n")
                        f.write(f"- Total tokens: {total_usage.get('total_tokens', 'N/A')}\n")
                
                if result.get('cost', 0) > 0:
                    f.write(f"- **Total Cost (USD): ${result['cost']:.6f}**\n")
                f.write(f"\n")
            
            f.write(f"## Plan\n\n{result['plan']}\n\n")
            f.write(f"## Literature Review\n\n{result['review']}\n\n")
            
            if result.get('references'):
                f.write(f"## References\n\n{result['references']}\n\n")
            
            f.write(f"## Papers Used ({result['n_papers_used']} papers)\n\n")
            for idx, paper in enumerate(result['papers_used'], 1):
                f.write(f"{idx}. **{paper['title_paper']}**\n")
                paper_id = paper.get('paper_id', 'N/A')
                if paper_id != 'N/A':
                    f.write(f"   - Paper ID: {paper_id}\n")
                f.write(f"   - Abstract: {paper.get('abstract', 'N/A')[:300]}...\n\n")
        
        # 保存单独的、可直接使用的文献综述文件（不包含plan和统计信息）
        review_only_path = output_path.replace('.json', '_review_only.md')
        with open(review_only_path, 'w', encoding='utf-8') as f:
            # 检查review是否已经有标题（第一个#开头的行）
            review_text = result['review'].strip()
            review_lines = review_text.split('\n')
            has_title = False
            
            # 检查前5行是否包含主标题（# 开头，不是 ##）
            for line in review_lines[:5]:
                stripped = line.strip()
                if stripped.startswith('# ') and not stripped.startswith('##'):
                    has_title = True
                    break
            
            # 在保存前再次清理多重右中括号（确保最终输出干净）
            review_text = self.clean_multiple_right_brackets(review_text)
            
            # 如果已有标题，直接写入review（不添加额外标题）
            # 如果没有标题，从review的第一段提取或使用topic生成一个标题
            if has_title:
                # 直接写入review，它已经包含标题
                f.write(review_text)
            else:
                # 如果没有标题，添加一个基于topic的标题
                topic_title = result['topic'].title()
                f.write(f"# Literature Review: {topic_title}\n\n")
                f.write(review_text)
            
            f.write(f"\n\n")
            
            # 写入References（检查是否已经包含在review中）
            if result.get('references'):
                review_lower = review_text.lower()
                # 检查是否已经有References部分
                if '## references' not in review_lower and '\n\n## references' not in review_lower:
                    f.write(f"## References\n\n")
                    f.write(result['references'])
                    f.write(f"\n")
        
        # 保存纯文本格式的文献综述（无markdown格式）
        review_txt_path = output_path.replace('.json', '_review_only.txt')
        with open(review_txt_path, 'w', encoding='utf-8') as f:
            # 移除markdown格式标记
            review_text = result['review']
            # 移除粗体、斜体等markdown标记
            review_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', review_text)
            review_text = re.sub(r'\*([^*]+)\*', r'\1', review_text)
            review_text = re.sub(r'#+\s+', '', review_text)  # 移除标题标记
            
            f.write(f"Literature Review: {result['topic']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(review_text)
            f.write(f"\n\n")
            f.write("References\n")
            f.write("=" * 80 + "\n\n")
            if result.get('references'):
                # 移除markdown格式
                refs_text = result['references']
                refs_text = re.sub(r'\[(\d+)\]', r'[\1]', refs_text)  # 保留引用编号
                f.write(refs_text)
        
        print(f"\n结果已保存到:")
        print(f"  - JSON（完整结果）: {output_path}")
        print(f"  - Markdown（完整报告）: {md_path}")
        print(f"  - Markdown（纯文献综述，可直接使用）: {review_only_path}")
        print(f"  - Text（纯文本格式）: {review_txt_path}")
        
        return output_path, md_path, review_only_path, review_txt_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="集成的检索-生成工作流")
    
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
        "--n_papers_for_generation",
        type=int,
        default=40,
        help="用于生成综述的论文数量（默认40，建议35-40）"
    )
    
    parser.add_argument(
        "-q",
        "--n_keywords",
        type=int,
        default=3,
        help="生成的关键词查询数量（默认3，至少2-3个）"
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
        default=8192,
        help="LLM最大token数（默认8192，API限制最大值，用于生成长篇综述）"
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
        "--search_sources",
        type=str,
        nargs="+",
        default=['arxiv', 'openalex', 'crossref', 'openreview', 'unpaywall'],
        choices=['arxiv', 'openalex', 'crossref', 'openreview', 'unpaywall'],
        help="检索数据源列表（默认: arxiv openalex crossref openreview unpaywall）。注意：crossref/openreview/unpaywall需要额外API配置，当前会跳过"
    )
    
    parser.add_argument(
        "--skip_rerank",
        action="store_true",
        help="跳过LLM重排序，直接返回检索结果（默认False，会进行重排序）"
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
        "n_papers_for_generation",
        "skip_rerank",
        "search_sources",
    ])
    
    # 确保n_keywords至少为3
    n_keywords = max(args.n_keywords, 3)
    if args.n_keywords < 3:
        print(f"Warning: n_keywords ({args.n_keywords}) is less than 3, using 3 instead")
    
    # 确保max_tokens不超过API限制
    max_tokens = min(args.max_tokens, 8192)
    if args.max_tokens > 8192:
        print(f"Warning: max_tokens ({args.max_tokens}) exceeds API limit (8192), using 8192 instead")
    
    config = Config(
        n_candidates=args.n_candidates,
        n_keywords=n_keywords,
        gen_engine=args.gen_engine,
        reranking_prompt_type=args.reranking_prompt_type,
        temperature=args.temperature,
        max_tokens=max_tokens,
        seed=args.seed,
        rerank_method="llm",
        search_engine="arxiv",
        skip_extractive_check=False,
        use_pdf=False,
        use_full_text=False,
        n_papers_for_generation=args.n_papers_for_generation,
        skip_rerank=args.skip_rerank,
        search_sources=args.search_sources,
    )
    
    return args, config


def main():
    """主函数"""
    args, config = parse_args()
    
    # 解析用户prompt
    parsed_prompt = parse_user_prompt(args.topic)
    
    # 提取核心主题用于检索
    core_topic = parsed_prompt['core_topic']
    
    # 设置随机种子
    import random
    random.seed(config.seed)
    
    # 创建工作流
    workflow = IntegratedWorkflow(config)
    
    # 运行工作流（传入core_topic和parsed_prompt）
    result, plan, review, references = workflow.run(core_topic, parsed_prompt)
    
    if result:
        print("\n" + "=" * 80)
        print("工作流完成！")
        print("=" * 80)
        
        print(f"\n生成的Plan:")
        print("-" * 80)
        print(plan[:500] + "..." if len(plan) > 500 else plan)
        
        print(f"\n生成的文献综述 (前1000字符):")
        print("-" * 80)
        print(review[:1000] + "..." if len(review) > 1000 else review)
        
        if references:
            print(f"\n生成的References (前500字符):")
            print("-" * 80)
            print(references[:500] + "..." if len(references) > 500 else references)
        
        # 保存结果
        output_file = args.output or f"integrated_workflow_results_{args.topic[:20].replace(' ', '_')}.json"
        workflow.save_results(result, output_file)
    else:
        print("工作流执行失败")


if __name__ == "__main__":
    main()

