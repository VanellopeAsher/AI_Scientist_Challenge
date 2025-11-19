import asyncio
import json
import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.llm_utils import run_llm_api


DEFAULT_LOCAL_RANKER_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "max_candidates": 150,
    "top_k": 40,
    "weights": {
        "similarity": 0.45,
        "norm_impact": 0.60,
        "intent_weight": 0.40,
        "FICE_like": 0.1,
    },
    "llm_model": os.getenv("SCI_LLM_MODEL", "deepseek-chat"),
    "temperature": 0.1,
    "max_tokens": 900,
    "max_concurrency": 12,
    "max_retries": 3,
    "retry_delay": 2.0,
    "cache_dir": None,
    "use_cache": True,
    "prompt_version": "v1",
}

SYSTEM_PROMPT = (
    "你是一名科学文献匹配与评估助手，负责根据提纲段落对候选论文进行多维度评分。"
    "严格遵守指令，只输出JSON，分数均在0到1之间。"
)

PROMPT_TEMPLATE = """你将执行四个独立的评分任务，并返回一个JSON：

1. similarity：判断候选论文能否直接支撑提纲段落，输出0-1分和一句理由。
2. norm_impact：综合发表年份、期刊/会议和引用情况，评估学术影响力，输出0-1分和一句理由。
3. intent_weight：论文对提纲段落写作目标的贡献度，0-1分并一句理由。
4. FICE_like：论文的新颖性/信息量/主题覆盖程度，0-1分并一句理由。

【提纲段落】
{outline_text}

【候选文献】
Title: {title}
Abstract: {abstract}
Publication Year: {year}
Venue or Source: {venue}
Raw Citation Count: {citations}

输出格式（必须是合法JSON）：
{{
  "similarity": {{"score": 0.x, "reason": "..." }},
  "norm_impact": {{"score": 0.x, "reason": "..." }},
  "intent_weight": {{"score": 0.x, "reason": "..." }},
  "FICE_like": {{"score": 0.x, "reason": "..." }}
}}
"""


def _to_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def load_local_ranker_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = DEFAULT_LOCAL_RANKER_CONFIG.copy()

    env_toggle = os.getenv("LOCAL_LLM_RANKING")
    if env_toggle is not None:
        config["enabled"] = _to_bool(env_toggle, config["enabled"])

    cfg_path = config_path or os.getenv("LOCAL_LLM_RANKER_CONFIG")
    if cfg_path and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            for key, value in user_cfg.items():
                if isinstance(config.get(key), dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
        except (json.JSONDecodeError, OSError):
            print(f"[LocalRanker] 警告：无法读取配置文件 {cfg_path}，将使用默认配置")

    if config["cache_dir"] is None:
        base_dir = Path(__file__).resolve().parents[2]
        cache_dir = base_dir / "cache" / "ranking"
        config["cache_dir"] = str(cache_dir)

    return config


class ScoreCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, key: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError:
            pass


class LocalRanker:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_local_ranker_config()
        self.cache = ScoreCache(self.config["cache_dir"], self.config.get("use_cache", True))
        self.weights = self.config.get("weights", DEFAULT_LOCAL_RANKER_CONFIG["weights"])

    def rank(
        self,
        outline_text: str,
        candidate_docs: List[Dict[str, Any]],
        topic_description: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.config.get("enabled", True):
            return candidate_docs
        if not candidate_docs:
            return candidate_docs

        top_n = min(len(candidate_docs), int(self.config.get("max_candidates", 150)))
        subset = candidate_docs[:top_n]
        outline_text = outline_text.strip() or (topic_description or "")

        print(
            f"\n[LocalRanker] 启动LLM多指标评分：候选 {len(candidate_docs)} -> "
            f"取前 {top_n} 篇进行评分（并发 {self.config.get('max_concurrency')}）"
        )

        tasks = [
            self._build_task(outline_text, paper, idx)
            for idx, paper in enumerate(subset)
        ]

        scored = self._execute_tasks(tasks)
        enriched: List[Dict[str, Any]] = []
        for paper, detail in scored:
            if detail is None:
                detail = self._empty_score_detail("评分失败，使用默认0分")
            total_score = self._combine_scores(detail)
            paper["llm_rank_score"] = total_score
            paper["llm_rank_detail"] = detail
            enriched.append(paper)

        enriched.sort(key=lambda x: x.get("llm_rank_score", 0), reverse=True)
        top_k = min(len(enriched), int(self.config.get("top_k", 40)))
        print(f"[LocalRanker] 评分完成，选出前 {top_k} 篇供后续流程使用")
        return enriched[:top_k]

    def _build_task(self, outline_text: str, paper: Dict[str, Any], idx: int) -> Dict[str, Any]:
        title = paper.get("title_paper") or paper.get("title") or f"Untitled-{idx}"
        abstract = (paper.get("abstract") or "Abstract not available").strip()
        abstract = " ".join(abstract.split())
        year = self._extract_year(paper)
        venue = paper.get("venue") or paper.get("journal") or paper.get("source") or "Unknown"
        citations = paper.get("citation_count", 0) or 0

        prompt = PROMPT_TEMPLATE.format(
            outline_text=outline_text,
            title=title,
            abstract=abstract[:2000],
            year=year,
            venue=venue,
            citations=citations,
        )

        payload = {
            "system_prompt": SYSTEM_PROMPT,
            "prompt": prompt,
        }

        cache_key = self._make_cache_key(outline_text, paper)
        return {
            "paper": paper,
            "payload": payload,
            "cache_key": cache_key,
        }

    def _execute_tasks(self, tasks: List[Dict[str, Any]]):
        async def runner():
            semaphore = asyncio.Semaphore(int(self.config.get("max_concurrency", 12)))
            coros = [self._score_single(task, semaphore) for task in tasks]
            return await asyncio.gather(*coros)

        try:
            return asyncio.run(runner())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(runner())
            finally:
                loop.close()

    async def _score_single(self, task: Dict[str, Any], semaphore: asyncio.Semaphore):
        cached = self.cache.load(task["cache_key"])
        if cached:
            return task["paper"], cached

        retries = int(self.config.get("max_retries", 3))
        delay = float(self.config.get("retry_delay", 2.0))
        for attempt in range(retries):
            try:
                async with semaphore:
                    response_text = await asyncio.to_thread(
                        self._call_llm, task["payload"]
                    )
                detail = self._parse_response(response_text)
                if detail:
                    self.cache.save(task["cache_key"], detail)
                    return task["paper"], detail
            except Exception as exc:
                print(f"[LocalRanker] 评分失败（第{attempt + 1}次）：{exc}")
            await asyncio.sleep(delay)

        return task["paper"], None

    def _call_llm(self, payload: Dict[str, Any]) -> str:
        result = run_llm_api(
            payload,
            gen_engine=self.config.get("llm_model", "deepseek-chat"),
            max_tokens=int(self.config.get("max_tokens", 900)),
            temperature=float(self.config.get("temperature", 0.1)),
        )
        if isinstance(result, dict):
            return result.get("response", "")
        return str(result)

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        candidate = self._extract_json_block(text)
        if not candidate:
            return None
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        detail = {
            "similarity": self._extract_field(data, "similarity"),
            "norm_impact": self._extract_field(data, "norm_impact"),
            "intent_weight": self._extract_field(data, "intent_weight"),
            "FICE_like": self._extract_field(data, "FICE_like"),
        }
        return detail

    def _extract_field(self, data: Dict[str, Any], key: str) -> Dict[str, Any]:
        raw = data.get(key, {})
        if isinstance(raw, dict):
            score = raw.get("score")
            reason = raw.get("reason") or raw.get("message") or ""
        else:
            score = raw
            reason = ""
        score = self._clamp_score(score)
        return {"score": score, "reason": reason}

    @staticmethod
    def _clamp_score(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _combine_scores(self, detail: Dict[str, Any]) -> float:
        total = 0.0
        for key, weight in self.weights.items():
            field = detail.get(key, {})
            total += weight * field.get("score", 0.0)
        return round(total, 6)

    def _make_cache_key(self, outline_text: str, paper: Dict[str, Any]) -> str:
        outline_hash = hashlib.sha256(
            (outline_text + self.config.get("prompt_version", "v1")).encode("utf-8")
        ).hexdigest()
        paper_id = paper.get("paper_id") or paper.get("title_paper") or "unknown"
        safe_meta = self._make_json_safe(paper)
        meta_bytes = json.dumps(safe_meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
        meta_hash = hashlib.sha256(meta_bytes).hexdigest()
        return f"{outline_hash}_{paper_id}_{meta_hash}"

    @staticmethod
    def _extract_year(paper: Dict[str, Any]) -> str:
        if paper.get("year"):
            return str(paper["year"])
        pub_date = paper.get("publication_date") or paper.get("pub_date") or ""
        if isinstance(pub_date, str):
            for token in pub_date.split("-"):
                if token.isdigit() and len(token) == 4:
                    return token
        return "Unknown"

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return None

    @staticmethod
    def _empty_score_detail(reason: str) -> Dict[str, Any]:
        empty = {"score": 0.0, "reason": reason}
        return {
            "similarity": empty.copy(),
            "norm_impact": empty.copy(),
            "intent_weight": empty.copy(),
            "FICE_like": empty.copy(),
        }

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: LocalRanker._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [LocalRanker._make_json_safe(v) for v in value]
        if isinstance(value, (str, int, float)) or value is None:
            return value
        if isinstance(value, bool):
            return value
        return str(value)


__all__ = ["LocalRanker", "load_local_ranker_config"]

