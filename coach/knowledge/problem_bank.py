"""题库:从外部导入题目(work.md §11.1 #5)。

**为什么不是"让 LLM 现场出题":** DeepTutor / WeSmartFlow 都是纯 LLM 出题 ——
它们没有静态题库,所以"推荐了没有题的知识点"这个状态在它们那里不存在。
但那条路要动的是"题库"这个前提本身,而且**生成的 expected 一旦写错会直接污染判分和
BKT**(错误答案被判对,BKT 就学到假信号)。

这里选择更省也更可控的做法:**从外部题库导入**。数据来源是公开题(LeetCode 等),
`expected` 是**有唯一正确答案的事实**,不需要 LLM 保证正确性。

**它解决的问题(P5 实测):**
- 计划 360 步里 **82%** 落在没有题的知识点上 → 白费
- 没有题的知识点**永远产生不了作答证据** → 根因定位在原理上够不着它

**题目格式(JSON):**
```json
{
  "problems": [
    {
      "id": "lc.70",
      "title": "爬楼梯",
      "source": "leetcode:70",
      "difficulty": 2.0,
      "judge_type": "exact_output",
      "kp_ids": ["algo.dp", "algo.recursion"],
      "test_cases": [{"input": "n=3", "expected": "3"},
                     {"input": "n=5", "expected": "8"}]
    }
  ]
}
```
`test_cases` 的**最后一条**是提交用例(见 `domain/grading.py`)。

用法:
    python -m coach.knowledge.problem_bank --list
    python -m coach.knowledge.problem_bank --from-file my_bank.json
    python -m coach.knowledge.problem_bank --from-url https://example.com/problems.json
    python -m coach.knowledge.problem_bank --coverage      # 看哪些知识点还没有题
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from coach import config
from coach.domain.models import Problem
from coach.knowledge.store import KnowledgeStore

#: 内置题库(随仓库提交,`--build` 会自动灌)
BUNDLED_BANK = Path(__file__).resolve().parent.parent.parent / "data" / "coach" / "problems" / "dsa_seed.json"

_REQUIRED_FIELDS = ("id", "title", "kp_ids", "test_cases")


def parse_bank(payload: Dict) -> List[Dict]:
    """校验并规整题库 JSON。脏数据**跳过而不是抛异常** —— 外部数据不该让整批失败。"""
    if not isinstance(payload, dict):
        raise ValueError("题库根节点必须是对象")
    problems = payload.get("problems")
    if not isinstance(problems, list):
        raise ValueError("题库缺少 problems 数组")

    cleaned = []
    for raw in problems:
        if not isinstance(raw, dict):
            continue
        if any(not raw.get(field) for field in _REQUIRED_FIELDS):
            continue
        if not isinstance(raw["test_cases"], list) or not raw["test_cases"]:
            continue
        if not isinstance(raw["kp_ids"], list):
            continue
        cleaned.append(
            {
                "id": str(raw["id"]),
                "title": str(raw["title"]),
                "source": raw.get("source"),
                "difficulty": raw.get("difficulty"),
                "judge_type": raw.get("judge_type", "exact_output"),
                "test_cases": raw["test_cases"],
                "kp_ids": [str(kp) for kp in raw["kp_ids"]],
            }
        )
    return cleaned


def load_from_file(path) -> List[Dict]:
    return parse_bank(json.loads(Path(path).read_text(encoding="utf-8")))


def fetch_remote(url: str, timeout: int = 30) -> List[Dict]:
    """从 URL 拉题库。用标准库 urllib —— 不为一个导入功能再加 HTTP 依赖。

    **注意**:拉回来的内容是外部输入,只当数据用;下面的 `import_problems`
    会校验结构并跳过脏数据。
    """
    request = urllib.request.Request(url, headers={"User-Agent": "coach-problem-bank/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"拉取题库失败:{url} —— {exc}") from exc
    return parse_bank(payload)


def import_problems(knowledge: KnowledgeStore, problems: List[Dict]) -> Dict:
    """写进库。**跳过引用了不存在知识点的题目** —— 不留悬空引用。

    返回统计:写入几条、跳过几条、以及跳过的原因。
    """
    written = 0
    skipped: List[tuple] = []
    for item in problems:
        missing = [kp for kp in item["kp_ids"] if not knowledge.has_concept(kp)]
        if missing:
            skipped.append((item["id"], f"引用了不存在的知识点 {missing}"))
            continue
        knowledge.upsert_problem(
            Problem(
                id=item["id"],
                title=item["title"],
                source=item.get("source"),
                difficulty=item.get("difficulty"),
                judge_type=item.get("judge_type", "exact_output"),
                test_cases=item["test_cases"],
                kp_ids=item["kp_ids"],
            )
        )
        written += 1
    return {"written": written, "skipped": skipped}


def expected_answer(knowledge: KnowledgeStore, problem_id: str) -> Optional[str]:
    """取某题的「正确提交」—— 判分是拿最后一条用例的 expected 比,所以就是它。"""
    problem = knowledge.get_problem(problem_id)
    if not problem or not problem.test_cases:
        return None
    return str(problem.test_cases[-1].get("expected"))


def coverage(knowledge: KnowledgeStore) -> Dict:
    """哪些知识点有题、哪些没有。没题的点 = 永远产生不了证据 = 根因定位够不着。"""
    concepts = [c.id for c in knowledge.list_concepts()]
    covered = knowledge.kp_ids_with_problems()
    return {
        "total_concepts": len(concepts),
        "covered": len([c for c in concepts if c in covered]),
        "missing": sorted(c for c in concepts if c not in covered),
        "total_problems": len(knowledge.list_problems()),
    }


def seed_from_bank(knowledge: KnowledgeStore, path=BUNDLED_BANK) -> Dict:
    """灌内置题库。文件不存在就跳过(仓库里带了,正常不会缺)。"""
    bank = Path(path)
    if not bank.exists():
        return {"written": 0, "skipped": [], "note": f"题库文件不存在:{bank}"}
    return import_problems(knowledge, load_from_file(bank))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.knowledge.problem_bank", description="从外部题库导入题目")
    parser.add_argument("--db", default=None, help="SQLite 路径,默认 data/coach/coach.db")
    parser.add_argument("--from-file", help="从本地 JSON 文件导入")
    parser.add_argument("--from-url", help="从 URL 拉取并导入")
    parser.add_argument("--seed", action="store_true", help="导入内置题库")
    parser.add_argument("--list", action="store_true", help="列出当前库里的题目")
    parser.add_argument("--coverage", action="store_true", help="看哪些知识点还没有题")
    args = parser.parse_args(argv)

    from coach.knowledge.schema import open_db

    conn = open_db(args.db, check_same_thread=False)
    try:
        knowledge = KnowledgeStore(conn)

        if args.list:
            for problem in knowledge.list_problems():
                print(f"  {problem.id:10} {problem.title:28} kp={problem.kp_ids}")
            print(f"共 {len(knowledge.list_problems())} 道")

        if args.coverage:
            stats = coverage(knowledge)
            print(f"\n题目 {stats['total_problems']} 道,"
                  f"覆盖 {stats['covered']}/{stats['total_concepts']} 个知识点")
            if stats["missing"]:
                print(f"★ 没有题的知识点({len(stats['missing'])} 个)—— "
                      f"它们永远产生不了作答证据,根因定位够不着:")
                for kp in stats["missing"]:
                    print(f"    {kp}")
            else:
                print("★ 每个知识点都有题")

        batches = []
        if args.seed:
            batches.append(("内置题库", seed_from_bank(knowledge)))
        if args.from_file:
            batches.append((args.from_file, import_problems(knowledge, load_from_file(args.from_file))))
        if args.from_url:
            batches.append((args.from_url, import_problems(knowledge, fetch_remote(args.from_url))))

        for label, result in batches:
            print(f"\n{label}:写入 {result['written']} 道")
            for pid, reason in result["skipped"]:
                print(f"  [跳过] {pid}:{reason}")
            if result.get("note"):
                print(f"  {result['note']}")

        if not any([args.list, args.coverage, batches]):
            parser.print_help()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
