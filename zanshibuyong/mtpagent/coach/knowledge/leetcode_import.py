"""从 LeetCode 学习计划导入题目(work.md §11.1 #5 的题库来源之一)。

**为什么要走这条路:** `problem_bank.py` 定义了题库格式,但格式需要人肉准备数据。
LeetCode 的公开学习计划接口(`leetcode.cn/graphql/`,无需登录)能给出成建制的题单。
真正的工作量在**转换**上,而转换里有三处必须显式处理 —— 处理错了会污染判分,
判分错了 BKT 就学到假信号,根因定位跟着歪(这正是 `problem_bank.py` 文档头警告的事):

1. **题单接口不给测试用例。** 要逐题再请求一次 `question`,把 `metaData`(参数名)+
   `exampleTestcases`(输入行)+ `content`(HTML 里的 Output)拼出来。示例 HTML 有两种
   形态(`<pre>` 纯文本 / `<span class="example-io">`),统一按"去掉标签后找 Output 行"处理。
2. **设计题表达不了。** 最小栈 / Trie / LRU / 数据流中位数这类没有"输入→输出"的判分
   语义,直接跳过,不硬塞。
3. **答案不唯一的题不能用严格判分。** 两数之和返回 `[0,1]` 或 `[1,0]` 都对,而
   `grading.grade` 是字符串相等 —— 这类题跳过并进报告,等判分口径支持了再收。

产物是**标准题库 JSON**(与内置 `dsa_seed.json` 同构),**不直接入库** ——
映射对不对需要人过一遍,再走 `problem_bank.import_problems` 这条唯一入库路径。

用法:
    python -m coach.knowledge.leetcode_import --plan top-100-liked
    python -m coach.knowledge.leetcode_import --plan top-100-liked --out my.json
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from coach import config

GRAPHQL_URL = "https://leetcode.cn/graphql/"
_USER_AGENT = "Mozilla/5.0 (compatible; coach-problem-bank/1.0)"

#: 题单最多保留几条用例(判分只用最后一条,前面的是样例)
MAX_TEST_CASES = 3

_PLAN_QUERY = """
query studyPlanV2Detail($slug: String!) {
  studyPlanV2Detail(planSlug: $slug) {
    name
    planSubGroups {
      name
      questions { titleSlug translatedTitle questionFrontendId difficulty }
    }
  }
}
"""

_QUESTION_QUERY = """
query question($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    metaData
    exampleTestcases
    content
  }
}
"""

#: 学习计划的分组名 → 我们图里的知识点 id。
#: **这是整份映射里唯一需要人判断的地方** —— 组名是 LeetCode 的,概念是我们的,
#: 两者不是一一对应。默认给整组挂同一个知识点,个别题在 `SLUG_TO_KP` 里覆盖。
GROUP_TO_KP: Dict[str, List[str]] = {
    "哈希": ["ds.hash_table"],
    "双指针": ["algo.two_pointers"],
    "滑动窗口": ["algo.sliding_window"],
    "子串": ["algo.sliding_window"],
    "普通数组": ["ds.array"],
    "矩阵": ["ds.array"],
    "链表": ["ds.linked_list"],
    "二叉树": ["algo.dfs", "algo.recursion"],
    "图论": ["algo.bfs", "algo.dfs"],
    "回溯": ["algo.backtracking", "algo.recursion"],
    "二分查找": ["algo.binary_search"],
    "栈": ["ds.stack"],
    "堆": [],           # 图里没有"堆"这个概念
    "贪心算法": ["algo.greedy"],
    "动态规划": ["algo.dp"],
    "多维动态规划": ["algo.dp"],
    "技巧": [],          # "技巧"不是知识点,个别题在下面单独挂
}

#: 逐题覆盖:分组默认映射不准、或所在组没有映射时,单独指。
SLUG_TO_KP: Dict[str, List[str]] = {
    # 堆组(组映射为空)。★ 必须带上 quicksort:快速选择就是快排的 partition 那一步,
    #   而且内置题库里 lc.215 是 quicksort 唯一的题目来源,漏了它这个知识点就没题可做。
    "kth-largest-element-in-an-array": ["algo.quicksort", "algo.divide_conquer", "algo.sorting"],
    "top-k-frequent-elements": ["ds.hash_table", "algo.sorting"],
    # 技巧组(组映射为空)
    "sort-colors": ["algo.sorting", "algo.two_pointers"],
    "find-the-duplicate-number": ["algo.binary_search", "algo.two_pointers"],
    # 图论组里不是 BFS/DFS 的那些
    "course-schedule": ["algo.topological_sort", "algo.dfs"],
    # 链表组里其实是排序算法的
    "sort-list": ["algo.mergesort", "ds.linked_list"],
    "merge-k-sorted-lists": ["algo.mergesort", "ds.linked_list"],
    "merge-two-sorted-lists": ["ds.linked_list", "algo.recursion"],
    # 二分查找组里是别的算法的
    "median-of-two-sorted-arrays": ["algo.binary_search", "algo.divide_conquer"],
}

#: 答案不唯一:严格字符串判分会误判,不能按 exact_output 收。
#: 判据是"同一份正确答案可以有多种合法输出"(顺序无关 / 多解)。
MULTI_ANSWER_SLUGS = frozenset(
    {
        "two-sum",  # 题目明说可以任意顺序
        "group-anagrams",
        "3sum",
        "permutations",
        "subsets",
        "letter-combinations-of-a-phone-number",
        "combination-sum",
        "palindrome-partitioning",
        "n-queens",
        "top-k-frequent-elements",
        "merge-intervals",
        "longest-palindromic-substring",  # 最长回文子串可能不唯一
    }
)

#: 原地修改题:LeetCode 的 Output 描述的是**入参被改完后的状态**,
#: 我们的判分口径拿它当答案字符串,能对上但要靠学生按同样格式提交 —— 记进报告备查。
IN_PLACE_SLUGS = frozenset(
    {
        "move-zeroes",
        "rotate-array",
        "rotate-image",
        "set-matrix-zeroes",
        "sort-colors",
        "next-permutation",
        "merge-intervals",
        "flatten-binary-tree-to-linked-list",
        "remove-nth-node-from-end-of-list",
        "swap-nodes-in-pairs",
        "reverse-linked-list",
        "reverse-nodes-in-k-group",
    }
)

_DIFFICULTY = {"EASY": 1.5, "MEDIUM": 3.0, "HARD": 4.5}

_OUTPUT_LINE = re.compile(r"^(?:Output|输出)\s*[:：]\s*(.*)$")

#: 空集合入参,如 `head=[]` / `s=""`。LeetCode 常把这种边界样例放最后一个,
#: 而判分**取最后一条当提交用例** —— 拿空输入当提交用例没有任何区分度,
#: 所以要把它挤掉(仅当还有别的用例可留)。
_DEGENERATE_INPUT = re.compile(r"=\s*(?:\[\]|\"\"|''|\{\})\s*(?:,|$)")


class LeetCodeError(RuntimeError):
    """抓取或解析失败。"""


def _post(query: str, variables: Dict, referer: str, timeout: int = 30) -> Dict:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            "Referer": referer,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise LeetCodeError(f"请求失败:{referer} —— {exc}") from exc
    if payload.get("errors"):
        raise LeetCodeError(f"接口返回错误:{payload['errors']}")
    return payload.get("data") or {}


def fetch_plan(plan_slug: str) -> List[Tuple[str, Dict]]:
    """拉学习计划的题单,返回 [(分组名, 题目 dict), ...]。"""
    data = _post(
        _PLAN_QUERY,
        {"slug": plan_slug},
        referer=f"https://leetcode.cn/studyplan/{plan_slug}/",
    )
    plan = data.get("studyPlanV2Detail")
    if not plan:
        raise LeetCodeError(f"题单不存在:{plan_slug}")

    listing: List[Tuple[str, Dict]] = []
    for group in plan.get("planSubGroups") or []:
        for question in group.get("questions") or []:
            listing.append((group.get("name") or "", question))
    return listing


def fetch_question(title_slug: str) -> Dict:
    """拉单题的 metaData / exampleTestcases / content。"""
    data = _post(
        _QUESTION_QUERY,
        {"titleSlug": title_slug},
        referer=f"https://leetcode.cn/problems/{title_slug}/",
    )
    question = data.get("question")
    if not question:
        raise LeetCodeError(f"题目不存在:{title_slug}")
    return question


def extract_outputs(content_html: str) -> List[str]:
    """从题目 HTML 里抠出所有 Output。

    不按标签结构解析 —— 示例 HTML 至少两种写法(`<pre>` 里是纯文本,
    `<p>` 里 Output 值被 `<span class="example-io">` 包着)。统一"去标签 + 找 Output 行"
    对两种都成立,也不用为新增格式再改一次。

    去标签前**先把块级标签换成换行** —— 否则相邻的 `<p>...</p><p>...</p>`
    会被拼成一行,Output 的值和后面那句 Explanation 粘在一起。
    """
    text = re.sub(r"<br\s*/?>", "\n", content_html or "")
    text = re.sub(r"</(?:p|pre|div|li|h[1-6]|tr)>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    outputs = []
    for line in text.splitlines():
        match = _OUTPUT_LINE.match(line.strip())
        if match:
            outputs.append(match.group(1).strip())
    return outputs


def _param_names(meta_data: Optional[str]) -> List[str]:
    if not meta_data:
        return []
    try:
        parsed = json.loads(meta_data)
    except json.JSONDecodeError:
        return []
    return [p.get("name", "") for p in parsed.get("params", []) if p.get("name")]


def build_test_cases(question: Dict) -> List[Dict[str, str]]:
    """把 metaData + exampleTestcases + content 拼成我们的 test_cases。

    返回空列表表示"这题拼不出可判分的用例"(设计题等),由调用方决定跳过。
    用例条数取输入/输出的较小值 —— 两边数量对不齐时宁可少收,不猜配对。
    """
    params = _param_names(question.get("metaData"))
    if not params:
        return []
    raw_lines = [line for line in (question.get("exampleTestcases") or "").splitlines() if line.strip()]
    outputs = extract_outputs(question.get("content") or "")
    if not raw_lines or not outputs:
        return []

    cases = []
    width = len(params)
    for index, chunk in enumerate(raw_lines[i : i + width] for i in range(0, len(raw_lines), width)):
        if index >= len(outputs):
            break
        if len(chunk) != width:
            break
        given = ", ".join(f"{name}={value}" for name, value in zip(params, chunk))
        cases.append({"input": given, "expected": outputs[index]})
        if len(cases) >= MAX_TEST_CASES:
            break

    substantial = [case for case in cases if not _DEGENERATE_INPUT.search(case["input"])]
    cases = substantial or cases
    # LeetCode 的示例按「典型 → 边界」排,而判分取**最后一条**当提交用例。
    # 把第一条(最典型的)转到最后,免得提交用例落在 `root=[1]` 这种没什么可判的边界样例上。
    if len(cases) > 1:
        cases = cases[1:] + cases[:1]
    return cases


def kp_ids_for(group_name: str, title_slug: str) -> List[str]:
    """题目该挂哪些知识点:逐题覆盖优先,否则用分组默认。"""
    if title_slug in SLUG_TO_KP:
        return list(SLUG_TO_KP[title_slug])
    return list(GROUP_TO_KP.get(group_name, []))


def to_bank(
    plan_slug: str,
    fetch_plan: Callable[[str], List[Tuple[str, Dict]]] = fetch_plan,
    fetch_question: Callable[[str], Dict] = fetch_question,
    sleep: float = 0.0,
) -> Tuple[Dict, Dict]:
    """抓取并转换整份题单,返回 (题库 dict, 报告 dict)。

    **不写库** —— 回一份标准题库 JSON 让人先过一眼映射。
    """
    listing = fetch_plan(plan_slug)
    problems: List[Dict] = []
    skipped: Dict[str, List[Dict]] = {"design": [], "multi_answer": [], "unmapped": [], "parse_failed": []}

    for group_name, question in listing:
        title_slug = question.get("titleSlug")
        entry = {
            "slug": title_slug,
            "no": question.get("questionFrontendId"),
            "title": question.get("translatedTitle") or title_slug,
            "group": group_name,
        }

        kp_ids = kp_ids_for(group_name, title_slug)
        if not kp_ids:
            skipped["unmapped"].append(entry)
            continue
        if title_slug in MULTI_ANSWER_SLUGS:
            skipped["multi_answer"].append(entry)
            continue

        if sleep:
            time.sleep(sleep)
        try:
            detail = fetch_question(title_slug)
        except LeetCodeError as exc:
            skipped["parse_failed"].append({**entry, "reason": str(exc)})
            continue

        test_cases = build_test_cases(detail)
        if not test_cases:
            skipped["design"].append(entry)
            continue

        problems.append(
            {
                "id": f"lc.{question.get('questionFrontendId')}",
                "title": question.get("translatedTitle") or title_slug,
                "source": f"leetcode:{question.get('questionFrontendId')}",
                "difficulty": _DIFFICULTY.get((question.get("difficulty") or "").upper()),
                "judge_type": "exact_output",
                "kp_ids": kp_ids,
                "test_cases": test_cases,
            }
        )

    bank = {
        "name": f"LeetCode 学习计划 —— {plan_slug}",
        "note": (
            "由 coach/knowledge/leetcode_import.py 从 leetcode.cn 公开接口生成。"
            "test_cases 是题面示例(最后一条当提交用例),不是官方判题数据。"
        ),
        "problems": problems,
    }
    # 原地修改题单列出来备查(仍会导入,只是判分口径要留意)
    by_slug = {question.get("titleSlug"): question for _, question in listing}
    report = {
        "plan": plan_slug,
        "total_in_plan": len(listing),
        "imported": len(problems),
        "skipped": skipped,
        "in_place": [
            {"slug": slug, "title": by_slug[slug].get("translatedTitle")}
            for slug in sorted(IN_PLACE_SLUGS & set(by_slug))
        ],
    }
    return bank, report


def write_bank(bank: Dict, report: Dict, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {**bank, "import_report": report}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def format_report(report: Dict) -> str:
    lines = [
        f"题单 {report['plan']}:共 {report['total_in_plan']} 题,可导入 {report['imported']} 题",
    ]
    labels = {
        "unmapped": "图里没有对应知识点",
        "multi_answer": "答案不唯一(严格判分会误判)",
        "design": "设计题(无可判分输出)",
        "parse_failed": "解析失败",
    }
    for key, label in labels.items():
        items = report["skipped"].get(key) or []
        if not items:
            continue
        lines.append(f"\n跳过 [{label}] {len(items)} 题:")
        for item in items:
            extra = f" —— {item['reason']}" if item.get("reason") else ""
            lines.append(f"    {item.get('no','?'):>4} {item.get('title','?')} ({item.get('slug')}){extra}")
    if report.get("in_place"):
        lines.append(f"\n注意:其中 {len(report['in_place'])} 题是原地修改题,Output 是入参被改完的状态")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coach.knowledge.leetcode_import",
        description="从 LeetCode 学习计划生成标准题库 JSON(不写库)",
    )
    parser.add_argument("--plan", default="top-100-liked", help="学习计划的 slug,默认 top-100-liked")
    parser.add_argument("--out", default=None, help="输出 JSON 路径,默认 data/coach/problems/<plan>.json")
    parser.add_argument("--sleep", type=float, default=0.25, help="逐题请求的间隔秒数(别打疼对方)")
    args = parser.parse_args(argv)

    out = Path(args.out) if args.out else config.DATA_DIR / "problems" / f"{args.plan}.json"
    try:
        bank, report = to_bank(args.plan, sleep=args.sleep)
    except LeetCodeError as exc:
        print(f"抓取失败:{exc}", file=sys.stderr)
        return 1

    path = write_bank(bank, report, out)
    print(format_report(report))
    print(f"\n已写入 {path}")
    print("下一步:人工过一眼 kp_ids 映射,再跑")
    print(f"  python -m coach.knowledge.problem_bank --from-file {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
