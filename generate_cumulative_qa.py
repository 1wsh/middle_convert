#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# =========================
# 默认路径 / 常量
# =========================
DEFAULT_INPUT_JSON = "/mmu_mllm_hdd_2/wangshihan/code/Streaming_Proactive/datasets/CG-Bench/cgbench.json"
DEFAULT_FRAME_ROOT = "/mmu_mllm_hdd_2/wangshihan/code/Streaming_Proactive/datasets/CG-Bench/extract_frames/"
DEFAULT_OUTPUT_DIR = "/mmu_mllm_hdd_2/wangshihan/code/Streaming_Proactive/datasets/CG-Bench/cumulative_qa/"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen3.5-122B-A10B"

# 允许的累积演化机制(刻意排除纯计数,要求多样)。
PATTERNS = [
    "running_superlative", "latest_state", "process_chain", "causal_accumulation",
    "relation_update", "hypothesis_revision", "attribute_composition", "comparative_evolution",
]

SYSTEM_PROMPT = """You are a meticulous annotator building training data for a STREAMING video
assistant that watches a video 1 fps and must proactively answer — and RE-answer — questions
as new visual evidence arrives.

THINKING BUDGET: Keep your internal reasoning SHORT — at most a few sentences. Do NOT enumerate
or describe every frame/clip one by one. Think briefly, then output the JSON. Spend your tokens
on the JSON answer, not on the reasoning.

You will be shown frames sampled from ONE video. Each frame is preceded by a text marker
[t=<sec>s] giving its exact second in the video. Use those seconds as your timestamps.

Your job: design 1-3 CUMULATIVE-EVOLVING question-answer pairs for this video.

A cumulative-evolving question is one whose correct answer CHANGES MULTIPLE TIMES as the video
progresses, where each new answer builds on the previous one. Do NOT use plain counting.
Pick the mechanism that best fits the video content (variety across pairs is encouraged):
  - running_superlative: answer = the most [big/fast/high/expensive/...] instance seen so far;
    overwritten only when a more extreme one appears.
  - latest_state: answer = an object's CURRENT state; updates on each change; may flip back
    and forth (non-monotonic), e.g. a door open/closed, who holds the ball, traffic-light color.
  - process_chain: answer = the ordered list of steps completed so far; append on each new step.
  - causal_accumulation: answer = the list of reasons/causes for some outcome so far; append.
  - relation_update: answer = current owner of an object / relation between two entities;
    overwritten when it transfers.
  - hypothesis_revision: answer = the most likely judgement given current clues (what/where/who);
    REVISED when a new clue overturns it.
  - attribute_composition: answer = the accumulated description of one target's known attributes;
    extend when a new attribute becomes visible.
  - comparative_evolution: answer = which of A vs B is currently ahead/winning; overwritten on
    a lead change.

HARD REQUIREMENTS
1. Each question MUST have at least 2 (ideally 3-5) distinct moments where the answer CHANGES.
   If the video cannot support that, do not emit the question.
2. Each change maps to an EXACT second = the earliest frame second at which the supporting
   evidence is visible. Timestamps must be strictly increasing.
3. Every answer change must be grounded in VISIBLE evidence — no speculation. State, for each
   change, what changed versus the previous answer and why.
4. The question text MUST NOT reveal the answer, the total, or the final conclusion. Phrase it
   so the model cannot be sure without watching.
5. Provide 4 single-choice options (A-D) for the FINAL answer; exactly one is the correct final
   value, the others are strong distractors.
6. PATTERN DIVERSITY: when you emit more than one QA pair for a video, each pair MUST use a
   DIFFERENT mechanism. Do NOT make every pair `latest_state`. Actively prefer the
   harder/under-used mechanisms when the video supports them — especially `hypothesis_revision`,
   `comparative_evolution`, `causal_accumulation`, `running_superlative`.
7. ADJACENT ANSWERS MUST DIFFER: consecutive timeline items must have DIFFERENT answers. Never
   emit two adjacent points with the same answer (e.g. "Counter" then "Counter"). Only record a
   point when the answer actually changes from the immediately previous one. (A value may recur
   later if something different happened in between, e.g. Counter -> POS -> Counter is fine.)
8. STABLE, LOCALISABLE EVIDENCE: every timeline change must be supported by evidence that stays
   visible for SEVERAL seconds — a persistent state, scene, object, or on-screen text. Do NOT
   anchor a change on a fleeting instantaneous action (e.g. a single sprinkle of salt, one quick
   gesture) that is visible in only a frame or two — such moments cannot be located reliably at
   1 fps. If a change can only be judged from a momentary action, do not make it a timeline point.
9. IDENTITY/ATTRIBUTE CLARITY: when the answer is a specific person/object identity or attribute,
   pick the timestamp where that identity/attribute is CLEARLY visible (front-facing, close, well
   lit) — not a blurry, distant, or back-facing frame.

OUTPUT — STRICT JSON ONLY (no markdown, no prose):
{
  "qa_pairs": [
    {
      "pattern": "<one of the mechanisms above>",
      "question": "...(English, no answer leaked)",
      "timeline": [
        {"time_sec": 18, "answer": "...",
         "evidence": "one sentence: the visible evidence at this moment that changed the answer"},
        {"time_sec": 44, "answer": "...", "evidence": "..."}
      ],
      "final_answer": "...(== last timeline answer)",
      "choices": ["...correct final value...", "...", "...", "..."],
      "right_answer": "A"
    }
  ]
}
(The "evidence" field is used only to verify timestamps and will not be kept in the final data,
but you MUST still provide it for every timeline item.)

SELF-CHECK BEFORE OUTPUT
- pattern is NOT plain counting; pairs use varied mechanisms.
- if >1 pair, the pairs use DIFFERENT mechanisms (not all latest_state).
- no two ADJACENT timeline items share the same answer.
- timeline has >=2 items, strictly increasing time_sec, each answer derivable from its evidence.
- latest_state / comparative_evolution show possible flips, not a monotonic accumulation.
- question text leaks no answer/total; final_answer == last timeline answer, is in choices,
  and right_answer letter matches it.
Output JSON only."""


# 第二阶段:时间戳细化。对某个变化点,只给它附近【逐秒密集帧】,让模型精确定位证据
# 最早出现的那一秒(秒级精度),并确认该变化是否真实存在。
REFINE_SYSTEM_PROMPT = """You are refining ONE answer-change moment of a streaming-video QA item.

THINKING BUDGET: Reason in one or two sentences at most. Do NOT describe every frame. Then output
the JSON.

You are given: the question, the answer BEFORE this change, the answer AFTER this change, and a
short burst of CONSECUTIVE per-second frames around the suspected change. Each frame is tagged
[t=<sec>s] with its exact second.

Find the EARLIEST second at which the visible evidence for the AFTER answer first appears in this
burst. If the evidence is clearly visible, return that exact second. If the change does NOT
actually happen anywhere in this burst (the AFTER evidence is never visible), say so.

OUTPUT — STRICT JSON ONLY:
{"found": true, "time_sec": <int second from the tags>, "confidence": "high" | "low", "evidence": "one sentence on the exact visible cue"}
or
{"found": false, "reason": "why the AFTER evidence is not visible in this burst"}
Set confidence="high" only when the AFTER evidence is clearly and unambiguously visible at that
second; use "low" if you are inferring it, it is partly occluded/blurry, or the exact second is
uncertain. Output JSON only, no prose."""


# =========================
# 基础工具(内联,不依赖外部文件)
# =========================
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def image_to_data_url(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def strip_code_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|xml|html|text)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def list_frame_files(video_dir: str) -> List[str]:
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        files.extend(Path(video_dir).glob(ext))

    def frame_number(p: Path) -> int:
        m = re.search(r"frame_(\d+)", p.name)
        if m:
            return int(m.group(1))
        nums = re.findall(r"\d+", p.stem)
        return int(nums[-1]) if nums else 0

    return [str(p) for p in sorted(files, key=frame_number)]


def locate_video_dir(
    frame_root: str,
    video_uid: str,
    video_dir_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    root = Path(frame_root)
    if video_dir_map and video_uid in video_dir_map:
        mapped = video_dir_map[video_uid]
        p = Path(mapped)
        if not p.is_absolute():
            p = root / mapped
        if p.exists() and p.is_dir():
            return str(p)

    for c in [root / video_uid, root / f"_{video_uid}"]:
        if c.exists() and c.is_dir():
            return str(c)

    if root.exists() and root.is_dir():
        for d in root.iterdir():
            if d.is_dir() and video_uid in d.name:
                return str(d)
    return None


def call_model(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.4,
    max_tokens: int = 16384,
    retry: int = 3,
    sleep_seconds: float = 2.0,
) -> str:
    last_err = None
    for i in range(retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            print(f"[WARN] model call failed, retry {i + 1}/{retry}: {e}")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Model call failed after {retry} retries: {last_err}")


# =========================
# 采样 / 构造 / 解析 / 校验
# =========================
def sample_frames_by_second(
    frame_files: List[str],
    duration: Optional[int],
    sample_every_sec: int,
    max_frames: int,
) -> List[Tuple[int, str]]:
    """返回 [(second, path)],按秒采样并 cap 到 max_frames(再均匀降采样)。"""
    sec_map: Dict[int, str] = {}
    for p in frame_files:
        m = re.search(r"frame_(\d+)", Path(p).name)
        if m:
            sec_map[int(m.group(1))] = p
    if not sec_map:
        return []
    max_sec = max(sec_map.keys())
    if duration is not None:
        max_sec = min(max_sec, int(duration))

    picked = [(s, sec_map[s]) for s in range(1, max_sec + 1)
              if s % sample_every_sec == 0 and s in sec_map]
    if not picked and 1 in sec_map:
        picked = [(1, sec_map[1])]
    # 第一帧总是带上,便于早期定位。
    if picked and picked[0][0] != 1 and 1 in sec_map:
        picked = [(1, sec_map[1])] + picked

    if len(picked) > max_frames:
        idxs = [int(i * len(picked) / max_frames) for i in range(max_frames)]
        picked = [picked[i] for i in idxs]
    return picked


def build_user_content(frames: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
    """构造多模态 user content:每帧前加 [t=Xs] 文本标记。"""
    content: List[Dict[str, Any]] = [{
        "type": "text",
        "text": (f"This video has {frames[-1][0]} seconds of content. "
                 f"{len(frames)} sampled frames follow, each tagged with its exact second. "
                 "Design the cumulative-evolving QA pairs now."),
    }]
    for sec, path in frames:
        content.append({"type": "text", "text": f"[t={sec}s]"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    return content


def parse_qa_json(text: str) -> Optional[Dict[str, Any]]:
    """从 teacher 输出里稳健抽取 JSON。"""
    text = strip_code_fence(text or "")
    # 去掉可能的外层 <think>...</think>。
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S | re.I)
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    blob = text[i:j + 1]
    try:
        return json.loads(blob)
    except Exception:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
        except Exception:
            return None


def _norm_ans(s: Any) -> str:
    """答案归一化:去首尾空白、去成对引号、统一小写、压空白、去尾部标点。用于宽松比较。"""
    t = str(s or "").strip().lower()
    t = t.strip('"\'“”‘’ ')
    t = re.sub(r"\s+", " ", t)
    t = t.rstrip(".。!?!?,,;; ")
    return t.strip()


# final_answer 与 timeline 末项「不要求文字匹配」的 pattern:
#   - 追加型(process_chain/causal_accumulation/attribute_composition):final 是累积全量,末项是单步;
#   - 总结型(comparative_evolution/hypothesis_revision):final 是对演化的总结/最终判断,
#     末项是"当前这一刻"的描述,用词本就不同。
# 这些 pattern 只校验 final 能对到某个选项,不比对 final↔末项。
SUMMARY_PATTERNS = {
    "process_chain", "causal_accumulation", "attribute_composition",
    "comparative_evolution", "hypothesis_revision",
}
# 仍要求 final≈末项(终值就是当前态)的 pattern:latest_state / running_superlative / relation_update。


def _tok_set(s: str):
    """归一化后的词集合(用于模糊相似度)。"""
    return set(re.findall(r"[a-z0-9]+", _norm_ans(s)))


def _similar(a: str, b: str) -> float:
    """两字符串的 Jaccard 词重叠度 [0,1]。"""
    sa, sb = _tok_set(a), _tok_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _answer_match(a: str, b: str) -> bool:
    """宽松答案相等:归一化全等 / 互为子串 / 高词重叠。用于吸收措辞、标点、前缀差异。"""
    na, nb = _norm_ans(a), _norm_ans(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return _similar(a, b) >= 0.6


def _best_choice_idx(final: str, choices: List[str]) -> int:
    """返回与 final 最匹配的选项下标:优先精确/包含,否则取最高相似度。"""
    fn = _norm_ans(final)
    # 1) 归一化全等
    for i, c in enumerate(choices):
        if _norm_ans(c) == fn:
            return i
    # 2) 互为子串
    for i, c in enumerate(choices):
        nc = _norm_ans(c)
        if nc and (nc in fn or fn in nc):
            return i
    # 3) 最高相似度
    sims = [(_similar(final, c), i) for i, c in enumerate(choices)]
    sims.sort(reverse=True)
    return sims[0][1] if sims else 0


def fix_increasing_time(qa: Dict[str, Any]) -> Dict[str, Any]:
    """把 timeline 的 time_sec 修正为严格递增:逆序/相等的点上移到 prev+1,而非整条作废。"""
    tl = qa.get("timeline")
    if not isinstance(tl, list) or not tl:
        return qa
    fixed = []
    last_t = -1
    for ev in tl:
        ev = dict(ev)
        try:
            t = int(ev.get("time_sec"))
        except Exception:
            t = last_t + 1
        if t <= last_t:
            t = last_t + 1
        ev["time_sec"] = t
        last_t = t
        fixed.append(ev)
    qa = dict(qa)
    qa["timeline"] = fixed
    return qa


def dedup_adjacent_timeline(qa: Dict[str, Any]) -> Dict[str, Any]:
    """合并 timeline 中【相邻】且答案相同的点(保留较早的时间戳),不动中间隔着不同答案的重复。
    例:Counter->Counter 合并为一个 Counter;Counter->POS->Counter 全保留。"""
    tl = qa.get("timeline")
    if not isinstance(tl, list) or len(tl) < 2:
        return qa
    merged: List[Dict[str, Any]] = []
    for ev in tl:
        if merged and _norm_ans(ev.get("answer", "")) == _norm_ans(merged[-1].get("answer", "")):
            # 与上一个相邻点答案相同 -> 跳过(保留更早的那个点)。
            continue
        merged.append(ev)
    if len(merged) != len(tl):
        qa = dict(qa)
        qa["timeline"] = merged
    return qa


def slim_qa_for_output(qa: Dict[str, Any]) -> Dict[str, Any]:
    """最终落盘清洗:只保留 pattern/question/choices/right_answer/final_answer 和
    timeline(每项仅 time_sec + answer)。去掉 evidence/change/tracked_target/refine_status 等中间字段。"""
    slim_tl = []
    for ev in qa.get("timeline", []):
        try:
            t = int(ev.get("time_sec"))
        except Exception:
            continue
        item = {"time_sec": t, "answer": str(ev.get("answer", "")).strip()}
        # 保留时间置信度(high/low),供下游对低置信变化点降权或过滤。
        conf = ev.get("time_confidence")
        if conf:
            item["time_confidence"] = conf
        slim_tl.append(item)
    return {
        "pattern": qa.get("pattern"),
        "question": qa.get("question", ""),
        "choices": qa.get("choices", []),
        "right_answer": str(qa.get("right_answer", "")).strip().upper(),
        "final_answer": str(qa.get("final_answer", "")).strip(),
        "timeline": slim_tl,
    }


def validate_qa(qa: Dict[str, Any]) -> Tuple[bool, str]:
    """逐条自检一个 qa_pair,返回 (是否合格, 原因)。
    会就地修正 qa 的 right_answer/final_answer 以对齐 choices(宽松匹配),不轻易因措辞差异作废。"""
    if not isinstance(qa, dict):
        return False, "not_dict"
    pattern = qa.get("pattern")
    if pattern not in PATTERNS:
        return False, f"bad_pattern:{pattern}"
    tl = qa.get("timeline")
    if not isinstance(tl, list) or len(tl) < 2:
        return False, "timeline_lt2"
    last_t = -1
    for ev in tl:
        if not isinstance(ev, dict) or "time_sec" not in ev or "answer" not in ev:
            return False, "timeline_item_malformed"
        try:
            t = int(ev["time_sec"])
        except Exception:
            return False, "time_sec_not_int"
        if t <= last_t:
            return False, "time_not_increasing"
        last_t = t
        # evidence 只在细化阶段需要;若已被清洗或缺失,不再强制(终值校验更重要)。
    choices = qa.get("choices")
    if not isinstance(choices, list) or len(choices) != 4:
        return False, "choices_not_4"
    if any(not str(c).strip() for c in choices):
        return False, "empty_choice"
    final = str(qa.get("final_answer", "")).strip()
    if not final:
        return False, "empty_final"

    # final 与 timeline 最后一项的一致性(宽松):
    #   - 总结型(process_chain/comparative_evolution/hypothesis_revision 等):final 是
    #     累积全量或演化总结,与末项用词本就不同 -> 不比对,只需 final 能对到选项。
    #   - 其它型(latest_state/running_superlative/relation_update):final 与末项需宽松匹配。
    if pattern not in SUMMARY_PATTERNS:
        last_ans = tl[-1].get("answer", "")
        if not _answer_match(final, last_ans):
            return False, "final_ne_last_timeline"

    # final 必须能对到某个选项(宽松);自动把 right_answer 对齐到最匹配选项,
    # 并把 final_answer 规整为该选项的原文,保证三者一致。
    idx = _best_choice_idx(final, choices)
    if not _answer_match(final, choices[idx]):
        return False, "final_not_in_choices"
    qa["right_answer"] = chr(65 + idx)
    qa["final_answer"] = str(choices[idx]).strip()

    if not str(qa.get("question", "")).strip():
        return False, "empty_question"
    return True, "ok"


# =========================
# 第二阶段:时间戳细化
# =========================
def _build_sec_map(frame_files: List[str]) -> Dict[int, str]:
    sec_map: Dict[int, str] = {}
    for p in frame_files:
        m = re.search(r"frame_(\d+)", Path(p).name)
        if m:
            sec_map[int(m.group(1))] = p
    return sec_map


def sample_dense_window(
    sec_map: Dict[int, str],
    center_sec: int,
    half_window: int,
    max_sec: int,
    max_frames: int,
) -> List[Tuple[int, str]]:
    """取 [center-half, center+half] 的逐秒帧,cap 到 max_frames(均匀降采样)。"""
    lo = max(1, center_sec - half_window)
    hi = min(max_sec, center_sec + half_window)
    picked = [(s, sec_map[s]) for s in range(lo, hi + 1) if s in sec_map]
    if len(picked) > max_frames:
        idxs = [int(i * len(picked) / max_frames) for i in range(max_frames)]
        picked = [picked[i] for i in idxs]
    return picked


def parse_refine_json(text: str) -> Optional[Dict[str, Any]]:
    text = strip_code_fence(text or "")
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S | re.I)
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", text[i:j + 1]))
        except Exception:
            return None


def refine_one_change(
    client: OpenAI,
    model: str,
    question: str,
    prev_answer: str,
    after_answer: str,
    coarse_sec: int,
    sec_map: Dict[int, str],
    max_sec: int,
    half_window: int,
    refine_max_frames: int,
    temperature: float,
) -> Dict[str, Any]:
    """对单个变化点做密集细化,返回 {found, time_sec?, evidence?, reason?}。"""
    frames = sample_dense_window(sec_map, coarse_sec, half_window, max_sec, refine_max_frames)
    if len(frames) < 2:
        return {"found": False, "reason": "too_few_dense_frames"}

    content: List[Dict[str, Any]] = [{
        "type": "text",
        "text": (f"Question: {question}\n"
                 f"Answer BEFORE this change: {prev_answer or '(none / no answer yet)'}\n"
                 f"Answer AFTER this change: {after_answer}\n"
                 f"Suspected change near t={coarse_sec}s. Per-second frames around it follow."),
    }]
    for sec, path in frames:
        content.append({"type": "text", "text": f"[t={sec}s]"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})

    messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    try:
        raw = call_model(client, model, messages, temperature=temperature, max_tokens=16384)
    except Exception as e:
        return {"found": False, "reason": f"refine_call_failed:{e}"}
    r = parse_refine_json(raw)
    if not isinstance(r, dict):
        return {"found": False, "reason": "refine_parse_failed"}
    # 约束 time_sec 落在窗口内的真实帧上。
    if r.get("found") and "time_sec" in r:
        try:
            t = int(r["time_sec"])
        except Exception:
            return {"found": False, "reason": "refine_time_not_int"}
        valid_secs = [s for s, _ in frames]
        if t not in valid_secs:
            # 吸附到最近的采样秒。
            t = min(valid_secs, key=lambda s: abs(s - t))
        r["time_sec"] = t
    return r


def refine_timeline(
    qa: Dict[str, Any],
    client: OpenAI,
    model: str,
    sec_map: Dict[int, str],
    max_sec: int,
    half_window: int,
    refine_max_frames: int,
    temperature: float,
) -> Dict[str, Any]:
    """对一个 qa 的 timeline 逐变化点细化时间戳。
    细化只用于【修正时间戳】,不改答案、不丢点:found 时用细化秒,未 found 时保留粗时间戳兜底。"""
    tl = qa.get("timeline", [])
    refined: List[Dict[str, Any]] = []
    prev_answer = ""
    for ev in tl:
        after = str(ev.get("answer", "")).strip()
        try:
            coarse = int(ev.get("time_sec"))
        except Exception:
            coarse = 1
        res = refine_one_change(
            client, model, qa.get("question", ""), prev_answer, after,
            coarse, sec_map, max_sec, half_window, refine_max_frames, temperature,
        )
        new_ev = dict(ev)
        new_ev["coarse_time_sec"] = coarse
        if res.get("found") and "time_sec" in res:
            new_ev["time_sec"] = res["time_sec"]
            if str(res.get("evidence", "")).strip():
                new_ev["evidence"] = res["evidence"]
            new_ev["refine_status"] = "refined"
            # 置信度:模型自报 high/low;缺省按 high 处理(命中即可信)。
            conf = str(res.get("confidence", "high")).strip().lower()
            new_ev["time_confidence"] = "low" if conf == "low" else "high"
        else:
            # 细化失败 -> 保留粗时间戳兜底,不丢点(尤其 hypothesis_revision 等推测型,
            # 密集帧里"看不到证据"是常态,丢点会把整条 timeline 砍到 <2)。
            # 粗时间戳来自稀疏采样,精度差,标记为 low 供下游降权/过滤。
            new_ev["time_sec"] = coarse
            new_ev["refine_status"] = f"kept_coarse:{res.get('reason', 'not_found')}"
            new_ev["time_confidence"] = "low"
        refined.append(new_ev)
        prev_answer = after
    # 细化后保证时间戳严格递增(吸附/兜底可能造成相等/逆序)。
    cleaned: List[Dict[str, Any]] = []
    last_t = -1
    for ev in refined:
        try:
            t = int(ev["time_sec"])
        except Exception:
            t = last_t + 1
        if t <= last_t:
            t = last_t + 1
        ev["time_sec"] = t
        last_t = t
        cleaned.append(ev)
    qa = dict(qa)
    qa["timeline"] = cleaned
    return qa


# =========================
# 单视频处理
# =========================
def process_one_video(
    video_uid: str,
    duration: Optional[int],
    client: OpenAI,
    model: str,
    frame_root: str,
    output_dir: str,
    sample_every_sec: int,
    max_frames: int,
    video_dir_map: Optional[Dict[str, str]],
    overwrite: bool,
    temperature: float,
    refine: bool = True,
    refine_half_window: int = 15,
    refine_max_frames: int = 31,
) -> str:
    out_path = os.path.join(output_dir, f"{video_uid}.json")
    if os.path.exists(out_path) and not overwrite:
        return "skip_exist"

    video_dir = locate_video_dir(frame_root, video_uid, video_dir_map)
    if not video_dir:
        save_json({"video_uid": video_uid, "error": "video_dir_not_found", "qa_pairs": []}, out_path)
        return "no_video_dir"

    frame_files = list_frame_files(video_dir)
    frames = sample_frames_by_second(frame_files, duration, sample_every_sec, max_frames)
    if len(frames) < 2:
        save_json({"video_uid": video_uid, "error": "too_few_frames", "qa_pairs": []}, out_path)
        return "too_few_frames"

    # ---- 第一阶段:粗看全片,出题 + 粗略 timeline ----
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(frames)},
    ]
    try:
        raw = call_model(client, model, messages, temperature=temperature, max_tokens=16384)
    except Exception as e:
        save_json({"video_uid": video_uid, "error": f"call_failed:{e}", "qa_pairs": []}, out_path)
        return "call_failed"

    parsed = parse_qa_json(raw)
    if not parsed or not isinstance(parsed.get("qa_pairs"), list):
        save_json({"video_uid": video_uid, "error": "parse_failed", "raw": raw[:4000],
                   "qa_pairs": []}, out_path)
        return "parse_failed"

    coarse_kept, rejected = [], []
    for qa in parsed["qa_pairs"]:
        qa = dedup_adjacent_timeline(qa)   # 合并相邻相同答案
        qa = fix_increasing_time(qa)       # 时间戳逆序自动修正,不整条作废
        ok, why = validate_qa(qa)
        if ok:
            coarse_kept.append(qa)
        else:
            rejected.append({"reason": f"coarse:{why}", "qa": qa})

    # ---- 第二阶段:逐变化点密集细化时间戳(秒级)----
    sec_map = _build_sec_map(frame_files)
    max_sec = max(sec_map.keys()) if sec_map else 0
    if duration is not None:
        max_sec = min(max_sec, int(duration))

    kept = []
    for qa in coarse_kept:
        if refine:
            qa = refine_timeline(
                qa, client, model, sec_map, max_sec,
                refine_half_window, refine_max_frames, temperature,
            )
            qa = dedup_adjacent_timeline(qa)   # 细化吸附后可能再次相邻重复
        # 细化可能丢点 / 改 final_answer,需重新自检。
        ok, why = validate_qa(qa)
        if ok:
            slim = slim_qa_for_output(qa)
            slim["num_updates"] = len(slim["timeline"])
            kept.append(slim)
        else:
            rejected.append({"reason": f"post_refine:{why}", "qa": qa})

    save_json({
        "video_uid": video_uid,
        "duration": duration,
        "num_sampled_frames": len(frames),
        "sample_every_sec": sample_every_sec,
        "refine": refine,
        "refine_half_window": refine_half_window,
        "model": model,
        "qa_pairs": kept,
        "rejected": rejected,
    }, out_path)
    return f"ok:{len(kept)}kept/{len(rejected)}rej"


# =========================
# 主流程
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    ap.add_argument("--frame_root", default=DEFAULT_FRAME_ROOT)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--video_dir_map", default=None)
    ap.add_argument("--max_duration", type=int, default=1800, help="只处理 <= 该秒数的视频")
    ap.add_argument("--sample_every_sec", type=int, default=8, help="第一阶段每隔多少秒采一帧")
    ap.add_argument("--max_frames", type=int, default=120, help="第一阶段送入 teacher 的最大帧数")
    ap.add_argument("--refine", dest="refine", action="store_true", default=True,
                    help="开启第二阶段时间戳细化(默认开)")
    ap.add_argument("--no_refine", dest="refine", action="store_false",
                    help="关闭第二阶段细化(只跑粗 timeline)")
    ap.add_argument("--refine_half_window", type=int, default=15,
                    help="细化时每个变化点取 [t-h, t+h] 的逐秒帧")
    ap.add_argument("--refine_max_frames", type=int, default=31,
                    help="细化单次最大帧数(须 <= 服务端图片上限)")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="冒烟:只处理(切片后)前 N 个视频")
    ap.add_argument("--start_idx", type=int, default=None,
                    help="处理去重排序后视频列表的 [start_idx, end_idx) 区间;也可用环境变量 VID_START")
    ap.add_argument("--end_idx", type=int, default=None,
                    help="见 --start_idx;也可用环境变量 VID_END")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_json(args.input_json)

    # 按 video_uid 去重(每个视频出一次题),保留最短 duration。
    vid_dur: Dict[str, int] = {}
    for it in data:
        d = int(it.get("duration", 0) or 0)
        if d <= args.max_duration:
            uid = it["video_uid"]
            if uid not in vid_dur or d < vid_dur[uid]:
                vid_dur[uid] = d
    videos = sorted(vid_dur.items())   # 稳定排序:同样的输入 -> 同样的全局顺序,分片才可靠
    total_all = len(videos)

    # 范围分片:命令行 --start_idx/--end_idx 优先,其次环境变量 VID_START/VID_END。
    start_idx = args.start_idx
    end_idx = args.end_idx
    if start_idx is None and os.environ.get("VID_START") is not None:
        start_idx = int(os.environ["VID_START"])
    if end_idx is None and os.environ.get("VID_END") is not None:
        end_idx = int(os.environ["VID_END"])
    s = start_idx if start_idx is not None else 0
    e = end_idx if end_idx is not None else total_all
    videos = videos[s:e]
    if args.limit:
        videos = videos[:args.limit]

    video_dir_map = load_json(args.video_dir_map) if args.video_dir_map else None

    print(f"[INFO] videos<= {args.max_duration}s 全量={total_all}  本分片[{s}:{e}]取={len(videos)}  model={args.model}")
    print(f"[INFO] stage1: sample_every_sec={args.sample_every_sec} max_frames={args.max_frames}")
    print(f"[INFO] stage2(refine={args.refine}): half_window={args.refine_half_window} "
          f"refine_max_frames={args.refine_max_frames}")
    print(f"[INFO] workers={args.num_workers} out={args.output_dir}")

    def _worker(item):
        uid, dur = item
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        return uid, process_one_video(
            uid, dur, client, args.model, args.frame_root, args.output_dir,
            args.sample_every_sec, args.max_frames, video_dir_map,
            args.overwrite, args.temperature,
            refine=args.refine, refine_half_window=args.refine_half_window,
            refine_max_frames=args.refine_max_frames,
        )

    done = 0
    stat = Counter()
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(_worker, v): v[0] for v in videos}
        for fut in as_completed(futs):
            uid = futs[fut]
            try:
                _, status = fut.result()
            except Exception as e:
                status = f"fatal:{e}"
            stat[status.split(":")[0]] += 1
            done += 1
            print(f"[PROGRESS] {done}/{len(videos)} {uid} -> {status}")
    print(f"[DONE] status分布: {dict(stat)}")


if __name__ == "__main__":
    main()
