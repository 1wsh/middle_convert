#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_cumulative_qa_split.py
===============================
对【>= 30 分钟的长视频】按 10 分钟切段，逐段生成累积演化 QA。

关键点：不切视频、不重抽帧。抽帧命名 frame_NNNN.jpg 的帧号 == 秒数(1fps)，
所以“切成 10 分钟段”只是【对帧列表按秒区间切片 + 段内秒号归零(方案A)】。

切段规则(与 split_long_videos.py 对齐)：
  - 只处理帧目录里【最大帧号 >= --min_frames(默认1800，即>=30min)】的视频。
    真实时长以【帧目录最大帧号】为准，不信 cgbench.json 的 duration。
  - 段 i 覆盖全局帧号 [i*600+1, (i+1)*600]；本地秒 = 全局秒 - i*600，范围 1..600。
  - 尾段到实际最后一帧；若尾段不足 60s，则并入前一段。
  - 命名用分钟区间：<video_uid>_0_10 / _10_20 / ... / _20_22(尾段用 floor(maxframe/60))。

每段是一个自洽的“10 分钟短视频”：模型只看本段帧、只见本地秒(1..600)，
完全不知道前段内容 —— 累积演化只在段内发生。产出 timeline 的 time_sec 落在 1..600。

输出：cumulative_qa_split/<video_uid>_<a>_<b>.json，一段一个文件。
      下游 clean_cumulative_qa.py 聚合时 qid 自然是 "<video_uid>_<a>_<b>__q<idx>"。

复用 generate_cumulative_qa.py 的全部提示词/调用/校验/细化逻辑，本文件只加“分段+归零”这层。

用法：
  python scripts/generate_cumulative_qa_split.py \
      --frame_root extract_frames/ \
      --output_dir cumulative_qa_split/ \
      --base_url http://127.0.0.1:8000/v1 \
      --model Qwen3.5-122B-A10B \
      --num_workers 4
"""
import argparse
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# 同目录复用(运行 `python scripts/xxx.py` 时 scripts/ 在 sys.path[0])。
from generate_cumulative_qa import (
    DEFAULT_FRAME_ROOT,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    save_json,
    list_frame_files,
    locate_video_dir,
    call_model,
    build_user_content,
    parse_qa_json,
    dedup_adjacent_timeline,
    fix_increasing_time,
    validate_qa,
    refine_timeline,
    slim_qa_for_output,
)

DEFAULT_OUTPUT_DIR = "/mmu_mllm_hdd_2/wangshihan/code/Streaming_Proactive/datasets/CG-Bench/cumulative_qa_split/"

CLIP_SECONDS = 600  # 每段 10 分钟
MIN_FRAMES = 1800   # 只处理最大帧号 >= 该值(即时长>=30min)的视频
TAIL_MERGE_SEC = 60  # 尾段不足该秒数则并入前一段


# =========================
# 分段 / 段内归零
# =========================
def frame_number(name: str) -> Optional[int]:
    m = re.search(r"frame_(\d+)", name)
    return int(m.group(1)) if m else None


def max_frame_number(frame_files: List[str]) -> int:
    """帧目录里的最大帧号(== 视频真实秒数)。空返回 0。"""
    nums = [frame_number(Path(p).name) for p in frame_files]
    nums = [n for n in nums if n is not None]
    return max(nums) if nums else 0


def plan_segments(max_frame: int) -> List[List[int]]:
    """按最大帧号切段。返回 [[frame_start, frame_end, start_min, end_min], ...]。
    frame_start/frame_end 是【全局帧号(含两端)】；start_min/end_min 仅用于命名。"""
    n_seg = max(1, math.ceil(max_frame / CLIP_SECONDS))
    segs: List[List[int]] = []
    for i in range(n_seg):
        f_start = i * CLIP_SECONDS + 1
        f_end = min((i + 1) * CLIP_SECONDS, max_frame)
        start_min = i * 10
        end_min = (i + 1) * 10 if i < n_seg - 1 else max_frame // 60
        segs.append([f_start, f_end, start_min, end_min])

    # 尾段不足 60s(起止分钟相同) -> 并入前一段。
    if len(segs) >= 2 and (segs[-1][1] - segs[-1][0] + 1) < TAIL_MERGE_SEC:
        last = segs.pop()
        segs[-1][1] = last[1]      # 帧尾延伸到真实结尾
        segs[-1][3] = last[3]      # 结束分钟取真实值
    return segs


def build_local_sec_map(
    frame_files: List[str], frame_start: int, frame_end: int
) -> Dict[int, str]:
    """取全局帧号 ∈ [frame_start, frame_end] 的帧，秒号归零：
    local_sec = global_sec - (frame_start - 1)，使段内秒从 1 起(与原脚本 frame_0001 起一致)。"""
    offset = frame_start - 1
    local_map: Dict[int, str] = {}
    for p in frame_files:
        g = frame_number(Path(p).name)
        if g is not None and frame_start <= g <= frame_end:
            local_map[g - offset] = p
    return local_map


def sample_from_sec_map(
    sec_map: Dict[int, str], sample_every_sec: int, max_frames: int
) -> List[Tuple[int, str]]:
    """按秒采样并 cap 到 max_frames(均匀降采样)。输入/输出都用【本地秒】。
    逻辑与 generate_cumulative_qa.sample_frames_by_second 一致，但直接吃 sec_map。"""
    if not sec_map:
        return []
    max_sec = max(sec_map.keys())
    picked = [(s, sec_map[s]) for s in range(1, max_sec + 1)
              if s % sample_every_sec == 0 and s in sec_map]
    if not picked and 1 in sec_map:
        picked = [(1, sec_map[1])]
    # 第一帧总是带上，便于早期定位。
    if picked and picked[0][0] != 1 and 1 in sec_map:
        picked = [(1, sec_map[1])] + picked

    if len(picked) > max_frames:
        idxs = [int(i * len(picked) / max_frames) for i in range(max_frames)]
        picked = [picked[i] for i in idxs]
    return picked


# =========================
# 单段处理
# =========================
def process_one_segment(
    video_uid: str,
    seg: List[int],
    frame_files: List[str],
    client: OpenAI,
    model: str,
    output_dir: str,
    sample_every_sec: int,
    max_frames: int,
    overwrite: bool,
    temperature: float,
    refine: bool,
    refine_half_window: int,
    refine_max_frames: int,
) -> str:
    frame_start, frame_end, start_min, end_min = seg
    seg_label = f"{start_min}_{end_min}"
    out_path = os.path.join(output_dir, f"{video_uid}_{seg_label}.json")
    if os.path.exists(out_path) and not overwrite:
        return "skip_exist"

    # 段内归零后的 sec_map(本地秒 -> 帧路径)，下游全部消费本地秒。
    sec_map = build_local_sec_map(frame_files, frame_start, frame_end)
    max_sec = max(sec_map.keys()) if sec_map else 0  # 段长(本地)
    frames = sample_from_sec_map(sec_map, sample_every_sec, max_frames)
    if len(frames) < 2:
        save_json({"video_uid": video_uid, "segment": seg_label,
                   "error": "too_few_frames", "qa_pairs": []}, out_path)
        return "too_few_frames"

    # ---- 第一阶段：粗看本段，出题 + 粗略 timeline ----
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(frames)},
    ]
    try:
        raw = call_model(client, model, messages, temperature=temperature, max_tokens=16384)
    except Exception as e:
        save_json({"video_uid": video_uid, "segment": seg_label,
                   "error": f"call_failed:{e}", "qa_pairs": []}, out_path)
        return "call_failed"

    parsed = parse_qa_json(raw)
    if not parsed or not isinstance(parsed.get("qa_pairs"), list):
        save_json({"video_uid": video_uid, "segment": seg_label, "error": "parse_failed",
                   "raw": raw[:4000], "qa_pairs": []}, out_path)
        return "parse_failed"

    coarse_kept, rejected = [], []
    for qa in parsed["qa_pairs"]:
        qa = dedup_adjacent_timeline(qa)
        qa = fix_increasing_time(qa)
        ok, why = validate_qa(qa)
        if ok:
            coarse_kept.append(qa)
        else:
            rejected.append({"reason": f"coarse:{why}", "qa": qa})

    # ---- 第二阶段：逐变化点密集细化时间戳(本地秒)----
    kept = []
    for qa in coarse_kept:
        if refine:
            qa = refine_timeline(
                qa, client, model, sec_map, max_sec,
                refine_half_window, refine_max_frames, temperature,
            )
            qa = dedup_adjacent_timeline(qa)
        ok, why = validate_qa(qa)
        if ok:
            slim = slim_qa_for_output(qa)
            slim["num_updates"] = len(slim["timeline"])
            kept.append(slim)
        else:
            rejected.append({"reason": f"post_refine:{why}", "qa": qa})

    save_json({
        "video_uid": video_uid,
        "segment": seg_label,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "duration": max_sec,               # 段内本地时长(秒)
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
    ap.add_argument("--frame_root", default=DEFAULT_FRAME_ROOT)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--video_dir_map", default=None)
    ap.add_argument("--min_frames", type=int, default=MIN_FRAMES,
                    help="只处理最大帧号 >= 该值的视频(1800 即 >=30min)")
    ap.add_argument("--sample_every_sec", type=int, default=8, help="第一阶段每隔多少秒采一帧")
    ap.add_argument("--max_frames", type=int, default=120, help="第一阶段送入 teacher 的最大帧数")
    ap.add_argument("--refine", dest="refine", action="store_true", default=True)
    ap.add_argument("--no_refine", dest="refine", action="store_false")
    ap.add_argument("--refine_half_window", type=int, default=15)
    ap.add_argument("--refine_max_frames", type=int, default=31)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="冒烟：只处理前 N 个视频")
    ap.add_argument("--start_idx", type=int, default=None,
                    help="处理排序后长视频列表的 [start_idx, end_idx)；也可用环境变量 VID_START")
    ap.add_argument("--end_idx", type=int, default=None, help="见 --start_idx；也可用环境变量 VID_END")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 遍历抽帧目录，用最大帧号定“真实时长”，筛出长视频(>=min_frames)。
    root = Path(args.frame_root)
    video_dir_map = None
    if args.video_dir_map:
        import json
        video_dir_map = json.load(open(args.video_dir_map, encoding="utf-8"))

    long_videos: List[Tuple[str, int]] = []  # (video_uid, max_frame)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        video_uid = d.name.lstrip("_") if d.name.startswith("_") else d.name
        # 两级筛选：先用便宜的 exists 探测第 min_frames 号帧(4位补零)快速初筛，
        # 只对通过初筛的目录再 glob 求真实 max_frame(昂贵)，避免全量 glob 拖慢冷启动。
        probe = d / f"frame_{args.min_frames:04d}.jpg"
        if not probe.exists():
            continue
        mf = max_frame_number(list_frame_files(str(d)))
        if mf >= args.min_frames:
            long_videos.append((video_uid, mf))
    total_all = len(long_videos)

    # 范围分片(与原脚本一致)。
    start_idx = args.start_idx
    end_idx = args.end_idx
    if start_idx is None and os.environ.get("VID_START") is not None:
        start_idx = int(os.environ["VID_START"])
    if end_idx is None and os.environ.get("VID_END") is not None:
        end_idx = int(os.environ["VID_END"])
    s = start_idx if start_idx is not None else 0
    e = end_idx if end_idx is not None else total_all
    long_videos = long_videos[s:e]
    if args.limit:
        long_videos = long_videos[:args.limit]

    # 展开成“段任务”列表。
    seg_tasks: List[Tuple[str, List[int]]] = []
    for uid, mf in long_videos:
        for seg in plan_segments(mf):
            seg_tasks.append((uid, seg))

    print(f"[INFO] 长视频(>= {args.min_frames}帧) 全量={total_all}  本分片[{s}:{e}]取={len(long_videos)}  "
          f"展开段任务={len(seg_tasks)}  model={args.model}")
    print(f"[INFO] stage1: sample_every_sec={args.sample_every_sec} max_frames={args.max_frames}")
    print(f"[INFO] stage2(refine={args.refine}): half_window={args.refine_half_window} "
          f"refine_max_frames={args.refine_max_frames}")
    print(f"[INFO] workers={args.num_workers} out={args.output_dir}")

    def _worker(task):
        uid, seg = task
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        video_dir = locate_video_dir(args.frame_root, uid, video_dir_map)
        if not video_dir:
            seg_label = f"{seg[2]}_{seg[3]}"
            save_json({"video_uid": uid, "segment": seg_label,
                       "error": "video_dir_not_found", "qa_pairs": []},
                      os.path.join(args.output_dir, f"{uid}_{seg_label}.json"))
            return uid, seg, "no_video_dir"
        frame_files = list_frame_files(video_dir)
        status = process_one_segment(
            uid, seg, frame_files, client, args.model, args.output_dir,
            args.sample_every_sec, args.max_frames, args.overwrite, args.temperature,
            refine=args.refine, refine_half_window=args.refine_half_window,
            refine_max_frames=args.refine_max_frames,
        )
        return uid, seg, status

    done = 0
    stat = Counter()
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(_worker, t): t for t in seg_tasks}
        for fut in as_completed(futs):
            task = futs[fut]
            try:
                uid, seg, status = fut.result()
            except Exception as ex_e:
                uid, seg, status = task[0], task[1], f"fatal:{ex_e}"
            stat[status.split(":")[0]] += 1
            done += 1
            print(f"[PROGRESS] {done}/{len(seg_tasks)} {uid}_{seg[2]}_{seg[3]} -> {status}")
    print(f"[DONE] status分布: {dict(stat)}")


if __name__ == "__main__":
    main()
