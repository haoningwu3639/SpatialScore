import os
import re
import json
import argparse
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from vllm import LLM, SamplingParams

# Import local utility functions
from utils.util import extract_yes_no, extract_option, extract_number, extract_numeric_with_unit


# ---------------------------
# vLLM init with auto-shrink (aligned with test_output.py)
# ---------------------------
def init_vllm_with_autoshrink(model_path: str, tps: int, gpu_mem_util: float, batch_size: int) -> LLM:
    """
    Try to initialize vLLM with a descending list of gpu_memory_utilization.
    Avoids failing at startup when free GPU memory < desired utilization.
    """
    util_candidates = []
    if gpu_mem_util is not None and gpu_mem_util > 0:
        util_candidates.append(gpu_mem_util)
    util_candidates += [0.30, 0.20, 0.10]

    last_err = None
    tried = []
    for util in util_candidates:
        if util in tried or util <= 0:
            continue
        tried.append(util)
        try:
            print(f"[vLLM] Trying gpu_memory_utilization={util} ...", flush=True)
            llm = LLM(
                model=model_path,
                tensor_parallel_size=tps,
                gpu_memory_utilization=util,
                swap_space=4,
                max_num_seqs=max(1, batch_size),
                max_model_len=6000,
                trust_remote_code=True,
                enforce_eager=True,
            )
            print(f"[vLLM] OK at gpu_memory_utilization={util}", flush=True)
            return llm
        except Exception as e:
            last_err = e
            print(f"[vLLM] Failed at util={util}: {str(e)[:200]}...", flush=True)
            continue
    raise RuntimeError(f"vLLM init failed after tries {tried}: {last_err}")


def _vllm_complete_batch(llm: LLM, prompts: List[str], max_new_tokens: int = 2048) -> List[str]:
    """Batched vLLM generate, aligned with test_output.py."""
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        n=1,
        stop=None,
    )
    outs = llm.generate(prompts, sp, use_tqdm=False)
    results = []
    for out in outs:
        if out and out.outputs:
            text = out.outputs[0].text
            results.append(text.strip() if text else "")
        else:
            results.append("")
    return results


# ---------------------------
# 1. Configuration & Prompts
# ---------------------------

_LLM_PROMPT_DISCRETE = """You are a strict checker. Your task is ONLY to determine if the model's prediction matches the ground truth AFTER proper parsing/mapping.

Do NOT answer/solve the question. Do NOT explain.
Rules to consider:
- If the prediction is a number that corresponds exactly to an option's content, treat it as that option.
- Ignore minor format differences like missing parentheses or case.

Output ONLY one word: YES or NO.

Ground Truth: {gt_answer}
Prediction: {pred_answer}
Options:
{options}
"""

_LLM_PROMPT_TEMPLATE = """You are an evaluator. Your ONLY job is to compute a score using the following algorithm. Do NOT answer or solve the question.

TASK TYPE:
- If Type == "counting": treat both GT and PRED as plain scalar numbers (no unit conversion).
- If Type == "distance": parse numeric value + unit; if PRED unit is missing, borrow GT unit; if both are missing and both look like plain numbers, treat as scalar.
- If a numeric RANGE like "10-15" appears, use the MAX value (e.g., 15).

ALGORITHM (VSI-Bench MRA):
1) Compute abs_dist_norm:
   - For scalar/counting: abs_dist_norm = abs(pred - gt) / gt   (if gt == 0, set abs_dist_norm = +Infinity)
   - For distance: convert both to centimeters using:
       meter (m): 100 cm; centimeter (cm): 1 cm; millimeter (mm): 0.1 cm; inch (in): 2.54 cm; foot (ft): 30.48 cm.
     Then abs_dist_norm = abs(pred_cm - gt_cm) / gt_cm  (if gt_cm == 0, set +Infinity).
2) For thresholds C = linspace(start, end, steps) with steps = int((end - start)/interval + 2):
     accuracy(C) = 1 if abs_dist_norm <= (1 - C) else 0
   mean_relative_accuracy = average of accuracy(C) over all thresholds.
3) The final score is this mean_relative_accuracy, a float in [0,1].

IMPORTANT OUTPUT RULE:
- After you finish the calculation, OUTPUT EXACTLY ONE LINE at the end in the form:
  output: <float>
  For example: output: 0.83

Config:
- start={start}
- end={end}
- interval={interval}

Inputs:
- Type: {open_type}   # "counting" or "distance"
- gt_answer: {gt_answer}
- pred_answer: {pred_answer}
"""

# ---------------------------
# aligned helpers
# ---------------------------
_NUMBER_ONLY_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")
_HAS_UNIT_RE = re.compile(r"\d+\.?\d*\s*[a-zA-Z]+")

def _is_plain_number_no_unit(s: str) -> bool:
    """True if s looks like a bare number (int/float) without any unit."""
    s = str(s or "")
    return bool(_NUMBER_ONLY_RE.match(s)) and not _HAS_UNIT_RE.search(s)

def calculate_mra(pred: float, target: float, start: float, end: float, interval: float) -> float:
    """
    Core Mean Relative Accuracy (VSI-Bench).
    """
    if target == 0:
        return 1.0 if pred == 0 else 0.0
    num_pts = int((end - start) / interval + 2)
    thresholds = np.linspace(start, end, num_pts)
    abs_dist_norm = abs(pred - target) / target
    scores = abs_dist_norm <= (1 - thresholds)
    return float(scores.mean())

def determine_open_type(sample: Dict[str, Any]) -> str:
    """
    - If task/sub_task indicates counting -> counting
    - Else if it's open-ended AND (gt or pred is plain number w/o unit) -> counting
    - Else -> distance
    """
    q_type = str(sample.get("question_type", "")).lower()
    sub = (sample.get("sub_task") or sample.get("subtask") or "").lower()
    task = (sample.get("task") or "").lower()
    gt = str(sample.get("gt_answer") or sample.get("gt") or "")
    pred = str(sample.get("pred_answer") or sample.get("response") or "")

    if "count" in sub or "count" in task:
        return "counting"

    is_open = "open" in q_type
    if is_open and (_is_plain_number_no_unit(gt) or _is_plain_number_no_unit(pred)):
        return "counting"

    return "distance"

def _parse_llm_yes_no(text: str) -> bool:
    t = (text or "").strip().lower()
    # Be tolerant: accept YES/NO if it appears anywhere; prefer first token-like match.
    if re.search(r"\byes\b", t):
        return True
    if re.search(r"\bno\b", t):
        return False
    return False

def _extract_llm_mra_output(text: str) -> float:
    # Robustly grab the LAST "output: <number>" (same as your prior code)
    matches = list(re.finditer(r'output\s*:\s*([0-9]*\.?[0-9]+)', text or "", flags=re.IGNORECASE))
    if not matches:
        return 0.0
    val = float(matches[-1].group(1))
    return max(0.0, min(1.0, val))

# ---------------------------
# 2. Stage 1: Rule-based Logic
# ---------------------------
def evaluate_via_rules(sample: Dict, mra_cfg: Dict) -> Tuple[float, bool, str]:
    """Stage-1 rules. Return (stage1_score, stage1_correct, method)."""
    q_type = str(sample.get("question_type", "")).lower()
    gt = str(sample.get("gt_answer") or sample.get("gt") or "")
    pred = str(sample.get("pred_answer") or sample.get("response") or "")

    # A. Multiple Choice (exact whitelist match, aligned with test_output.py)
    if q_type in ["multi-choice", "multiple-choice", "mc", "multi choice"]:
        options = sample.get("options", [])
        if not options and sample.get("question"):
            options = re.findall(r"\([A-F]\)\s*([^\n]+)", sample.get("question"))

        gt_letter = extract_option(gt)
        pred_letter = extract_option(pred)

        if not re.fullmatch(r"[A-F]", str(pred_letter) if pred_letter else ""):
            num = extract_number(pred)
            mapped = None
            if options:
                for idx, op in enumerate(options):
                    if str(op).strip() == str(num).strip():
                        mapped = chr(ord('A') + idx)
                        break
                    if str(op).strip().isdigit() and str(num).strip().isdigit():
                        if int(op) == int(num):
                            mapped = chr(ord('A') + idx)
                            break
            if mapped is not None:
                pred_letter = mapped

        gt_letter = (str(gt_letter).upper() if gt_letter else None)
        pred_letter = (str(pred_letter).upper() if pred_letter else None)
        is_correct = (pred_letter is not None and gt_letter is not None and pred_letter == gt_letter)
        return (1.0 if is_correct else 0.0), is_correct, "rule-choice"

    # B. Judgement (Yes/No) (exact whitelist match, aligned with test_output.py)
    if q_type in ["judgement", "judgment", "yes/no", "yesno"]:
        is_correct = extract_yes_no(pred).lower() == extract_yes_no(gt).lower()
        return (1.0 if is_correct else 0.0), is_correct, "rule-judgement"

    # C. Open-Ended (Numeric/MRA), aligned with test_output.py
    if q_type in ["open-ended", "open ended", "open"]:
        open_type = determine_open_type(sample)

        # ---- counting path ----
        if open_type == "counting":
            try:
                g_val = float(extract_number(gt))
                p_val = float(extract_number(pred))
                score = calculate_mra(p_val, g_val, **mra_cfg)
                return score, (score == 1.0), "rule-counting"
            except Exception:
                return 0.0, False, "rule-counting-fail"

        # ---- distance path: try cm conversion first, then scalar fallback ----
        res = extract_numeric_with_unit(pred, gt=gt, tolerance=2.0)
        try:
            p_val = float(res.get("answer_value"))
            g_val = float(res.get("gt_value"))
            score = calculate_mra(p_val, g_val, **mra_cfg)
            return score, (score == 1.0), "rule-distance"
        except Exception:
            def _pick_scalar(x: str):
                # support ranges like "10-15" by taking the max
                nums = re.findall(r'-?\d+(?:\.\d+)?', str(x or ""))
                return max([float(v) for v in nums]) if nums else None

            pr_s = _pick_scalar(pred)
            gt_s = _pick_scalar(gt)
            if pr_s is not None and gt_s is not None:
                try:
                    score = calculate_mra(pr_s, gt_s, **mra_cfg)
                    return score, (score == 1.0), "rule-distance-scalar-fallback"
                except Exception:
                    return 0.0, False, "rule-distance-fail"
            return 0.0, False, "rule-distance-fail"

    # D. Unknown question_type: align with test_output.py (no implicit string-equality scoring)
    return 0.0, False, "rule-unknown"

# ---------------------------
# 3. Output Formatting
# ---------------------------
def display_metrics(metrics_map):
    """Prints formatted result tables for different granularity levels."""
    for key, label in [('category', 'CATEGORY'), ('task', 'TASK'), ('sub_task', 'SUB-TASK')]:
        print(f"\n{'='*25} {label} {'='*25}")
        print(f"{'Group Name':<45} | {'Acc %':<10} | {'Score/Total':<15}")
        print("-" * 80)
        for name, stats in sorted(metrics_map[key].items()):
            acc = (stats['score_sum'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{str(name)[:43]:<45} | {acc:>7.2f}% | {stats['score_sum']:>6.1f}/{stats['total']}")

# ---------------------------
# 4. Process Manager
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified SpatialScore Evaluation")
    parser.add_argument("--input", required=True, help="Path to directory containing all_results.json")
    parser.add_argument("--llm_path", default="/remote-home/share/huggingface/gpt-oss-20b", help="Path to LLM for Stage 2")
    parser.add_argument("--no_llm", action="store_true", help="Skip LLM-based Stage 2 evaluation")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM")
    # Aligned with test_output.py
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16, help="LLM inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens for LLM. Open-ended MRA prompts need enough budget to finish the algorithm.")
    parser.add_argument("--mra_start", type=float, default=0.5)
    parser.add_argument("--mra_end", type=float, default=0.95)
    parser.add_argument("--mra_interval", type=float, default=0.05)
    args = parser.parse_args()

    mra_cfg = {"start": args.mra_start, "end": args.mra_end, "interval": args.mra_interval}

    input_file = os.path.join(args.input, "all_results.json")
    output_dir = args.input

    # ---- Load data ----
    with open(input_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        raw_data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.split("\n") if l.strip()]

    # ---- Dedup by id, then filter empty predictions ----
    unique_data = {}
    for sample in raw_data:
        sid = str(sample.get("id"))
        unique_data[sid] = sample
    data = list(unique_data.values())
    data = [s for s in data if str(s.get("pred_answer") or s.get("response") or "").strip() != ""]
    print(f"Data Loaded: {len(raw_data)} | After Dedup & Filter: {len(data)}")

    # ---- Stage 1: Rule-based scoring ----
    # Discrete (multi-choice / judgement) samples use rule-based score as final.
    scored_results: List[Dict[str, Any]] = []
    llm_tasks: List[Dict[str, Any]] = []  # only open-ended

    for entry in tqdm(data, desc="Stage 1: Rule evaluation"):
        s1_score, s1_correct, method = evaluate_via_rules(entry, mra_cfg)
        entry_res = {
            **entry,
            "score": s1_score,                  # final score (will be overwritten for open-ended after Stage-2)
            "method": method,
            "s1_score": s1_score,
            "s1_correct": bool(s1_correct),
        }

        if not args.no_llm:
            q_type = str(entry.get("question_type", "")).lower()
            is_open = "open" in q_type

            if is_open:
                gt_raw = entry.get("gt_answer") or entry.get("gt") or ""
                pred_raw = entry.get("pred_answer") or entry.get("response") or ""
                prompt = _LLM_PROMPT_TEMPLATE.format(
                    start=mra_cfg["start"], end=mra_cfg["end"], interval=mra_cfg["interval"],
                    open_type=determine_open_type(entry),
                    gt_answer=json.dumps(gt_raw, ensure_ascii=False),
                    pred_answer=json.dumps(pred_raw, ensure_ascii=False),
                )
                llm_tasks.append({"prompt": prompt, "index": len(scored_results), "kind": "open"})

        scored_results.append(entry_res)

    # ---- Stage 2: LLM-based MRA for open-ended (batched, aligned with test_output.py) ----
    if llm_tasks and not args.no_llm:
        print(f"Executing Stage 2 LLM verification for {len(llm_tasks)} open-ended samples...")
        llm = init_vllm_with_autoshrink(
            model_path=args.llm_path,
            tps=args.tp_size,
            gpu_mem_util=args.gpu_memory_utilization,
            batch_size=args.batch_size,
        )

        prompts = [t["prompt"] for t in llm_tasks]
        total_batches = (len(prompts) + args.batch_size - 1) // args.batch_size

        for batch_id in tqdm(range(0, len(prompts), args.batch_size),
                             desc="Stage 2 LLM inference", unit="batch", total=total_batches):
            batch_prompts = prompts[batch_id:batch_id + args.batch_size]
            batch_metas = llm_tasks[batch_id:batch_id + args.batch_size]
            outputs = _vllm_complete_batch(llm, batch_prompts, max_new_tokens=args.max_new_tokens)

            for text, task_meta in zip(outputs, batch_metas):
                idx = task_meta["index"]
                s1 = float(scored_results[idx]["s1_score"])
                s2 = _extract_llm_mra_output(text)
                final_score = (s1 + s2) / 2.0
                scored_results[idx]["s2_score"] = float(s2)
                scored_results[idx]["s2_raw"] = text  # keep raw LLM output for debugging
                scored_results[idx]["score"] = float(final_score)
                scored_results[idx]["method"] = "fused-rule-llm (mean)"
                scored_results[idx]["final_correct"] = (final_score == 1.0)

    # ---- Aggregation of final metrics ----
    group_keys = ['category', 'task', 'sub_task', 'source_dataset']
    metrics = {k: defaultdict(lambda: {'score_sum': 0.0, 'total': 0}) for k in group_keys}
    for r in scored_results:
        for k in group_keys:
            val = r.get(k) or "unknown"
            metrics[k][val]['score_sum'] += float(r['score'])
            metrics[k][val]['total'] += 1

    # ---- Printing and Saving ----
    display_metrics(metrics)
    n = max(1, len(scored_results))
    total_score = sum(float(r['score']) for r in scored_results)
    print(f"\nOVERALL PERFORMANCE: {total_score / n * 100:.2f}% ({total_score:.1f}/{len(scored_results)})")

    with open(os.path.join(output_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
        json.dump(scored_results, f, indent=2, ensure_ascii=False)

    summary = {
        gk: {n_: {"accuracy": v["score_sum"] / v["total"], "count": v["total"]} for n_, v in d.items()}
        for gk, d in metrics.items()
    }
    summary['overall'] = {"accuracy": total_score / n, "count": len(scored_results)}
    with open(os.path.join(output_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved per-sample: {os.path.join(output_dir, 'detailed_results.json')}")
    print(f"Saved summary:    {os.path.join(output_dir, 'summary_report.json')}")


if __name__ == "__main__":
    main()