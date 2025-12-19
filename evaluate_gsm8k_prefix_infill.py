import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from lit_gpt.config import Config
from lit_gpt.diffmodel import TransEncoder
from lit_gpt.utils import lazy_load


def _normalize_text(s: str) -> str:
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_question_prefix(s: str) -> str:
    s = s.strip()
    if s.lower().startswith("question:"):
        s = s.split(":", 1)[1].strip()
    return _normalize_text(s)


def _is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        return first.startswith("version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _download_text(url: str) -> str:
    with urlopen(url) as resp:  # nosec - user-controlled URLs not accepted
        return resp.read().decode("utf-8")


def ensure_gsm8k_txt(split: str, out_path: Path) -> None:
    """Write a gsm8k `{split}.txt` file in the `question||answer` format.

    `answer` is the official GSM8K answer string containing chain-of-thought + `#### final`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not _is_git_lfs_pointer(out_path):
        return

    url = (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"
        f"{split}.jsonl"
    )
    raw = _download_text(url).strip().splitlines()
    with out_path.open("w", encoding="utf-8") as f:
        for line in raw:
            ex = json.loads(line)
            question = _normalize_text(ex["question"])
            answer = _normalize_text(ex["answer"])
            f.write(f"{question}||{answer}\n")


def iter_gsm8k_examples_from_txt(path: Path) -> Iterable[Tuple[str, str, str]]:
    """Yields (question, thought, answer) text pieces."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if "||" not in line or "####" not in line:
                continue
            question_part, rest = line.split("||", 1)
            thought_part, answer_part = rest.split("####", 1)
            question = "Question: " + question_part.strip()
            thought = "Answer: " + thought_part.strip()
            answer = "####" + answer_part.strip()
            yield question, thought, answer


def _get_cuda_amp_dtype() -> torch.dtype:
    # bf16 tensor cores are available on Ampere (SM8x) and newer.
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
    return torch.float16


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def diff_sample_infill(
    model: torch.nn.Module,
    x: torch.Tensor,
    condition_mask: torch.Tensor,
    *,
    alg: str = "origin",
    steps: int = 512,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    eps: float = 1e-5,
    dim: int = 32000,
    device: str = "cuda",
) -> torch.Tensor:
    x = x.clone().to(device)
    condition_mask = condition_mask.to(device).to(torch.bool)

    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    amp_dtype = _get_cuda_amp_dtype() if str(device).startswith("cuda") else torch.float32

    for i in range(steps):
        mask_index = x == dim
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[condition_mask] = dim
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits, un_logits = logits[mask_index], un_logits[mask_index]
            else:
                logits = model(x)[mask_index]

        if cfg_scale > 0.0:
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == "origin":
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + dim
            transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
            logits_with_noise = add_gumbel_noise(logits[transfer_index_t_s], temperature=temperature)
            x0[transfer_index_t_s] = torch.argmax(logits_with_noise, dim=-1)
            x[mask_index] = x0.clone()
        elif alg == "greddy":
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            logits_fp64 = logits.to(torch.float64)
            p = F.softmax(logits_fp64, dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            num_mask_token = mask_index.sum()
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
            if number_transfer_tokens > 0:
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + dim
                x0_[transfer_index] = x0[transfer_index].clone()
                x[mask_index] = x0_
        else:
            raise NotImplementedError(alg)

    return x


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def load_model_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    if ckpt_path.suffix == ".safetensors":
        return load_file(str(ckpt_path))

    with lazy_load(str(ckpt_path)) as ckpt:
        if not isinstance(ckpt, dict):
            raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

        # Common patterns:
        # - Fabric save state dict: {"model": model.state_dict(), ...}
        # - Torch save: {"state_dict": ...}
        if isinstance(ckpt.get("model"), dict):
            state_dict = ckpt["model"]
        elif isinstance(ckpt.get("state_dict"), dict):
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Materialize tensors while the zipfile is still open. `lazy_load` returns `NotYetLoadedTensor` objects
        # that require an open file handle; if we exit the context before materialization, `load_state_dict`
        # will fail with `'NoneType' object has no attribute 'get_storage_from_record'`.
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    for prefix in ("module.", "_forward_module.", "model."):
        state_dict = _strip_state_dict_prefix(state_dict, prefix)
    return state_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to .safetensors or Fabric .pth checkpoint")
    p.add_argument("--model", type=int, default=1028, help="Model size in M (e.g. 1028)")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--data_file", type=str, default=None, help="Optional path to gsm8k {split}.txt")
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--alg", type=str, default="greddy", choices=["origin", "greddy"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cfg_scale", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_examples", type=int, default=32, help="0 = run all")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output_file", type=str, default=None, help="Optional JSONL output path")
    p.add_argument("--show_condition", action="store_true", help="Print the conditioned CoT+answer text")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("This script currently requires CUDA for reasonable speed.")

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    model_name = f"Diff_LLaMA_{args.model}M"
    config = Config.from_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", padding_side="right", use_fast=True
    )

    model = TransEncoder(config).to(device)
    model.load_state_dict(load_model_state_dict(ckpt_path), strict=True)
    model.eval()

    data_file = Path(args.data_file) if args.data_file else Path(f"data/gsm8k/{args.split}.txt")
    ensure_gsm8k_txt(args.split, data_file)

    examples = list(iter_gsm8k_examples_from_txt(data_file))
    if args.num_examples > 0:
        examples = random.sample(examples, k=min(args.num_examples, len(examples)))

    mask_id = 32000
    eos_id = tokenizer.eos_token_id

    out_f = None
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    total = 0
    skipped = 0
    exact = 0

    batch_x: List[torch.Tensor] = []
    batch_cond: List[torch.Tensor] = []
    batch_meta: List[Dict[str, object]] = []

    def flush_batch() -> None:
        nonlocal total, exact
        if not batch_x:
            return
        x = torch.stack(batch_x, dim=0)
        cond = torch.stack(batch_cond, dim=0)
        out = diff_sample_infill(
            model,
            x,
            cond,
            alg=args.alg,
            steps=args.steps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            dim=mask_id,
            device=device,
        )
        for i, meta in enumerate(batch_meta):
            q_len = int(meta["q_len"])
            gt_question = str(meta["gt_question"])
            pred_question = tokenizer.decode(out[i, :q_len].tolist(), skip_special_tokens=True)

            total += 1
            if _strip_question_prefix(pred_question) == _strip_question_prefix(gt_question):
                exact += 1

            print(f"[{total}] EM={exact}/{total} ({exact/total:.3f}) q_len={q_len}")
            print("GT:", gt_question)
            print("PR:", pred_question)
            if args.show_condition:
                print("COND:", str(meta["condition"])[:400])
            print("-" * 80)

            if out_f:
                rec = {
                    "gt_question": gt_question,
                    "pred_question": pred_question,
                    "condition": meta["condition"],
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        batch_x.clear()
        batch_cond.clear()
        batch_meta.clear()

    for gt_question, thought, answer in examples:
        q_ids = tokenizer(gt_question, return_tensors="pt")["input_ids"][0]
        t_ids = tokenizer(thought, return_tensors="pt")["input_ids"][0]
        a_ids = tokenizer(answer, return_tensors="pt")["input_ids"][0]
        a_ids = torch.cat([a_ids, torch.tensor([eos_id], dtype=a_ids.dtype)], dim=-1)

        full = torch.cat([q_ids, t_ids, a_ids], dim=-1)
        if full.numel() > args.seq_len:
            full = full[: args.seq_len]

        used_len = full.numel()
        q_len = min(q_ids.numel(), used_len)
        if q_len >= used_len:
            skipped += 1
            continue

        x = torch.full((args.seq_len,), eos_id, dtype=torch.long)
        x[:used_len] = full
        x[:q_len] = mask_id

        cond = torch.zeros((args.seq_len,), dtype=torch.bool)
        cond[q_len:used_len] = True

        batch_x.append(x)
        batch_cond.append(cond)
        batch_meta.append(
            {
                "gt_question": gt_question,
                "condition": _normalize_text(thought + " " + answer),
                "q_len": q_len,
            }
        )

        if len(batch_x) >= args.batch_size:
            flush_batch()

    flush_batch()

    if out_f:
        out_f.close()

    if skipped:
        print(f"Skipped {skipped} examples due to length constraints.")
    print(f"Final normalized exact match: {exact}/{total} ({(exact/total) if total else 0.0:.3f})")


if __name__ == "__main__":
    main()
