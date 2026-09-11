# Copyright (c) 2026 BAAI. All rights reserved.
"""Run a small local checkpoint through the actual vLLM engine and worker."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--multiprocess", action="store_true")
    parser.add_argument("--reference-include", default=None)
    parser.add_argument("--reference-exclude", default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.04)
    args = parser.parse_args()
    if not args.reference and (args.reference_include is not None or args.reference_exclude is not None):
        parser.error("reference include/exclude requires --reference")
    for name, value in (("VLLM_FL_REFERENCE_INCLUDE", args.reference_include),
                        ("VLLM_FL_REFERENCE_EXCLUDE", args.reference_exclude)):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    os.environ.pop("VLLM_FL_CONFIG", None)
    os.environ["VLLM_FL_REFERENCE_MODE"] = "1" if args.reference else "0"
    if not args.reference:
        os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_FL_STRICT"] = "1"
    os.environ["VLLM_FL_PREFER"] = "vendor"
    os.environ["USE_FLAGGEMS"] = "0"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" if args.multiprocess else "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    report_dir = tempfile.mkdtemp(prefix=Path(args.output).stem + "-routes-",
                                  dir=Path(args.output).parent)
    os.environ["VLLM_FL_REFERENCE_REPORT_DIR"] = report_dir
    from vllm import LLM, SamplingParams
    attention_options = ({"attention_config": {"backend": args.attention_backend}}
                         if args.attention_backend else {})
    llm = LLM(
        model=args.model, skip_tokenizer_init=True, dtype="bfloat16",
        max_model_len=64, max_num_seqs=2, max_num_batched_tokens=64,
        gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
        enable_prefix_caching=False, seed=42, kv_cache_memory_bytes=64 * 1024**2,
        **attention_options,
    )
    requests = [{"prompt_token_ids": [1, 8, 9, 10]},
                {"prompt_token_ids": [1, 20, 21]}]
    result = llm.generate(
        requests, SamplingParams(temperature=0., max_tokens=4,
                                 ignore_eos=True, logprobs=5),
        use_tqdm=False,
    )
    output = {
        "model": args.model, "reference": args.reference,
        "multiprocess": args.multiprocess, "report_dir": report_dir,
        "reference_include": args.reference_include,
        "reference_exclude": args.reference_exclude,
        "attention_backend": args.attention_backend,
        "requests": [{"prompt": item.prompt_token_ids,
                       "tokens": list(item.outputs[0].token_ids),
                       "logprobs": [{str(k): float(v.logprob) for k, v in step.items()}
                                    for step in item.outputs[0].logprobs]}
                      for item in result],
    }
    if args.reference:
        from vllm_fl.reference import get_records
        events = {(r["pid"], r["op"], r["source"], r["implementation"], r["reason"]): r
                  for r in get_records()}
        for report in Path(report_dir).glob("reference-*.jsonl"):
            for line in report.read_text().splitlines():
                r = json.loads(line)
                events[(r["pid"], r["op"], r["source"], r["implementation"], r["reason"])] = r
        output["routes"] = list(events.values())
        assert output["routes"], "no reference execution evidence"
        assert not any(row["source"] in {"optimized_fallback", "error", "unavailable", "user_override_error"}
                       for row in output["routes"]), output["routes"]
        if args.attention_backend:
            executions = [row for row in output["routes"] if row["op"] == "attention"]
            assert executions and all(row["source"] == "user_override" for row in executions), executions
            if args.attention_backend == "TRITON_ATTN":
                assert all("vllm.v1.attention.backends.triton_attn.TritonAttentionImpl.forward"
                           == row["implementation"] for row in executions), executions
    Path(args.output).write_text(json.dumps(output, indent=2))
    print("SMOKE_OK", args.output)


if __name__ == "__main__":
    main()
