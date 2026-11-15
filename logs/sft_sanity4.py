from vllm import LLM, SamplingParams

def main():
    prompt = "You are a math expert.\nYou must answer in English.\n\nQuestion: What is 2+3? Please reason step by step and put your final answer within \\boxed{}."
    llm = LLM("/data3/yxfeng/cscwd27_iraker/run1/sft/policy_sft", trust_remote_code=True, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.85)
    p = SamplingParams(n=2, temperature=0.7, top_p=0.95, max_tokens=200, stop_token_ids=[151643, 151645])
    out = llm.generate([prompt], p, use_tqdm=False)
    print("=== sft checkpoint, explicit stop ===")
    for o in out[0].outputs:
        print(repr(o.text[:150]), "finish=", o.finish_reason, "ntok=", len(o.token_ids))

if __name__ == "__main__":
    main()
