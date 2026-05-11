import ray
import os
import numpy as np
import time
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


@ray.remote(num_gpus=1)
class DynamicBatchWorker:
    def __init__(self, worker_id, model_name, tokenizer, max_tokens, stop_token_id, temperature, top_p, top_k, lora_name):
        self.worker_id = worker_id
        self.stop_token_id = stop_token_id
        self.tokenizer = tokenizer
        self.lora_name = lora_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_num_seqs=256,
            max_model_len=8192,
            enforce_eager=False
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens
        )

    def generate(self, prompts: list, start_idx: int):
        start_time = time.time()
        if self.lora_name is None:
            outputs = self.llm.generate(prompts, self.sampling_params)
        else:
            outputs = self.llm.generate(prompts, self.sampling_params, lora_request=LoRARequest("lora model", 1, self.lora_name))
        
        total_tokens = sum(len(output.prompt_token_ids) + len(output.outputs[0].token_ids) for output in outputs)
        
        texts = []
        token_ids = []
        for output in outputs:
            response_token_ids = output.outputs[0].token_ids
            stop_index = response_token_ids.index(self.stop_token_id) if self.stop_token_id in response_token_ids else len(response_token_ids)
            truncated_token_ids = response_token_ids[:stop_index]
            texts.append(self.tokenizer.decode(truncated_token_ids))
            token_ids.append(response_token_ids)
        
        return {
            "texts": texts,
            "token_ids": token_ids,
            "start_idx": start_idx,
            "worker_id": self.worker_id,
            "num_prompts": len(prompts),
            "total_tokens": total_tokens,
            "time_cost": time.time() - start_time
        }
        
def infer(prompts, gpus, model_name, max_tokens=2048, stop_token_id=151645, temperature=0.7, top_p=0.9, top_k=40, lora_name=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    num_gpus = len(gpus.split(','))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ray.init()
    workers = [DynamicBatchWorker.remote(i, model_name, tokenizer, max_tokens, stop_token_id, temperature, top_p, top_k, lora_name) for i in range(num_gpus)]
    prompt_chunks = np.array_split(prompts, num_gpus)
    
    split_starts = []
    current = 0
    for chunk in prompt_chunks:
        split_starts.append(current)
        current += len(chunk)
    
    futures = [workers[i].generate.remote(chunk, split_starts[i]) for i, chunk in enumerate(prompt_chunks)]
    
    results = [None] * len(prompts) 
    token_ids = [None] * len(prompts)
    stats = defaultdict(list)
    with tqdm(total=len(prompts), desc="Processing", unit="prompt") as pbar:
        while futures:
            ready, futures = ray.wait(futures, timeout=5.0)
            if not ready:
                continue
            
            for future in ready:
                result = ray.get(future)
                s = result["start_idx"]
                chunk_texts = result["texts"]
                results[s:s+len(chunk_texts)] = chunk_texts
                
                chunk_token_ids = result["token_ids"]
                token_ids[s:s+len(chunk_texts)] = chunk_token_ids
                
                stats["tokens"].append(result["total_tokens"])
                stats["time"].append(result["time_cost"])
                stats["worker"].append(result["worker_id"])
                pbar.update(len(chunk_texts))
                pbar.set_postfix({
                    "Speed (tokens/s)": f"{result['total_tokens']/result['time_cost']:.0f}",
                    "GPU": result["worker_id"]
                })
    
    total_tokens = sum(stats["tokens"])
    total_time = max(stats["time"])
    print(f"\n完成 {len(prompts)} 条推理 | "
          f"总Tokens: {total_tokens:,} | "
          f"总耗时: {total_time / 60:.2f} minutes, {total_time / 3600:.2f} hours | "
          f"平均速度: {total_tokens / total_time:.0f} tokens/s")
    ray.shutdown()
    # return results, token_ids
    return results