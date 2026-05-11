import json
import yaml
import time
import concurrent.futures
import os
import os.path as osp
from tqdm import tqdm
from chat import chat_ali

def identify_proper_noun(prompt, model, retry_num=3):
    retry_count = 0
    while retry_count < retry_num:
        try:
            response, _ = chat_ali(prompt, model)
        except Exception as e:
            print(f"trpe: {model}发生异常：{e}")
            retry_count += 1
            time.sleep(1)
        else:
            return response
    return None


if __name__ == '__main__':
    dirname = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(dirname, '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    trpe_model = config['trpe_model']
    trpe_num_workers = config['trpe_num_workers']
    lang = config['lang']
    dataset_path = osp.join(dirname, '..', 'LLaMAFactory', 'data', f'proper_noun_queries_train_{lang}.json')
    output_dir = osp.join(dirname, '..', 'LLaMAFactory', 'TermRecognition', 'train', trpe_model, lang)
    save_file_path = osp.join(output_dir, 'generated_predictions.jsonl')
    exception_index_path = osp.join(output_dir, 'final_exception_indexes.jsonl')
    os.makedirs(output_dir, exist_ok=True)
    
    print('[2-2] Request online service to identify terms...')
    if osp.exists(save_file_path):
        print(f'  Indentify file already exists, skip processing.')
    else:
        begin_time = time.time()
        generated_predictions = []
        queries = json.load(open(dataset_path, 'r', encoding='utf-8'))
        tasks = [[query["instruction"], trpe_model] for query in queries]
        with concurrent.futures.ThreadPoolExecutor(max_workers=trpe_num_workers) as executor:
            futures = [executor.submit(identify_proper_noun, *para) for para in tasks]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
                pass
        results = [future.result() for future in futures]
        final_exception_indexes = [i for i, result in enumerate(results) if result is None]
        generated_predictions = [{"prompt": query["instruction"], "label": "", "predict": result} for query, result in zip(queries, results)]
        for i in final_exception_indexes:
            generated_predictions[i]["predict"] = "无专有名词"
                    
        json.dump(final_exception_indexes,  open(exception_index_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)  
        with open(save_file_path, 'w', encoding='utf-8') as f:
            for item in generated_predictions:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        end_time = time.time()
        print(f'Finished in {end_time - begin_time} seconds.')
    print('[2-2] Request online service to identify terms.')