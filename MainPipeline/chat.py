import requests
from openai import OpenAI
import json

# qwen2.5-72b-instruct, qwen-max-latest, deepseek-v3
def chat_ali(text, model, logprobs=False, temperature=0.0):
    api_key = 'api_key'
    base_url = 'base_url'
    if model in ['qwen2.5-14b-instruct']:
        api_key = 'api_key'
    headers = {'Authorization': api_key,
               'Content-Type': 'application/json'}
    if model == 'deepseek-v3' or model == 'deepseek-r1':
        q = {"messages": [{"content": text, "role": "user"}], "model": model, "temperature": temperature, "stream": False}
    else:
        q = {"messages": [{"content": text, "role": "user"}], "model": model, "temperature": temperature, "stream": False, "logprobs": logprobs}
    r = requests.post(
        base_url,
        json=q,
        headers=headers,
        timeout=600
    )
    
    resp_json = r.json()
    resp_text = resp_json['choices'][0]['message']['content']
    if logprobs:
        probs = resp_json['choices'][0]['logprobs']['content']
        return resp_text, probs
    else:
        return resp_text, None
    
def chat_qwen3(txt, model, enable_thinking=True):
    headers = {'Authorization': 'api_key',
               'Content-Type': 'application/json'}

    q = {"messages": [{"content": txt, "role": "user"}], "model": model, "enable_thinking": enable_thinking}
    q["stream"] = True
    r = requests.post(
        'base_url', # 弹外
        json=q,
        headers=headers,
        timeout=600
    )
    reasoning_text = ''
    for c in r.iter_lines():
        if c:
            c = c.decode()[6:]
            try:
                c = json.loads(c)
                reasoning_text += c['choices'][0]['delta']['reasoning_content']
            except:
                pass
    
    resp_text = ''
    for c in r.iter_lines():
        if c:
            c = c.decode()[6:]
            try:
                c = json.loads(c)
                resp_text += c['choices'][0]['delta']['content']
            except:
                pass
    return resp_text.strip(), reasoning_text.strip()

# gpt3.5, gpt4o, claude35sonnet
def chat_gpt(text, model, logprobs=False):
    headers = {'Authorization': 'api_key',
               'Content-Type': 'application/json'}

    q = {"messages": [{"content": text, "role": "user"}], "model": model, "stream": False, "logprobs": False}
    r = requests.post(
        'base_url', # 弹外
        json=q,
        headers=headers,
        timeout=600
    )
    
    resp_json = r.json()
    resp_text = resp_json['choices'][0]['message']['content']
    if logprobs:
        probs = resp_json['choices'][0]['logprobs']['content']
        return resp_text, probs
    else:
        return resp_text, None

# deepseek v3
def chat_ds3(text, logprobs=False):
    client = OpenAI(api_key="api_key", base_url="base_url")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": text},
        ],
        stream=False,
        logprobs=logprobs
    )
    
    resp_json = response.to_dict()
    resp_text = resp_json['choices'][0]['message']['content']
    if logprobs:
        probs = resp_json['choices'][0]['logprobs']['content']
        return resp_text, probs
    else:
        return resp_text, None