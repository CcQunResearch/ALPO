from utils import check_equal

def extract_zh_from_prompt(src_prompt):
    if '原文：' in src_prompt:
        start = src_prompt.find('原文：')
        res = src_prompt[start+3:].lstrip()
        end = res.find('翻译结果：')
        return [elem[elem.find('.')+1:].strip() for elem in res[:end].strip().split('\n') if elem.strip()!='']
    else:
        print('No translation found')
        return '' 

def extract_tr_from_prompt(src_prompt):
    if '专有名词翻译：' in src_prompt:
        start = src_prompt.find('专有名词翻译：')
        res = src_prompt[start+7:].lstrip()
        end = res.find('\n\n')
        return [elem for elem in res[:end].strip().split('\n') if elem.strip()!='']
    else:
        print('No translation found')
        return []
    
def extract_tar(texts):
    res = []
    for text in texts.split('\n'):
        start = text.find('.')
        if start != -1:
            res.append(text[start + 1:].strip())
    return res

def check_quality(prompt, pred, print_log=True):
    prompt_zhs = extract_zh_from_prompt(prompt)
    pred_tars = extract_tar(pred)
    if len(prompt_zhs) != len(pred_tars):
        if print_log:
            print(f"  len(prompt_zhs) != len(pred_tars): {len(prompt_zhs)}!={len(pred_tars)}")
        return False
    return True