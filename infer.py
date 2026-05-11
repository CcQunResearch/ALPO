import yaml
import os.path as osp

command = """# 指定使用的环境并定义模型与语言变量
source [your_env_path]
sft_model="{}"
tr_model="{}"
trpe_model="{}"
lang="{}"
gpus="{}"
port="{}"

cd MainPipeline
python \[8-1\]extract_test_data.py
python \[8-2\]extract_test_data.py
python \[8-3\]extract_test_data.py
python \[8-4\]extract_test_data.py
python \[9\]prediction_2_subtitle.py
python \[10\]evaluate.py"""

if __name__ == "__main__":
    dirname = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(dirname, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    sft_model = config['sft_model']
    tr_model = config['tr_model']
    trpe_model = config['trpe_model']
    lang = config['lang']
    gpus = config['gpus']
    port = config['port']
    
    run_command = command.format(sft_model, tr_model, trpe_model, lang, gpus, port)
    run_command = run_command.replace('$sft_model','${sft_model}')
    run_command = run_command.replace('$tr_model','${tr_model}')
    run_command = run_command.replace('$trpe_model','${trpe_model}')
    run_command = run_command.replace('$lang','${lang}')
    run_command = run_command.replace('$gpus','${gpus}')
    run_command = run_command.replace('$port','${port}')
    with open(osp.join(dirname, 'infer_command.sh'), 'w', encoding='utf-8') as file:
        file.write(run_command)
