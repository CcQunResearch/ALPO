import yaml
import os.path as osp

command_a = """# Step 0: 指定使用的环境并定义模型与语言变量
source [your_env_path]
sft_model="{}"
tr_model="{}"
trpe_model="{}"
lang="{}"
gpus="{}"
model_path="{}"
sft_proportion="{}"
dpo_mode="{}"
dpo_finetuning_type="{}"
alignment_method="{}"
alpo_finetuning_type="{}"
gpu_num="{}"

# Step 1: 数据预处理
cd MainPipeline
python \[0\]index_data.py
python \[1\]preprocess.py
python \[2-1\]identify_proper_noun.py"""

command_b1 = """# Step 2: PE术语识别
python \[2-2\]identify_proper_noun.py"""

command_b2 = """# Step 2: PE术语识别
cd ../LLaMAFactory
# CUDA_VISIBLE_DEVICES=$gpus llamafactory-cli train TermRecognition/train/$trpe_model/$lang/predict.yaml"""

command_c = """# Step 3: SFT与术语识别数据集构建
cd ../MainPipeline
python \[2-3\]identify_proper_noun.py
python \[3-1\]construct_dataset.py
python \[3-2\]construct_sample_dataset.py
python \[3-3\]construct_tr_dataset.py

# Step 4: 训练SFT模型与术语识别模型
cd ../LLaMAFactory
CUDA_VISIBLE_DEVICES=$gpus llamafactory-cli train YamlConfig/sft_$sft_model_$trpe_model_$lang_$sft_proportion.yaml
rm -r $model_path/llamafactory/$sft_model_$trpe_model/$lang/sft_default/checkpoint*
CUDA_VISIBLE_DEVICES=$gpus llamafactory-cli train YamlConfig/tr_$tr_model_$trpe_model_$lang.yaml
rm -r $model_path/llamafactory/$tr_model_$trpe_model/$lang/tr_default/checkpoint*"""

command_dpo = """# Step 5: DPO质量对齐采样
cd ../MainPipeline
python \[4\]sample_inference.py
python \[5\]label_preference.py
python \[6-dpo\]construct_dpo_dataset.py

cd ../LLaMAFactory
CUDA_VISIBLE_DEVICES=$gpus llamafactory-cli train YamlConfig/dpo_$sft_model_$trpe_model_$lang_$dpo_mode_$sft_proportion_$dpo_finetuning_type.yaml
rm -r $model_path/llamafactory/$sft_model_$trpe_model/$lang/dpo_$dpo_mode_$sft_proportion_$dpo_finetuning_type/checkpoint*"""

command_alpo = """# Step 5: ALPO质量对齐采样与训练
cd ../MainPipeline
python \[4\]sample_inference.py
python \[5\]label_preference.py
python \[6\]construct_alpo_dataset.py
CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=$gpu_num \[7\]alpo_train.py"""

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
    trpe_mode = config['trpe_mode']
    model_path = config['model_path']
    sft_proportion = config['sft_proportion']
    dpo_mode = config['dpo_mode']
    dpo_finetuning_type = config['dpo_finetuning_type']
    alignment_method = config.get('alignment_method', 'dpo')
    alpo_finetuning_type = config.get('alpo_finetuning_type', dpo_finetuning_type)
    gpu_num = len(gpus.split(','))
    
    if trpe_mode == 'online':
        command_b = command_b1
    else:
        command_b = command_b2
    if sft_proportion == 1.0:
        command_d = ''
    elif alignment_method == 'alpo':
        command_d = command_alpo
    elif alignment_method == 'dpo':
        command_d = command_dpo
    else:
        raise ValueError(f'Unsupported alignment_method: {alignment_method}')
    command = f'{command_a}\n\n{command_b}\n\n{command_c}\n\n{command_d}'
    
    run_command = command.format(sft_model, tr_model, trpe_model, lang, gpus, model_path, sft_proportion, dpo_mode, dpo_finetuning_type, alignment_method, alpo_finetuning_type, gpu_num)
    run_command = run_command.replace('$sft_model','${sft_model}')
    run_command = run_command.replace('$tr_model','${tr_model}')
    run_command = run_command.replace('$trpe_model','${trpe_model}')
    run_command = run_command.replace('$lang','${lang}')
    run_command = run_command.replace('$gpus','${gpus}')
    run_command = run_command.replace('$model_path','${model_path}')
    run_command = run_command.replace('$dpo_mode','${dpo_mode}')
    run_command = run_command.replace('$dpo_finetuning_type','${dpo_finetuning_type}')
    run_command = run_command.replace('$alignment_method','${alignment_method}')
    run_command = run_command.replace('$alpo_finetuning_type','${alpo_finetuning_type}')
    run_command = run_command.replace('$gpu_num','${gpu_num}')
    if sft_proportion == 1.0:
        run_command = run_command.replace('_$sft_proportion', '')
    else:
        run_command = run_command.replace('$sft_proportion', '${sft_proportion}')
        run_command = run_command.replace('sft_default', 'sft_${sft_proportion}')
    with open(osp.join(dirname, 'train_command.sh'), 'w', encoding='utf-8') as file:
        file.write(run_command)
