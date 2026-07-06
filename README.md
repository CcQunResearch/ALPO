# ALPO

Official code for **From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization**.

## Environment

Core dependencies:

```text
python >= 3.10
transformers == 4.41.2
vllm == 0.4.3
ray
pandas
pyyaml
```

Optional dependencies:

```text
deepspeed   # for ALPO ZeRO-3 training
peft        # only needed when alpo_finetuning_type: lora
```

Before running the pipeline, update the environment activation line in `train.sh` and `infer.sh`:

```shell
source [your_env_path]
```

You also need to configure your online LLM service credentials in `MainPipeline/chat.py` if you use online term recognition, preference labeling, or evaluation.

## Dataset

Download the MuSC dataset from:

[https://mega.nz/file/oIYxUKZC#jqIDxjuu1OU2jEcOihVKsuoU1aA3-FelL4dFppzAkyg](https://mega.nz/file/oIYxUKZC#jqIDxjuu1OU2jEcOihVKsuoU1aA3-FelL4dFppzAkyg)

After downloading, arrange the data under the repository root following this structure:

```text
ALPO
├─ Data
│  ├─ Label
│  └─ Source
│     └─ {lang}
│        ├─ source(train)
│        └─ source(test)
├─ LLaMAFactory
│  ├─ data
│  ├─ TermRecognition
│  ├─ VividnessAlignment
│  └─ YamlConfig
├─ Inference
├─ Log
└─ MainPipeline
```

The demo archive `MuSC Demo.tar.xz` is included only as a small example.

## Configuration

Edit `config.yaml` before running.

Important fields:

```yaml
lang: zh2vi
gpus: 0,1,2,3,4,5,6,7
model_path: /path/to/models

# Use ALPO by default. Set to dpo to run the old DPO/LLaMAFactory alignment path.
alignment_method: alpo

# ALPO requires sft_proportion < 1.0 so part of the training data is reserved for alignment.
sft_proportion: 0.8
```

## Training

Run:

```shell
bash train.sh
```

This first generates `train_command.sh` from `config.yaml`, then executes the numbered pipeline.

Training order:

```text
[0]index_data.py
[1]preprocess.py
[2-1]identify_proper_noun.py
[2-2]identify_proper_noun.py
[2-3]identify_proper_noun.py
[3-1]construct_dataset.py
[3-2]construct_sample_dataset.py
[3-3]construct_tr_dataset.py
[4]sample_inference.py
[5]label_preference.py
[6]construct_alpo_dataset.py
[7]alpo_train.py
```

If `alignment_method: dpo`, the alignment step uses:

```text
[6-dpo]construct_dpo_dataset.py
```

instead of `[6]construct_alpo_dataset.py` and `[7]alpo_train.py`.

## Inference and Evaluation

Run:

```shell
bash infer.sh
```

Inference order:

```text
[8-1]extract_test_data.py
[8-2]extract_test_data.py
[8-3]extract_test_data.py
[8-4]extract_test_data.py
[9]prediction_2_subtitle.py
[10]evaluate.py
```

Generated predictions and evaluation results are saved under:

```text
Inference/{lang}/
```

## License

The code in this repository and the MuSC dataset are licensed separately.

- **MuSC dataset**: released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See [`DATA_LICENSE`](DATA_LICENSE).
- **Citation**: If you use the MuSC dataset or this repository, please cite our paper.

## Citation

If you use this code or dataset, please cite our paper:

```bibtex
@article{cui2026utterance,
  title={From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization},
  author={Cui, Chaoqun and Wang, Shijing and Huang, Liangbin and Gu, Qingqing and Huang, Zhaolong and Zeng, Xiao and Mao, Wenji},
  journal={arXiv preprint arXiv:2602.01068},
  year={2026}
}
```
