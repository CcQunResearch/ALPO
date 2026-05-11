from prompt_template import llm_template, stop_token_dict, stop_token_id_dict, template_dict


DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'


def get_legacy_template_name(model_name):
    if model_name not in template_dict:
        raise ValueError(f'Unsupported model template: {model_name}')
    return template_dict[model_name]


def apply_model_chat_template(tokenizer, instruction, model_name=None, system_prompt=DEFAULT_SYSTEM_PROMPT):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': instruction},
    ]
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if model_name is None:
        return instruction
    template_name = get_legacy_template_name(model_name)
    if template_name not in llm_template:
        raise ValueError(f'Unsupported legacy template: {template_name}')
    return llm_template[template_name].format(instruction)


def get_generation_stop_token_id(tokenizer, model_name=None):
    if getattr(tokenizer, 'eos_token_id', None) is not None:
        return tokenizer.eos_token_id
    if model_name is None:
        return None
    return stop_token_id_dict.get(get_legacy_template_name(model_name))


def get_generation_stop_text(tokenizer, model_name=None):
    if getattr(tokenizer, 'eos_token', None):
        return tokenizer.eos_token
    if model_name is None:
        return ''
    return stop_token_dict.get(get_legacy_template_name(model_name), '')
