import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Union

import bitsandbytes as bnb
import torch
from datasets import Dataset, load_dataset
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class AdditionalArgs:
    """
    Additional arguments for loading data and
    output control.
    """

    model_name: str = field(
        metadata={
            "help": "Path to model directory or Huggingface model name",
            "required": True,
        },
    )

    train_file: Union[str, List[str]] = field(
        metadata={
            "help": "Path to CSV file(s) containing --TEXT_col and --SUMMARY_col for training",
            "required": True,
        },
    )

    test_file: Union[str, List[str]] = field(
        metadata={
            "help": "Path to CSV file(s) containing --TEXT_col for testing",
            "required": True,
        },
    )

    output_merged_dir: str = field(
        metadata={
            "help": "Path to the trained full model",
            "required": True,
        },
    )

    max_target_length: int = field(
        default=512,
        metadata={
            "help": "Maximum length of a newly generated string",
            "required": False,
        },
    )

    text_column: str = field(
        default="TEXT",
        metadata={
            "help": "Name of column containing input text to the summarization model",
            "required": False,
        },
    )

    summary_column: str = field(
        default="SUMMARY",
        metadata={
            "help": "Name of column containing target summary text to the summarization model",
            "required": False,
        },
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Use Huggingface auth token or not. Turn False if running offline",
            "required": False,
        },
    )

    custom_prompt: str = field(
        default=None,
        metadata={
            "help": "Prompt that is concatenated together with the input for training and inference",
            "required": False,
        },
    )

    output_args: str = field(
        default=None,
        metadata={
            "help": "Save all current arguments to the specified JSON file",
            "required": False,
        },
    )


def setup_model(model_name: str, bnb_config: BitsAndBytesConfig) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    return model


def setup_tokenizer(model_name: str, use_auth_token: bool) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def setup_data_collator(
    tokenizer: PreTrainedTokenizer,
) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # type: ignore
    )


def _save_args(parser: HfArgumentParser):
    all_args = parser.parse_args()
    output_json = all_args.output_args

    if output_json is None:
        return

    with open(output_json, "w") as fp:
        json.dump(vars(all_args), fp, sort_keys=True, indent=4)


def _tokenize_input(
    e: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_length: int
) -> BatchEncoding:
    return tokenizer(
        e["inputs"],
        truncation=True,
        max_length=max_length,
    )


def process_data(
    dataset: Dataset,
    custom_prompt: str,
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    text_column: str = "TEXT",
    summary_column: str = "SUMMARY",
    no_summary: bool = False,
) -> Dataset:
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{custom_prompt}"
    end = f"{END_KEY}"

    dic = {"inputs": []}

    for row in dataset:
        text = row[text_column]
        context = f"{INPUT_KEY}\n{text}"
        parts = [part for part in [blurb, instruction, context] if part]
        inputs = "\n\n".join(parts)
        inputs_encoded = tokenizer.encode(
            inputs, truncation=True, max_length=max_input_length
        )
        inputs = tokenizer.decode(inputs_encoded, skip_special_tokens=True)
        if no_summary:
            inputs = inputs + "\n\n" + RESPONSE_KEY + "\n"
        else:
            summary = row[summary_column]
            response = f"{RESPONSE_KEY}\n{summary}"
            parts = [part for part in [inputs, response, end] if part]
            inputs = "\n\n".join(parts)
        dic["inputs"].append(inputs)
    return Dataset.from_dict(dic)


def setup_dataset(
    csv_specs: List[str],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_target_length: int,
    custom_prompt: str,
    text_column: str = "TEXT",
    summary_column: str = "SUMMARY",
    no_summary: bool = False,
) -> Dataset:
    dataset = load_dataset("csv", data_files=csv_specs)
    tok = partial(
        _tokenize_input,
        tokenizer=tokenizer,
        max_length=max_input_length + max_target_length,
    )
    d = process_data(
        dataset["train"],
        custom_prompt,
        tokenizer,
        max_input_length,
        text_column,
        summary_column,
        no_summary,
    )
    d = d.map(tok, batched=True)
    return d


def get_max_length(model: PreTrainedModel) -> int:
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def create_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def create_peft_config(modules: List[str]) -> LoraConfig:
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    return LoraConfig(
        r=8,  # dimension of the updated matrices
        lora_alpha=16,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.05,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )


def find_all_linear_names(model: PreTrainedModel) -> List[str]:
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def setup_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    trainer_args: TrainingArguments,
    train_dataset: Dataset,
    data_collator: DataCollatorForLanguageModeling,
) -> Trainer:
    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()
    # Use the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)
    # Get lora module names
    modules = find_all_linear_names(model)
    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)
    return Trainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


def run(trainer: Trainer, output_dir: str) -> None:
    trainer.train()  # type: ignore
    trainer.model.save_pretrained(output_dir)  # type: ignore
    trainer.tokenizer.save_pretrained(output_dir)  # type: ignore


def merge_and_save(
    output_dir: str,
    output_merged_dir: str,
    use_bfloat: bool,
) -> None:
    dtype = torch.bfloat16 if use_bfloat else torch.float16
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir, device_map="auto", torch_dtype=dtype
    )
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_merged_dir)


def test_predict(
    output_merged_dir: str,
    test_dataset: Dataset,
    max_target_length: int = 512,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(output_merged_dir)
    model = AutoModelForCausalLM.from_pretrained(
        output_merged_dir,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    output_test_preds_file_path = os.path.join(
        output_merged_dir, "test_generations.txt"
    )
    generations = []
    for data in tqdm(test_dataset, total=len(test_dataset)):
        output = model.generate(
            torch.tensor([data["input_ids"]]),
            do_sample=True,
            max_new_tokens=max_target_length,
        )
        output_str = tokenizer.batch_decode(output, skip_special_tokens=True)
        generations.append(output_str[0])
        with open(output_test_preds_file_path, "w") as f:
            f.write("\n\n".join(generations))
