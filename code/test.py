from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union

import pandas as pd
import torch
from functions import get_max_length, setup_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser


@dataclass
class Arguments:
    """
    Arguments for loading data and model.
    """

    model_name: str = field(
        metadata={
            "help": "Path to model directory or Huggingface model name",
            "required": True,
        },
    )

    test_file: Union[str, List[str]] = field(
        metadata={
            "help": "Path to CSV file(s) containing --TEXT_col for testing",
            "required": True,
        },
    )

    output_file: str = field(
        default=None,
        metadata={
            "help": "Save outputs to the specified CSV file",
            "required": False,
        },
    )

    chat: bool = field(
        default=False,
        metadata={
            "help": "Use chat mode or not.",
            "required": False,
        },
    )

    max_target_length: int = field(
        default=512,
        metadata={
            "help": "Maximum length of a newly generated string",
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

    text_column: str = field(
        default="TEXT",
        metadata={
            "help": "Name of column containing input text to the summarization model",
            "required": False,
        },
    )


def get_args() -> Tuple[Any, ...]:
    """
    Parse command line arguments.

    :return: Arguments
    """
    parser = HfArgumentParser(Arguments)

    return parser.parse_args_into_dataclasses()


def main(args: Arguments):
    print("Model Setup......")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_auth_token=args.use_auth_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto"
    )
    max_length = get_max_length(model)

    print("Testing Data Setup......")
    test_dataset = setup_dataset(
        csv_specs=[args.test_file[0]],
        tokenizer=tokenizer,
        max_input_length=max_length - args.max_target_length,
        max_target_length=args.max_target_length,
        custom_prompt=args.custom_prompt,
        chat=args.chat,
        text_column=args.text_column,
        no_summary=True,
    )

    print("Start Testing......")
    generations = []
    for data in tqdm(test_dataset, total=len(test_dataset)):
        input_length = len(data["inputs"])
        output_tokens = model.generate(
            torch.tensor([data["input_ids"]]),
            do_sample=True,
            max_new_tokens=args.max_target_length,
        )
        output_str = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        summary = output_str[0][input_length:]
        generations.append(summary)

    if args.output_file:
        df = pd.read_csv(args.test_file[0])
        df["generated"] = generations
        df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    args = get_args()
    main(args[0])
