from sys import argv
from os.path import exists
import random
from transformers import TrainingArguments, HfArgumentParser
import torch
from typing import Any, Tuple
from functions import (
    AdditionalArgs,
    run,
    setup_data_collator,
    setup_dataset,
    setup_model,
    setup_tokenizer,
    setup_trainer,
    create_bnb_config,
    get_max_length,
    merge_and_save,
    _save_args,
    test_predict,
)


def args() -> Tuple[Any, ...]:
    """
    Parse command line arguments for both the Trainer
    and the Additional Args. Attempts to interpret the
    first argument as a path to a JSON file. If so, read
    the args from that file instead, and ignore writing
    the arguments to another JSON file, if requested in
    the arguments.

    :return: TrainingArguments, AdditionalArguments
    """
    parser = HfArgumentParser([TrainingArguments, AdditionalArgs])

    # Attempt read from JSON if only one argument is provided
    if len(argv) == 2 and exists(argv[1]):
        print("Loading arguments from JSON...")
        return parser.parse_json_file(argv[1])

    # Only save args if we are not already loading from JSON
    _save_args(parser)

    return parser.parse_args_into_dataclasses()


def main(t_args: TrainingArguments, required_args: AdditionalArgs):
    print("Model Setup......")
    bnb_config = create_bnb_config()
    tokenizer = setup_tokenizer(required_args.model_name, required_args.use_auth_token)
    model = setup_model(required_args.model_name, bnb_config)
    max_length = get_max_length(model)
    collator = setup_data_collator(tokenizer)

    print("Training Data Setup......")
    train_dataset = setup_dataset(
        csv_specs=required_args.train_file,
        tokenizer=tokenizer,
        max_input_length=max_length - required_args.max_target_length,
        max_target_length=required_args.max_target_length,
        custom_prompt=required_args.custom_prompt,
        text_column=required_args.text_column,
        summary_column=required_args.summary_column,
    )

    print("Testing Data Setup......")
    test_dataset = setup_dataset(
        csv_specs=required_args.test_file,
        tokenizer=tokenizer,
        max_input_length=max_length - required_args.max_target_length,
        max_target_length=required_args.max_target_length,
        custom_prompt=required_args.custom_prompt,
        text_column=required_args.text_column,
        no_summary=True,
    )

    print("Start Training......")
    trainer = setup_trainer(model, tokenizer, t_args, train_dataset, collator)
    model.config.use_cache = False
    run(trainer, t_args.output_dir)

    print("From Quantized to Full......")
    merge_and_save(
        t_args.output_dir,
        required_args.output_merged_dir,
        t_args.bf16,
    )

    torch.cuda.empty_cache()
    if t_args.do_predict:
        print("Predict on Testing Set......")
        test_predict(
            required_args.output_merged_dir,
            test_dataset,
            required_args.max_target_length,
        )


if __name__ == "__main__":
    t_args, required_args = args()
    random.seed(t_args.seed)
    main(t_args, required_args)
