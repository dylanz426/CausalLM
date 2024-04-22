import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import HfArgumentParser, T5ForConditionalGeneration, T5Tokenizer


@dataclass
class Arguments:
    """
    Arguments for evaluating generated summaries.
    """

    data_path: str = field(
        metadata={
            "help": "Path to the evaluation data in a csv file with text, summary, and (or) label columns",
            "required": True,
        },
    )

    output_path: str = field(
        metadata={
            "help": "Path to the outputs in a json file",
            "required": True,
        },
    )

    flant5_path: str = field(
        default="google/flan-t5-large",
        metadata={
            "help": "Name/Path of the entailment evaluation model",
            "required": False,
        },
    )

    sent_trans: str = field(
        default="all-mpnet-base-v2",
        metadata={
            "help": "Name/Path of the similarity evaluation model",
            "required": False,
        },
    )

    bert_path: str = field(
        default="roberta-large",
        metadata={
            "help": "Name/Path of the bertscore evaluation model",
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

    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of column containing generated summary text from the summarization model",
            "required": False,
        },
    )

    label_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of column containing gold summary labels",
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


def process_text(text: str) -> List[str]:
    """
    Split a string into sentences.

    :param text: the full string

    :return list of sentences split from the string
    """

    text = (
        text.replace("\xa0", " ")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\n", "  ")
        .replace("\t", "  ")
    )

    text = re.sub("\s{2,}", "  ", text)
    text = re.sub(r"\.+", ".", text)

    text = (
        text.replace(":  ", ": ")
        .replace(". ", ".***")
        .replace("? ", "?***")
        .replace("! ", "!***")
        .replace("  ", "***")
    )
    return [s.strip() for s in text.split("***") if len(s.strip()) > 5]


def compute_entailment_score(
    model_path: str,
    references: List[List[str]],
    queries: List[List[str]],
) -> List[float]:
    """
    Compute the SCALE scores to evaluate the generated answers.

    :param model_path: Name/Path of the entailment evaluation model
    :param references: list of reference sentences
    :param queries: list of target sentences

    :return the list of entailment scores
    """
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yes_no_tokens = [tokenizer("Yes").input_ids[0], tokenizer("No").input_ids[0]]
    results = []
    for reference, query in tqdm(zip(references, queries), total=len(references)):
        if len(reference) == 0 or len(query) == 0:
            results.append(0)
            continue
        result = []
        for h in query:
            sub_result = []
            for p in reference:
                prompt = f'{p} Question: Does this imply that "{h}"? Yes or No?'
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
                outputs = model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1,
                )
                scores = outputs["scores"][0][0][yes_no_tokens]
                sub_result.append(torch.nn.functional.softmax(scores, dim=0)[0].item())
            result.append(max(sub_result))
        results.append(sum(result) / len(result))

    return results


def compute_similarity_score(
    model_path: str,
    references: List[List[str]],
    queries: List[List[str]],
) -> List[float]:
    """
    Compute the similarity scores to evaluate the generated answers.

    :param model_path: Name/Path of the similarity evaluation model
    :param references: list of reference sentences
    :param queries: list of target sentences

    :return the list of similarity scores
    """
    sent_trans_model = SentenceTransformer(model_path)

    results = []
    for reference, query in tqdm(zip(references, queries), total=len(references)):
        if len(reference) == 0 or len(query) == 0:
            results.append(0)
            continue
        ref_embeddings = sent_trans_model.encode(reference)
        qry_embeddings = sent_trans_model.encode(query)
        preds_dict = util.semantic_search(qry_embeddings, ref_embeddings, top_k=1)
        result = []
        for i in range(len(preds_dict)):
            dic = preds_dict[i][0]
            result.append(dic["score"])
        results.append(sum(result) / len(result))

    return results


def evaluate_loaded(
    model_path: str, predictions: List[str], references: List[str]
) -> Dict:
    """
    Compute the similarity scores to evaluate the generated answers.

    :param model_path: Name/Path of the bert score evaluation model
    :param predictions: list of predicted summaries
    :param references: list of reference summaries

    :return lists of ROUGE scores and BERT scores in a dictionary
    """
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    results = rouge.compute(
        predictions=predictions, references=references, use_aggregator=False
    )
    b_results = bertscore.compute(
        predictions=predictions, references=references, lang="en", model_type=model_path
    )
    results["BERTscore-p"] = b_results["precision"]
    results["BERTscore-r"] = b_results["recall"]
    results["BERTscore-f1"] = b_results["f1"]
    return results


def main(args: Arguments):
    df = pd.read_csv(args.data_path)
    outputs = df[args.summary_column]
    output_sents = [process_text(text) for text in outputs]

    scores = {}
    if args.summary_column:
        inputs = df[args.text_column]
        input_sents = [process_text(text) for text in inputs]
        print("Computing entailment scores......")
        scores["Entailment"] = compute_entailment_score(
            args.flant5_path, input_sents, output_sents
        )
        print("Computing similarity-precision scores......")
        scores["Similarity_p"] = compute_similarity_score(
            args.sent_trans, inputs, outputs
        )

    if args.label_column:
        labels = df[args.label_column]
        label_sents = [process_text(text) for text in labels]
        print("Computing similarity-recall scores......")
        scores["Similarity_r"] = compute_similarity_score(
            args.sent_trans, output_sents, label_sents
        )
        print("Computing ROUGE and BERT scores......")
        scores.update(evaluate_loaded(args.bert_path, outputs, labels))

    with open(args.output_path, "w") as f:
        json.dump(scores, f)

    for key in sorted(scores.keys()):
        if len(scores[key]) > 0:
            print(key, sum(scores[key]) / len(scores[key]))
        else:
            print(key, 0)


if __name__ == "__main__":
    args = get_args()
    main(args[0])
