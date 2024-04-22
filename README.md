# CausalLM

This repo shows how to finetune, test, and evaluate decoder-only large language models (such as Llama-2, Llama-3, Gemma, Mistral, etc.) for summarization.

## Install

Before running the scripts, please install the required packages.
```
pip install -r requirements.txt
```

## Dataset

Process and save your data into csv files. The example data files included in the `data` folder are processed from [DialogSum](https://github.com/cylnlp/dialogsum).

## Training

All the arguments including training hyperparameters can be changed in the `code/config.json` file. Then we can train the model simply by running
```
python code/train.py code/config.json
```
We use PEFT and LoRA for efficient fine-tuning on a quantized model then merging the adaptor into the full model. The minimum GPU requirement is NVIDIA A10 Tensor Core GPU.

## Testing

Once the fine-tuning is finished, we can find the fine-tuned model in the `output_merged_dir` that we defined in the config before. Then we can test the model by generating some examples 
```
python code/test.py \
    --model_name $fine-tuned_model_location \
    --test_file "data/dialogsum_test.csv" \
    --output_file "results/outputs.csv" \
    --max_target_length 256 \
    --custom_prompt "You are a help assistant who summarizes a dialogue within two sentences." \
    --text_column "dialogue"
```
To get the maximum performance, make sure the testing prompt is consistent with the training prompt. The saved output file will then be used for evaluation.

## Evaluation

The evaluation focuses on comparing the generated summary and the input texts, or comparing the generated summary and the gold summary.

The entailment evaluation metric `SCALE` is based on one of my recent papers [Fast and Accurate Factual Inconsistency Detection Over Long Documents](https://aclanthology.org/2023.emnlp-main.105.pdf). Compared to traditional evaluation metrics such as ROUGE scores, SCALE much better approximates the human evaluations of the actual task. The original SCALE scores were designed to accommodate long documents, here shows a simplified version focusing on the entitlement relationship between generated summaries and the input texts. By answering a Yes or No question, the resulting logits from `Flan-T5` are used to compute the entailment scores.

We also use sentence transformers to compute the precision and recall based similarity scores. For each summary sentence, we compute the cosine similarity between its embedding and all the reference sentence embeddings and then find the reference sentence that is most similar.

Evaluation metrics commonly used for summarization such as `ROUGE` scores and `BERT` scores are also included to compare the generated summary and the gold summary.

All of these scores can be obtained by one run
```
python code/evaluate_results.py \
	--data_path "results/llama-2-v1-fixed.csv" \
	--output_path "results/evaluation_scores.json" \
	--text_column "dialogue" \
	--summary_column "generated" \
	--label_column "summary"
```

## Results
For the outputs file included in the `results` folder, we should be able to reproduce the following scores
| Metric | Score |
| ------ | ------ |
| Entailment | 0.5279392205374581|
| Similarity_p | 0.6745579335689544|
| Similarity_r | 0.7298713611081952|
| BERTscore-f1 | 0.9264104007482529|
| BERTscore-p | 0.9273491790294647|
| BERTscore-r | 0.9256300050020217|
| rouge1 | 0.5036950022488493|
| rouge2 | 0.25161247879223264|
| rougeL | 0.42515062496930306|
| rougeLsum | 0.4252046790233571|
