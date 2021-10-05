import datasets
import argparse
from tqdm import tqdm
from konlpy.tag import Mecab
from transformers import *

bleu_metric = datasets.load_metric('bleu')
meteor_metric = datasets.load_metric('meteor')
rouge_metric = datasets.load_metric('rouge')

def coverage_score(preds, concept_sets, tokenizer):
    covs = []
    for p, cs in tqdm(zip(preds,concept_sets), total=len(preds)):
        cs = set(cs)
        lemmas = set()
        for token in tokenizer(p):
            lemmas.add(token)
        cov = len(lemmas&cs)/len(cs)
        covs.append(cov)
    return sum(covs)/len(covs)

def scoring(preds, concept_sets, tokenizer):
    Coverage = coverage_score(preds, concept_sets, tokenizer)
    print(f"System level Concept Coverage: {Coverage*100:.2f}")


def main(args):
    print("Start KommonGen Evaluation")
    concept_sets = []
    bleu_predictions = []
    bleu_references = []
    met_references = []
    met_predictions = []

    if args.model == "kobart":
        model_tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')

    elif args.model == "mbart-50":
        model_tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-50')

    elif args.model == "kogpt2":
        model_tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')  # Tokenizer download

    if args.model not in ['kobart', 'kogpt2', 'mbart-50']:
        raise ValueError("One of kogpg2, kobart, and mbart-50 must be selected.")
    
    mecab_tokenizer = Mecab().morphs

    with open(args.reference_file, 'r', encoding='utf-8') as f, open(args.prediction_file, 'r', encoding='utf-8') as f1:
        refs_list = f.readlines()
        preds_list = f1.readlines()
        for ref, prd in zip(refs_list, preds_list):
            concept_set = mecab_tokenizer(ref.split(' = ')[0].replace('[SOS] ', '').strip())
            concept_sets.append(concept_set)

            # For BLEU score
            bleu_reference = [mecab_tokenizer(ref.split(' = ')[1].replace(' [EOS]', '').strip())]
            bleu_references.append(bleu_reference)
            bleu_prediction = mecab_tokenizer(prd.strip())
            bleu_predictions.append(bleu_prediction)

            # For METEOR score
            met_reference = ref.split(' = ')[1].replace(' [EOS]', '').strip()
            met_prediction = prd.strip()
            met_predictions.append(met_prediction)
            met_references.append(met_reference)

        # For ROUGE score
        preds = [prd.strip() for prd in preds_list]
        refs = [ref.strip() for ref in refs_list]
        rouge_references = [' '.join(list(map( str, model_tokenizer(ref.split(' = ')[1].replace(' [EOS]', ''))['input_ids']))) for ref in refs]
        rouge_predictions = [' '.join(list(map(str, model_tokenizer(prd)['input_ids']))) for prd in preds]

        bleu_score = bleu_metric.compute(predictions = bleu_predictions, references = bleu_references, max_order = 4)
        print("BLEU 3: ", round(bleu_score['precisions'][2],4))
        print("BLEU 4: ", round(bleu_score['precisions'][3],4))

        meteor_score = meteor_metric.compute(predictions = met_predictions, references = met_references)
        print("METEOR: ", round(meteor_score['meteor'], 4))

        rouge_score = rouge_metric.compute(predictions = rouge_predictions, references = rouge_references, use_stemmer=True)
        print("ROUGE-2: ", rouge_score['rouge2'])
        print("ROUGE-L: ", rouge_score['rougeL'])

        scoring(preds, concept_sets, mecab_tokenizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--reference_file", type=str)
    parser.add_argument("--prediction_file", type=str)
    args = parser.parse_args()
    main(args)