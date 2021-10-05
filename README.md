# KommonGen
한국어 생성 모델의 상식 추론을 위한 KommonGen 데이터셋입니다.

본 데이터셋은 한글 및 한국어 2021년 **KommonGen: 한국어 생성 모델의 상식 추론 평가 데이터셋** 논문의 내용을 기반으로 합니다.

## Requirements

Mecab의 경우 각자 OS에 맞게 설치해야합니다. 

```
python 3.7
konlpy 0.5.2
mecab-python 0.996-ko-0.9.2
transformers 4.5.1
nltk 3.6.2
tqdm 4.6.2
datasets 1.12.1
rouge-score 0.0.4
sentencepiece 0.1.96
```
## Results

Model | ROUGE-2 | ROUGE-L | BLEU 3 | BLEU 4 | METEOR | Coverage
---- | ---- | ---- | ---- | ---- | ---- | ---- 
**KoGPT2** | 34.78 | 53.02 | 27.31 | 17.78 | 35.48 | 77.04
**KoBART** | 44.53 | 60.64 | 35.24 | 25.27 | 45.03 | 89.80
**mBART-50** | **64.95** | **73.90** | **39.74** | **28.72** | **47.68** | **94.06**

## Usage

**kogpt2/kobart/mbart-50**

```
python kommongen_evaluation.py --model kogpt2 --reference_file examples/kommongen_test_1.1.txt --prediction_file examples/kogpt2.txt
python kommongen_evaluation.py --model kobart --reference_file examples/kommongen_test_1.1.txt --prediction_file examples/kobart.txt
python kommongen_evaluation.py --model mbart-50 --reference_file examples/kommongen_test_1.1.txt --prediction_file examples/mbart-50.txt
```

## Citation
```
@inproceedings{jay2021kommongen,
  title={KommonGen: A Dataset for Korean Generative Commonsense Reasoning Evaluation},
  author={Jaehyung Seo, Chanjun Park, Hyeonseok Moon, Sugyeong Eo, Myunghoon Kang, Seounghoon Lee, and Heuiseok Lim},
  booktitle={Proceedings of the 33th Annual Conference on Human & Cognitive Language Technology},
  affilation={Korea University, NLP & AI},
  month={October},
  year={2021}
}
```

## Acknowledgments

Our dataset is based on [CommonGen](https://github.com/INK-USC/CommonGen "CommonGen"). We thank the authors for their academic contribution to us the opportunity to do this research.
