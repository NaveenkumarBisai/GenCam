# modules/eval/scorer.py

import evaluate
from nltk.tokenize import word_tokenize

bleu = evaluate.load("bleu")
bert = evaluate.load("bertscore")

def compute_bleu(pred, ref):
    pred = pred.lower().strip()
    ref = ref.lower().strip()
    return bleu.compute(predictions=[pred], references=[[ref]])["bleu"]

def compute_bertscore(pred, ref):
    result = bert.compute(predictions=[pred], references=[ref], lang="en")
    return result["f1"][0]
