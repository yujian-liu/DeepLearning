import evaluate
import numpy as np
from transformers import EvalPrediction

metric = evaluate.load('glue','mrpc')

# print(metric.inputs_description)

predictions = [0, 1, 0]
references = [0, 1, 1]
final_score = metric.compute(predictions=predictions, references=references)
print(final_score)


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.argmax(axis=1)
    return metric.compute(predictions=logits, references=labels)

eval_pred = EvalPrediction(
    predictions=np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
    label_ids=np.array([1, 1, 1, 1]),
)

print(compute_metrics(eval_pred))