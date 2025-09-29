import evaluate
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 分词
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# print(tokenizer)

def tokenizer_function(data):
    return tokenizer(
        data['sentence'],
        padding='max_length',
        truncation=True,
        max_length=30,
    )

# 评价函数
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.argmax(axis=1)
    return metric.compute(predictions=logits, references=labels)


if __name__ == '__main__':
    # datasets = load_dataset('glue', 'sst2')
    # datasets.save_to_disk('./data/glue_sst2')
    datasets = load_from_disk('./data/glue_sst2')

    # 数据处理
    datasets = datasets.map(tokenizer_function, batched=True, batch_size=1000, num_proc=4)

    dataset_train = datasets['train'].shuffle().select(range(1000))
    dataset_test = datasets['validation'].shuffle().select(range(200))

    # 释放内存
    del datasets

    # print(dataset_train)

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    # 参数总量（万）
    # print(sum([i.nelement() for i in model.parameters()]) / 10000)

    # 训练参数
    args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        eval_strategy='epoch',
        num_train_epochs=1,
        learning_rate=1e-4,
        weight_decay=1e-2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
    )

    # # 评估
    # print(trainer.evaluate())
    #
    # # 训练
    # trainer.train()
    # print(trainer.evaluate())
    #
    # # 保存模型
    # trainer.save_model(output_dir="./output")


    def collate_fn(data):
        label = [i['label'] for i in data]
        input_ids = [i['input_ids'] for i in data]
        token_type_ids = [i['token_type_ids'] for i in data]
        attention_mask = [i['attention_mask'] for i in data]

        label = torch.LongTensor(label)
        input_ids = torch.LongTensor(input_ids)
        token_type_ids = torch.LongTensor(token_type_ids)
        attention_mask = torch.LongTensor(attention_mask)

        return label, input_ids, token_type_ids, attention_mask

    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=4,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    for i, (label, input_ids, token_type_ids, attention_mask) in enumerate(loader_test):
        break

    def test():
        model = AutoModelForSequenceClassification.from_pretrained('./output')

        model.eval()

        out = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        out = out['logits'].argmax(dim=1)
        correct = (out == label).sum().item()
        return correct / len(label)

    print(test())