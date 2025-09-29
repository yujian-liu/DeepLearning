import random
import torch
from datasets import load_dataset, load_from_disk
from matplotlib import pyplot as plt
from torch import no_grad
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        dataset = load_from_disk(f'./data/ChnSentiCorp/{split}')

        def f(data):
            return len(data['text']) > 40
        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']

        # 将句子分为前后两句
        sents1 = text[:20]
        sents2 = text[20:40]
        label = 0

        # 一半概率替换后半句
        if random.randint(0, 1) == 0:
            j = random.randint(0, len(self.dataset) - 1)
            sents2 = self.dataset[j]['text'][20:40]
            label = 1

        return sents1, sents2, label

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def collate_fn(data):
    sents = [i[:2] for i in data]
    labels = [i[2] for i in data]

    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        return_tensors='pt',
        padding='max_length',
        max_length=45,
        add_special_tokens=True,
        return_length=True
    )

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = BertModel.from_pretrained('bert-base-chinese')
        for param in self.pretrained.parameters():
            param.requires_grad_(False)

        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with no_grad():
            out = self.pretrained(input_ids, attention_mask, token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        return out

# 训练
def train(device, epoches, learning_rate):
    dataset = Dataset('train')
    losses = []
    accuracies = []

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    model = Model().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epoches):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            out = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)
                losses.append(loss.item())
                accuracies.append(accuracy)
                print(epoch, i, loss.item(), accuracy)

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('loss')
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('accuracy')
    plt.show()

    return model

def test(model, device):
    model.eval()
    correct = 0
    total = 0
    dataset = Dataset('test')

    loader_test = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(correct / total)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = train(device, epoches=5, learning_rate=1e-5)
    test(model, device)