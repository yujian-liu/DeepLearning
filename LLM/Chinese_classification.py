import torch
from datasets import load_dataset, load_from_disk
from matplotlib import pyplot as plt
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        # self.dataset = load_dataset(path='lansinuote/ChnSentiCorp', split=split)
        self.dataset = load_from_disk(f'./data/ChnSentiCorp/{split}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label

# dataset = Dataset('train')
# print(len(dataset))
# print(dataset[0])

# dataset.dataset.save_to_disk('./data/ChnSentiCorp')

# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# print(tokenizer)

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding='max_length',
        max_length=500,
        return_tensors='pt',
        return_length=True
    )

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels

# 下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # last_hidden_state.shape = [16, 30, 768]
        self.pretrained = BertModel.from_pretrained('bert-base-chinese')

        for param in self.pretrained.parameters():
            param.requires_grad_(False)

        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        print(out.last_hidden_state.size())
        out = self.fc(out.last_hidden_state[:, 0])
        # out = out.softmax(dim=1)
        return out

def test():
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(
        dataset=Dataset(split='validation'),
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
    dataset = Dataset('train')
    epoches = 3

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=16,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    model = Model().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练
    model.train()
    losses = []
    accuracies = []
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
    plt.tight_layout()
    plt.show()

    test()
