import torch
from datasets import load_dataset, load_from_disk
from matplotlib import pyplot as plt
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        dataset = load_from_disk(f'./data/ChnSentiCorp/{split}')

        def f(data):
            return len(data['text']) > 30
        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        return text

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def collate_fn(data):
    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=data,
        truncation=True,
        padding='max_length',
        max_length=30,
        return_tensors='pt',
        return_length=True
    )

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']

    # 用mask替换每一句的第15个词
    labels = input_ids[:, 15].reshape(-1).clone()
    input_ids[:, 15] = tokenizer.get_vocab()[tokenizer.mask_token]

    return input_ids, attention_mask, token_type_ids, labels

# 下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = BertModel.from_pretrained('bert-base-chinese')
        for param in self.pretrained.parameters():
            param.requires_grad_(False)

        self.decoder = torch.nn.Linear(768, tokenizer.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(tokenizer.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids, attention_mask, token_type_ids)
        out = self.decoder(out.last_hidden_state[:, 15])
        return out

# 训练
def train(device, epoches):
    dataset = Dataset('train')
    losses = []
    accuracies = []

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=16,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    # for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    #     break
    # print(tokenizer.decode(input_ids[0]))
    # print(tokenizer.decode(labels[0]))

    model = Model().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4)
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

            if i % 100 == 0:
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

# 测试
def test(model, device):
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(
        dataset=Dataset(split='test'),
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
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

        print(tokenizer.decode(input_ids[0]))
        print(tokenizer.decode(out[0]), tokenizer.decode(labels[0]))

    print(correct / total)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train(device, epoches=5)
    test(model, device)
