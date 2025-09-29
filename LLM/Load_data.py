from datasets import load_dataset

dataset = load_dataset("lansinuote/ChnSentiCorp", split="train", trust_remote_code=True)

# print(dataset)
# print(dataset[0])

# 排序
print(dataset['label'][:10])

sorted_dataset = dataset.sort('label')
print(sorted_dataset['label'][:10])
print(sorted_dataset['label'][-10:])

# 打乱
shuffled_dataset = sorted_dataset.shuffle(seed=42)
print(shuffled_dataset['label'][:10])

# 选择
print(dataset.select([0, 10, 20, 30, 40, 50]))

# 过滤
def my_filter(data):
    return data['text'].startswith('选择')

start_with_ar = dataset.filter(my_filter)
print(len(start_with_ar))
print(start_with_ar['text'][0])
print(start_with_ar['text'][1])

# 切分
dataset_split = dataset.train_test_split(test_size=0.1)
print(len(dataset_split['train']))
print(len(dataset_split['test']))

dataset_shard = dataset.shard(num_shards=4, index=0)
print(dataset_shard)

dataset_rename = dataset.rename_column('text', 'textA')
print(dataset_rename)

dataset_remove = dataset.remove_columns('text')
print(dataset_remove)

# 改变dataset[i]的返回，仅返回columns内的数据
dataset_format = dataset.set_format(type='torch', columns=['label', 'text'])
print(dataset, dataset[0], dataset[1])

dataset.reset_format()

# map
def my_map(data):
    data['text'] = 'My sentence: ' + data['text']
    return data

dataset_map = dataset.map(my_map)
print(dataset_map['text'][:5])

# 保存加载
from datasets import load_from_disk

# dataset.save_to_disk('./data/ChnSentiCorp_train')
# dataset.to_csv('./data/ChnSentiCorp_train.csv')
# dataset.to_json('./data/ChnSentiCorp_train.json')
print(load_from_disk('./data/ChnSentiCorp_train'))