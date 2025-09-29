from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷，真有点郁闷。',
    '机器背面似乎被撕了张什么标签，残胶还在。'
]

# 编码两个句子
out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],

    # 长度大于max_length时截断
    truncation=True,

    max_length=30,
    padding='max_length',
    return_tensors=None,
    add_special_tokens=True,
)

# print(out)
# print(tokenizer.decode(out))

# 增强的编码函数
out_plus = tokenizer.encode_plus(
    text=sents[0],
    text_pair=sents[1],

    truncation=True,
    add_special_tokens=True,
    max_length=30,
    padding='max_length',

    # 返回类型，可取值tf,pt,np,默认返回list
    return_tensors=None,
    # 第一个句子和特殊符号的位置是0，第二个句子位置是1
    return_token_type_ids=True,
    # pad位置为0，其他位置是1
    return_attention_mask=True,
    # 特殊符号为1，其他为0
    return_special_tokens_mask=True,
    # 返回句子长度
    return_length=True,
)

# for k, v in out_plus.items():
#     print(k, ':',v)
#
# print(tokenizer.decode(out_plus['input_ids']))

out_batch = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],
    add_special_tokens=True,
    truncation=True,
    padding='max_length',
    max_length=15,
    return_tensors=None,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True,
)

# for k, v in out_batch.items():
#     print(k, ':',v)
#
# print(tokenizer.decode(out_batch['input_ids'][0]),
#       tokenizer.decode(out_batch['input_ids'][1]))

# 批量编码成对的句子
out_pair_batch = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
    add_special_tokens=True,
    truncation=True,
    padding='max_length',
    max_length=30,
    return_tensors=None,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True,
)

# for k, v in out_pair_batch.items():
#     print(k, ':',v)
#
# print(tokenizer.decode(out_pair_batch['input_ids'][0]))

out_new_pre = tokenizer.encode(
    text='月光的新希望[EOS]',
    text_pair=None,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    max_length=10,
    return_tensors=None
)

print(out_new_pre)
print(tokenizer.decode(out_new_pre))

# 获取字典
vocab = tokenizer.get_vocab()

print(type(vocab), len(vocab), '月光' in vocab)

# 添加新词
tokenizer.add_tokens(new_tokens=['月光', '希望'])
tokenizer.add_special_tokens({'eos_token': '[EOS]'})

vocab = tokenizer.get_vocab()

print(type(vocab), len(vocab), vocab['月光'], vocab['[EOS]'], vocab.get(21131))

out_new = tokenizer.encode(
    text='月光的新希望[EOS]',
    text_pair=None,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    max_length=8,
    return_tensors=None
)

print(out_new)
print(tokenizer.decode(out_new))