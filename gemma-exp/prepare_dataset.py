from datasets import load_dataset
import json

dataset = load_dataset(
    "yahma/alpaca-cleaned"
)

data = dataset["train"]

data = data.shuffle(seed=42)

train_data = data.select(range(500))
eval_data = data.select(range(500,550))


def convert(example):
    return {
        "instruction": example["instruction"],
        "input": example["input"],
        "output": example["output"]
    }


train_data = [convert(x) for x in train_data]
eval_data = [convert(x) for x in eval_data]


with open(
    "/home/zhsu/jyluo/gemma-exp/data_v3/train.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)


with open(
    "/home/zhsu/jyluo/gemma-exp/data_v3/eval.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)


print("train:", len(train_data))
print("eval:", len(eval_data))
