import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def preprocess_gsm8k(tokenizer, max_length=2048):
    train_dataset = []

    data = []
    file_path = 'data/gsm8k/train.txt'
    with open(file_path, 'r') as f:
        for line in f:
            data.append(line)

    for i in range(len(data)):
        d = data[i]

        if len(d.split('||')) != 2:
            continue
        if len(d.split('||')[1].split('####')) != 2:
            continue

        question, thought, answer = d.split('||')[0], d.split('||')[1].split('####')[0], d.split('####')[1]
        question = 'Question: ' + question
        thought = 'Answer: ' + thought
        answer = '####' + answer

        question = tokenizer(question, return_tensors="pt")['input_ids'][0]
        thought = tokenizer(thought, return_tensors="pt")['input_ids'][0]
        answer = tokenizer(answer, return_tensors="pt")['input_ids'][0]
        answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        length1 = question.shape[-1] + thought.shape[-1]
        length2 = length1 + answer.shape[-1]
        if length2 > max_length:
            # exclude prompts that are too long
            continue

        padding_length = 2048 - length1
        padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
        padded_data = torch.cat((question, thought, padding), dim=-1)
        train_dataset.append(dict(data=padded_data, input_length=torch.tensor(question.shape[-1]),
                                  length=torch.tensor(length1)))


        padding_length = 2048 - (question.shape[-1] + thought.shape[-1] + answer.shape[-1])
        padding = torch.full((padding_length,), tokenizer.eos_token_id, dtype=question.dtype)
        padded_data = torch.cat((question, thought, answer, padding), dim=-1)
        train_dataset.append(dict(data=padded_data, input_length=torch.tensor(length1),
                                  length=torch.tensor(length2)))


    train_dataset = CustomDataset(train_dataset)
    return train_dataset


def preprocess_gsm8k_prefix_infill_question(tokenizer, max_length=2048, pad_to_length=2048):
    train_dataset = []

    file_path = 'data/gsm8k/train.txt'
    with open(file_path, 'r') as f:
        for line in f:
            if "||" not in line or "####" not in line:
                continue
            question_part, rest = line.split("||", 1)
            thought_part, answer_part = rest.split("####", 1)

            question = "Question: " + question_part.strip()
            thought = "Answer: " + thought_part.strip()
            answer = "####" + answer_part.strip()

            question_ids = tokenizer(question, return_tensors="pt")["input_ids"][0]
            thought_ids = tokenizer(thought, return_tensors="pt")["input_ids"][0]
            answer_ids = tokenizer(answer, return_tensors="pt")["input_ids"][0]
            answer_ids = torch.cat((answer_ids, torch.tensor([tokenizer.eos_token_id])), dim=-1)

            full_ids = torch.cat((question_ids, thought_ids, answer_ids), dim=-1)
            if full_ids.numel() > max_length:
                full_ids = full_ids[:max_length]
            if question_ids.numel() > pad_to_length:
                continue
            used_len = min(full_ids.numel(), pad_to_length)
            full_ids = full_ids[:used_len]
            if question_ids.numel() >= full_ids.numel():
                continue

            padding = torch.full(
                (pad_to_length - full_ids.numel(),), tokenizer.eos_token_id, dtype=full_ids.dtype
            )
            padded_ids = torch.cat((full_ids, padding), dim=-1)

            condition_mask = torch.zeros((pad_to_length,), dtype=torch.bool)
            condition_mask[question_ids.numel() : full_ids.numel()] = True

            loss_mask = torch.zeros((pad_to_length,), dtype=torch.bool)
            loss_mask[: question_ids.numel()] = True

            train_dataset.append(
                dict(
                    data=padded_ids,
                    condition_mask=condition_mask,
                    loss_mask=loss_mask,
                    length=torch.tensor(full_ids.numel()),
                )
            )

    return CustomDataset(train_dataset)
