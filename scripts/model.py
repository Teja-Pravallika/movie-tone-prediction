from imports import *

max_token_length = 256
train_batch_size = 16
valid_batch_size = 16
learning_rate = 2e-05
seed = 42

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

class EmotionalToneData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, tone_to_num):
        self.tokenizer = tokenizer
        self.tone_to_num = tone_to_num
        self.data = dataframe
        self.text = dataframe.overview
        self.targets = dataframe.emotional_tones
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        tone_labels = [self.tone_to_num.get(tone_str, None) for tone_str in self.targets[index].split(", ")]
        tone_labels = [tl for tl in tone_labels if tl is not None]
        bce_labels = [0] * len(self.tone_to_num)
        for tl in tone_labels:
            bce_labels[tl] = 1

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(bce_labels, dtype=torch.float)
        }


def create_dataloaders(dataframe, tokenizer, max_len, tone_to_num, train_size=0.8):
    train_data = dataframe.sample(frac=train_size, random_state=seed)
    test_data = dataframe.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    print(f"FULL Dataset: {dataframe.shape}")
    print(f"TRAIN Dataset: {train_data.shape}")
    print(f"TEST Dataset: {test_data.shape}")

    training_set = EmotionalToneData(train_data, tokenizer, max_len, tone_to_num)
    testing_set = EmotionalToneData(test_data, tokenizer, max_len, tone_to_num)

    def seed_worker(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

    train_params = {
        'batch_size': train_batch_size,
        'shuffle': True,
        'num_workers': 0,
        'worker_init_fn': seed_worker
    }
    test_params = {
        'batch_size': valid_batch_size,
        'shuffle': False,
        'num_workers': 0,
        'worker_init_fn': seed_worker
    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, 29)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def initialize_model():
    model = RobertaClass()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    return model, loss_function, optimizer, device
