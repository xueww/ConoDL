import torch
from tokenizers import Tokenizer
from datasets import load_dataset
from models.modeling_progen import ProGenForCausalLM
from transformers import Trainer,DataCollatorForLanguageModeling,TrainingArguments,PreTrainedTokenizerFast,TrainerCallback
from torch.nn.utils import clip_grad_norm_

device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ProGenForCausalLM.from_pretrained('ProGen_checkpoints', torch_dtype=torch.float32).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(num_params)

def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

tokenizer = create_tokenizer_custom(file='tokenizer.json')

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

fast_tokenizer.eos_token = '<|eos|>'
fast_tokenizer.pad_token = fast_tokenizer.eos_token

paths = ["./data/conotoxin_train.txt"]

train_dataset = load_dataset("text", data_files=paths)
train_dataset = train_dataset.shuffle()

def encode(lines):
    a = tokenizer.encode(str(lines['text']))
    return fast_tokenizer(lines['text'], padding=True, add_special_tokens=True, truncation=True, max_length=70)

train_dataset.set_transform(encode)
train_dataset = train_dataset['train']


data_collator = DataCollatorForLanguageModeling(tokenizer=fast_tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./output_ConoGen_checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=1000,
    learning_rate=0.0001,
    save_total_limit=4,
    prediction_loss_only=False,
    remove_unused_columns=False,
    logging_steps=50,
    fp16=False,
)



trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,

)



trainer.train()