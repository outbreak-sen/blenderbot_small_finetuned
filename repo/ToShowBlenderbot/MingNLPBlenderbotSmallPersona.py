# !pip install mindnlp
# !pip install mindspore==2.4
# !export LD_PRELOAD=$LD_PRELOAD:/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-74ff64e9.so.1.0.0
# !yum install libsndfile
from mindnlp.transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer
from mindnlp.engine  import Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
import mindspore as ms
import os
# 设置运行模式和设备
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
# 设置 HF_ENDPOINT 环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 加载模型和分词器
print("加载模型和分词器")
model_name = "facebook/blenderbot_small-90M"
tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name)
print("模型和分词器加载完成")


from mindnlp.peft import LoraConfig, TaskType, get_peft_model

# Get the model with a PEFT adapter
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
# # 合理冻结模型中除 LoRA 参数以外的权重
# for name, param in model.named_parameters():
#     if "lora_" not in name:
#         param.requires_grad = False

# Print the trainable parameters of the model
# model.print_trainable_parameters()

# Print the trainable parameters of the model
model.print_trainable_parameters()

# 测试原始模型的输出
input = "Persona 1: Nice to meet you too. What are you interested in?"
print("input question:", input)
input_tokens = tokenizer([input], return_tensors="ms")
output_tokens = model.generate(**input_tokens)
print("output answer:", tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0])
# 设置填充标记（BlenderbotSmall默认无pad_token）
# tokenizer.pad_token = tokenizer.eos_token  # 用eos_token作为填充标记
# model.config.pad_token_id = tokenizer.eos_token_id
print("加载数据集")
# 加载 Persona-Chat 数据集
# 定义数据集保存路径
dataset_path = "./dataset_valid_preprocessed"
# 检查是否存在处理好的数据集
if os.path.exists(dataset_path):
    # 加载预处理后的数据集
    dataset_train = load_from_disk("./dataset_train_preprocessed")
    dataset_valid = load_from_disk("./dataset_valid_preprocessed")
else:
    dataset = load_dataset("google/Synthetic-Persona-Chat")
    print("dataset finished")
    print("dataset:", dataset)
    print("dataset['train'][1]:", dataset["train"][1])
    dataset_valid = dataset["validation"]
    dataset_train = dataset["train"]
    print("dataset_train.column_names:", 
          dataset_train.column_names)
    print("dataset_train:", dataset_train)
    print("dataset_train['Best Generated Conversation'][1]:\n", 
          dataset_train["Best Generated Conversation"][1])
    print("dataset_train['user 1 personas'][1]:", 
          dataset_train["user 1 personas"][1])
    print("dataset_train['user 2 personas'][1]:", 
          dataset_train["user 2 personas"][1])
    # 数据预处理：将对话格式化为上下文-回复对
    def format_dialogue(examples):
        inputs, targets = [], []
        for conversation in examples["Best Generated Conversation"]:
            lines = conversation.split("\n")
            for i in range(len(lines) - 1):
                # 我希望input是包含前文的，并且有名称的对象
                context = "\n".join(lines[:i+1])  # 上下文是当前行及之前的所有行
                reply = lines[i+1]  # 下一行是回复
                inputs.append(context.strip())
                # 我希望回答是不带User的，直接就是一句话
                reply = reply.replace("User 1: ", "")
                reply = reply.replace("User 2: ", "")
                targets.append(reply.strip())
        return {"input": inputs, "target": targets}
    # 应用预处理函数
    dataset_train = dataset_train.map(format_dialogue, batched=True
                                        , remove_columns=["user 1 personas"
                                                            , "user 2 personas"
                                                            , "Best Generated Conversation"])
    dataset_valid = dataset_valid.map(format_dialogue, batched=True
                                        , remove_columns=["user 1 personas"
                                                            , "user 2 personas"
                                                            , "Best Generated Conversation"])

    # 保存预处理后的数据集
    dataset_train.save_to_disk("./dataset_train_preprocessed")
    print("after format:")
    print("dataset_train.column_names:", 
          dataset_train.column_names)
    print("dataset_train['input'][1]:", 
          dataset_train["input"][1])
    print("dataset_train['target'][1]:", 
          dataset_train["target"][1])
    dataset_valid.save_to_disk("./dataset_valid_preprocessed")
print("数据集取五十分之一")
# 计算十分之一的数据量
train_size = len(dataset_train)
valid_size = len(dataset_valid)
train_subset_size = train_size // 50
valid_subset_size = valid_size // 50
# 使用 select 方法选择前十分之一的数据
dataset_train_tokenized = dataset_train.select(range(train_subset_size))
dataset_valid_tokenized = dataset_valid.select(range(valid_subset_size))
    
print("tokenizer数据集")
# 定义数据集保存路径
dataset_path = "./datasetTokenized_train_preprocessed"
# 检查是否存在处理好的数据集
if os.path.exists(dataset_path):
    # 加载预处理后的数据集
    dataset_train_tokenized = load_from_disk("./datasetTokenized_train_preprocessed")
    dataset_valid_tokenized= load_from_disk("./datasetTokenized_valid_preprocessed")
else:
    # 分词处理
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=128,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]#获得"labels" "input_ids" "attention_mask"
        return model_inputs

    dataset_train_tokenized = dataset_train_tokenized.map(tokenize_function, batched=True)
    dataset_valid_tokenized = dataset_train_tokenized.map(tokenize_function, batched=True)
    # 保存预处理后的数据集
    dataset_train_tokenized.save_to_disk("./datasetTokenized_train_preprocessed")
    dataset_valid_tokenized.save_to_disk("./datasetTokenized_valid_preprocessed")
print("dataset_train_tokenized",dataset_train_tokenized)
print("转化为mindspore格式数据集")
import mindspore.dataset as ds
def data_generator(dataset):
    for item in dataset:
        yield item["input_ids"], item["attention_mask"], item["labels"]

import mindspore.dataset as ds
# 将训练集和验证集转换为 MindSpore 数据集，注意forward函数中label要改成labels
def create_mindspore_dataset(dataset, shuffle=True):
    return ds.GeneratorDataset(
        source=lambda: data_generator(dataset),  # 使用 lambda 包装生成器
        column_names=["input_ids", "attention_mask", "labels"],
        shuffle=shuffle
    )
dataset_train_tokenized = create_mindspore_dataset(dataset_train_tokenized, shuffle=True)
dataset_valid_tokenized = create_mindspore_dataset(dataset_valid_tokenized, shuffle=False)
# 训练参数
TOKENS = 20
EPOCHS = 10
BATCH_SIZE = 4
# 定义训练参数
training_args = TrainingArguments(
    output_dir='./Mindsporeblenderbot_persona_finetuned',
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    
    save_steps=500,                  # Save checkpoint every 500 steps
    save_total_limit=2,              # Keep only the last 2 checkpoints
    logging_dir="./mindsporelogs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    eval_steps=500,                  # Evaluation frequency
    warmup_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,               # Weight decay
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train_tokenized,
    eval_dataset=dataset_valid_tokenized
)
# 开始训练
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
# 保存模型
model.save_pretrained("./blenderbot_dialogue_finetuned")
tokenizer.save_pretrained("./blenderbot_dialogue_finetuned")
fine_tuned_model = BlenderbotSmallForConditionalGeneration.from_pretrained("./blenderbot_dialogue_finetuned")
fine_tuned_tokenizer = BlenderbotSmallTokenizer.from_pretrained("./blenderbot_dialogue_finetuned")
# 再次测试对话
print("再次测试对话")
input = "Persona 1: Nice to meet you too. What are you interested in?"
print("input question:", input)
input_tokens = fine_tuned_tokenizer([input], return_tensors="ms")
output_tokens = fine_tuned_model.generate(**input_tokens)
print("output answer:", fine_tuned_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0])