import torch
import torch.nn as nn
import math
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.optim as optim

# 配置日志，每次运行时新建 output.txt 文件
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("output.txt", mode='w'),  # 以写入模式打开文件，每次新建
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()


class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于向输入嵌入中添加位置信息。
    Transformer 模型本身不具备捕捉序列顺序的能力，因此需要通过位置编码来注入位置信息。
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)  # 不作为模型参数

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]  # 广播机制添加位置编码
        return x


class TransformerEncoderLayer(nn.Module):
    """
    单层 Transformer 编码器层，包含多头自注意力机制和前馈神经网络。
    """

    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: [seq_len, batch_size, d_model]
        src_mask: [seq_len, seq_len]
        src_key_padding_mask: [batch_size, seq_len]
        """
        # 多头自注意力
        attn_output, _ = self.self_attn(src, src, src,
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)  # 移除了 return_attn_weights

        # 残差连接
        src = src + self.dropout1(attn_output)

        # 层归一化
        src = self.norm1(src)

        # 前馈网络
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(src))))

        # 残差连接
        src = src + self.dropout2(ff_output)

        # 层归一化
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """
    完整的 Transformer 编码器，由多个编码器层堆叠而成。
    """

    def __init__(self, d_model, n_heads, n_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        for i, layer in enumerate(self.layers):
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(src)


class TextClassifier(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, num_classes, dim_feedforward=2048, dropout=0.1):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(21128, d_model)  # bert-base-chinese 词表大小
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)  # 添加 Dropout 层
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(0, 1)  # [seq_len, batch_size, d_model]
        embedded = self.positional_encoding(embedded)
        attention_mask = ~attention_mask.bool()  # 转换为布尔类型并取反
        output = self.encoder(embedded, src_key_padding_mask=attention_mask)
        output = output.mean(dim=0)  # 取平均作为序列表示
        output = self.dropout(output)  # 应用 Dropout
        logits = self.fc(output)
        return logits


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


print("开始读取训练数据...")
# 读取训练数据
train_data = pd.read_excel('input.xlsx')
train_texts = train_data['text'].tolist()
train_labels = train_data['label'].tolist()

# 过滤掉小于等于 0 的标签
valid_indices = [i for i, label in enumerate(train_labels) if label > 0]
train_texts = [train_texts[i] for i in valid_indices]
train_labels = [train_labels[i] - 1 for i in valid_indices]

print("训练数据读取完成。")

print("开始读取测试数据...")
# 读取测试数据
test_data = pd.read_excel('test.xlsx')
test_texts = test_data['text'].tolist()
print("测试数据读取完成。")

print("开始加载分词器...")
# 加载分词器
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
print("分词器加载完成。")

# 数据集和数据加载器
max_length = 128
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 模型参数
d_model = 768 * 3
n_heads = 12 * 3
n_layers = 2
num_classes = 10
dim_feedforward = 2048
dropout = 0.1

print("开始创建模型...")
# 创建模型
model = TextClassifier(d_model, n_heads, n_layers, num_classes, dim_feedforward, dropout)
print("模型创建完成。")

# 计算模型参数量和字节数
total_params = sum(p.numel() for p in model.parameters())
total_bytes = total_params * 4  # 假设参数为 float32，每个参数占 4 字节
print(f"模型总参数量: {total_params}")
print(f"模型总参数量对应的字节数: {total_bytes} B")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)


# 训练模型
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
        if (batch_idx + 1) % 10 == 0:
            print(f"训练进度: {((batch_idx + 1) / num_batches) * 100:.2f}%")
    return total_loss / len(dataloader)


print("开始训练模型...")
# 训练模型
N_EPOCHS = 6
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch + 1}/{N_EPOCHS} 开始训练...")
    train_loss = train(model, train_dataloader, optimizer, criterion)
    print(f"Epoch {epoch + 1}/{N_EPOCHS} 训练完成，平均 Loss: {train_loss:.3f}")
print("模型训练完成。")

# 保存模型
model_path = "Dlg-context-grade-01B.pth"
torch.save(model.state_dict(), model_path)
print(f"模型已保存到 {model_path}")

# 加载模型用于推理
loaded_model = TextClassifier(d_model, n_heads, n_layers, num_classes, dim_feedforward, dropout)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model = loaded_model.to(device)
loaded_model.eval()

print("开始进行推理...")
# 推理
predictions = []
num_texts = len(test_texts)
with torch.no_grad():
    for idx, text in enumerate(test_texts):
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        output = loaded_model(input_ids, attention_mask)
        pred = torch.argmax(output, dim=1).item()
        pred = pred + 1  # 将预测结果转换回原始标签范围
        predictions.append(pred)
        if (idx + 1) % 10 == 0:
            print(f"推理进度: {((idx + 1) / num_texts) * 100:.2f}%")
print("推理完成。")

print("开始将推理结果写回测试数据...")
# 将推理结果写回测试数据
test_data['label'] = predictions
test_data.to_excel('test_result.xlsx', index=False)
print("推理结果写入完成。")