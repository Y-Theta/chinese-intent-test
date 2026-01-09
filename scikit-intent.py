from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import os

# # 1. 加载预训练的中文语义模型 (模型体积小，效果极佳)
# # 首次运行会下载模型（约 40MB）
model_name = './paraphrase-multilingual-MiniLM-L12-v2'
encoder = SentenceTransformer(model_name)

# # 2. 准备数据 (即便样本少，语义模型也能理解)
# train_data = {
#     "alarm": ["帮我定个闹钟", "设置提醒", "明早八点叫我", "开启闹铃"],
#     "weather": ["天气预报", "今天冷吗", "外面下雨吗", "查一下气温"],
#     "music": ["放首歌", "我想听周杰伦", "播一段轻音乐", "放点好听的曲子"],
#     "time": ["现在几点", "告诉我时间", "现在的北京时间"]
# }

# X_texts = []
# y_labels = []
# for label, texts in train_data.items():
#     X_texts.extend(texts)
#     y_labels.extend([label] * len(texts))

# # 3. 将文本转化为语义向量 (Embedding)
# print("正在提取语义特征...")
# X_embeddings = encoder.encode(X_texts)

# # 4. 训练分类器
# clf = LogisticRegression(class_weight='balanced')
# clf.fit(X_embeddings, y_labels)

# # 5. 导出模型 (保存编码器名称和分类器)
# model_package = {
#     'encoder_name': model_name,
#     'classifier': clf,
#     'labels': clf.classes_
# }
# joblib.dump(model_package, "semantic_intent_model.pkl")
# print("高级语义模型已导出。")

# 6. 加载并预测
def predict_semantic(text):
    data = joblib.load("semantic_intent_model.pkl")
    # 同样将输入文本转为向量
    test_embedding = encoder.encode([text])
    
    # 预测概率
    probs = data['classifier'].predict_proba(test_embedding)[0]
    best_idx = np.argmax(probs)
    
    return data['labels'][best_idx], probs[best_idx]

# 测试：即使“曲子”没在训练集出现，模型也能通过语义关联到 music
test_query = "明天西安的天气"
intent, score = predict_semantic(test_query)
print(f"输入: {test_query} \n预测意图: {intent} (置信度: {score:.4f})")