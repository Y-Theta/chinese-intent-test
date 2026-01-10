import os
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

class chinese_intent_train:
    def __init__(self, model_dir="./paraphrase-multilingual-MiniLM-L12-v2", classifier_path="classifier.pkl"):
        self.model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.model_dir = model_dir
        self.classifier_path = classifier_path
        self.tokenizer = None
        self.session = None

    def _mean_pooling(self, model_output, attention_mask):
        """
        手动实现 Mean Pooling，将 Token 向量转换为句子向量
        """
        token_embeddings = model_output[0] # First element contains all token embeddings
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def _get_embeddings(self, texts):
        """
        使用 ONNX Runtime 批量提取文本向量
        """
        if self.session is None or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.session = ort.InferenceSession(os.path.join(self.model_dir, "model.onnx"))

        # 对文本进行编码
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        
        # ONNX 推理输入
        onnx_inputs = {k: v for k, v in encoded_input.items()}
        model_output = self.session.run(None, onnx_inputs)
        
        # 池化处理
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def prepareModel(self, proxy=None):
        """
        下载并转换为 ONNX
        """
        if proxy:
            os.environ['http_proxy'] = proxy
            os.environ['https_proxy'] = proxy
            
        print("正在导出 ONNX 模型...")
        model = ORTModelForFeatureExtraction.from_pretrained(self.model_id, export=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        model.save_pretrained(self.model_dir)
        tokenizer.save_pretrained(self.model_dir)
        print(f"ONNX 模型保存至: {self.model_dir}")

    def train_semantic_model(self, xlsx_path):
        """
        读取 Excel，使用 ONNX 提取向量，训练分类器
        """
        print(f"读取数据: {xlsx_path}")
        df = pd.read_excel(xlsx_path)
        labels = df.iloc[:, 0].values
        texts = df.iloc[:, 1].values.tolist()

        print("使用 ONNX 提取特征向量 (这可能需要一点时间)...")
        # 分批处理防止 OOM (内存溢出)
        X_embeddings = self._get_embeddings(texts)

        print("训练 LogisticRegression 分类器...")
        clf = LogisticRegression(class_weight='balanced', max_iter=1000)
        clf.fit(X_embeddings, labels)

        joblib.dump(clf, self.classifier_path)
        print(f"分类器训练完成并导出至: {self.classifier_path}")

    def test_trainer(self, test_file_path="test.txt"):
        """
        加载本地 ONNX 和 分类器进行测试
        """
        if not os.path.exists(self.classifier_path):
            print("分类器文件不存在，请先执行训练。")
            return

        clf = joblib.load(self.classifier_path)
        
        with open(test_file_path, "r", encoding="utf-8") as f:
            test_cases = [line.strip() for line in f if line.strip()]

        if not test_cases:
            print("测试文件为空。")
            return

        print(f"\n{'测试文本':<30} | {'预测意图':<12} | {'置信度'}")
        print("-" * 65)

        embeddings = self._get_embeddings(test_cases)
        predictions = clf.predict(embeddings)
        probabilities = np.max(clf.predict_proba(embeddings), axis=1)

        for text, intent, prob in zip(test_cases, predictions, probabilities):
            print(f"{text:<30} | {intent:<12} | {prob:.4f}")

# --- 运行逻辑 ---
if __name__ == "__main__":
    trainer = chinese_intent_train()

    # 第一步：准备模型
    # trainer.prepareModel(proxy="http://192.168.5.120:20171")

    # 第二步：训练 (确保 excel 文件路径正确)
    # trainer.train_semantic_model("data.xlsx")

    # 第三步：测试
    trainer.test_trainer("test.txt")