import torch
from transformers import AutoTokenizer, AutoModel
import logging

class TextToEmbedding:
    def __init__(self, model_path, device=None, max_length=128):
        """
        初始化模型和分词器
        :param model_path: 预训练模型路径
        :param device: 使用的设备 (cuda 或 cpu)，默认自动检测
        :param max_length: 文本最大长度
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self._setup_logger()

    def _setup_logger(self):
        """
        设置日志记录器
        """
        self.logger = logging.getLogger("TextToEmbeddingLogger")
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _generate_embeddings(self, texts):
        """
        内部方法: 将文本转换为嵌入向量
        :param texts: List[str] 输入文本列表
        :return: List[torch.Tensor] 嵌入向量列表
        """
        embeddings = []
        for text in texts:
            # 分词和编码
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, 
                padding="max_length", max_length=self.max_length
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # 获取嵌入向量
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                # 平均池化生成句子嵌入
                sentence_embedding = torch.mean(last_hidden_state, dim=1)
                embeddings.append(sentence_embedding.cpu().squeeze(0))
        
        return embeddings

    def get_embeddings(self, texts):
        """
        接口方法: 获取文本嵌入向量
        支持两种输入:
            1. texts=[]: 传入一个文本列表
            2. texts="": 传入一个单独的字符串
        :param texts: List[str] 或 str
        :return: List[torch.Tensor] 嵌入向量列表
        """
        if isinstance(texts, str):
            texts = [texts]  # 转换为列表
        elif not isinstance(texts, list):
            raise ValueError("Input texts must be a list of strings or a single string.")
        
        self.logger.info(f"Processing {len(texts)} text(s) for embeddings...")
        embeddings = self._generate_embeddings(texts)
        self.logger.info(f"Generated embeddings for {len(texts)} text(s).")
        return embeddings