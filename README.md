<!--
 * @Date: 2024-12-04 16:54:16
 * @LastEditors: yangyehan 1958944515@qq.com
 * @LastEditTime: 2024-12-04 17:10:31
 * @FilePath: /herberta/README.md
 * @Description: 
-->
# introduction
Herberta Pretrain model experimental research model developed by the Angelpro Team, focused on Development of a pre-training model for herbal medicine.Based on the chinese-roberta-wwm-ext-large model, we do the MLM task to complete the pre-training model on the data of 675 ancient books and 32 Chinese medicine textbooks, which we named herberta, where we take the front and back words of herb and Roberta and splice them together. We are committed to make a contribution to the TCM big modeling industry. We hope it can be used:

Encoder for Herbal Formulas, Embedding Models
Word Embedding Model for Chinese Medicine Domain Data
Support for a wide range of downstream TCM tasks, e.g., classification tasks, labeling tasks, etc.

**🌟 Key Features**

​	1.	**Encoder for Herbal Formulas**: Build embeddings for herbal formulas and related concepts.

​	2.	**Domain-Specific Word Embeddings**: Specialized for the Chinese medicine domain.

​	3.	**Support for TCM Tasks**: Enables various downstream tasks, such as classification, labeling, and more.


# jingfang-HerbalFamily:
https://huggingface.co/collections/XiaoEnn/jingfang-herbalfamily-6756a48ea4b0a4a71a74c99f

**🔥 Update: Major Release!**
“Herberta has now received a major update. We have trained new pre-trained models on a larger dataset, with three versions: herberta_seq_512_V2, herberta_seq_128_V2, and herberta_V3_Modern. Their performance on downstream tasks is as follows:”

## Downstream Task: TCM Pattern Classification

### Task Definition
Using **321 pattern descriptions** extracted from TCM internal medicine textbooks, we evaluated the classification performance on four models:

1. **Herberta_seq_512_v2**: Pretrained on 700 ancient TCM books.
2. **Herberta_seq_512_v3**: Pretrained on 48 modern TCM textbooks.
3. **Herberta_seq_128_v2**: Pretrained on 700 ancient TCM books (128-length sequences).
4. **Roberta**: Baseline model without TCM-specific pretraining.


### Results

| Model Name              | Eval Accuracy | Eval F1   | Eval Precision | Eval Recall |
|--------------------------|---------------|-----------|----------------|-------------|
| **Herberta_seq_512_v2** | **0.9454**    | **0.9293** | **0.9221**     | **0.9454**  |
| **Herberta_seq_512_v3** | 0.8989        | 0.8704    | 0.8583         | 0.8989      |
| **Herberta_seq_128_v2** | 0.8716        | 0.8443    | 0.8351         | 0.8716      |
| **Roberta**             | 0.8743        | 0.8425    | 0.8311         | 0.8743      |

![image](https://github.com/user-attachments/assets/6b6fd9e2-086d-4de7-b525-7b3199f14d2d)

**The model labeled V3 was pre-trained on 48 modern Chinese medicine textbooks, while the models labeled V2 were all pre-trained on over 670 classical Chinese medicine texts, with herberta_seq_512 performing the best among them.**


## 🚀 QuickStart

### requirements
"transformers_version": "4.45.1"
```bash
pip install herberta
```

#### Use Huggingface
```python
from transformers import AutoTokenizer, AutoModel

# Replace "XiaoEnn/herberta" with the Hugging Face model repository name
model_name = "XiaoEnn/herberta"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Input text
text = "中医理论是我国传统文化的瑰宝。"

# Tokenize and prepare input
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

# Get the model's outputs
with torch.no_grad():
    outputs = model(**inputs)

# Get the embedding (sentence-level average pooling)
sentence_embedding = outputs.last_hidden_state.mean(dim=1)

print("Embedding shape:", sentence_embedding.shape)
print("Embedding vector:", sentence_embedding)
```



# Text Embedding Package

A Python package for converting texts into embeddings using pretrained transformer models.

## Installation

```bash
pip install herberta

```python
from herberta.embedding import TextToEmbedding

# Initialize the embedding model
embedder = TextToEmbedding("path/to/your/model")

# Single text input
embedding = embedder.get_embeddings("This is a sample text.")

# Multiple text input
texts = ["This is a sample text.", "Another example."]
embeddings = embedder.get_embeddings(texts)
```

