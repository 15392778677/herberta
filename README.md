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

Model Address:https://huggingface.co/XiaoEnn/herberta



## QuickStart

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

