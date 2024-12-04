<!--
 * @Date: 2024-12-04 16:54:16
 * @LastEditors: yangyehan 1958944515@qq.com
 * @LastEditTime: 2024-12-04 17:10:31
 * @FilePath: /herberta/README.md
 * @Description: 
-->
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

