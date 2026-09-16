# 向量嵌入与向量数据库技术详解

## 一、什么是向量嵌入（Embedding）

向量嵌入是将文本、图片等非结构化数据转换为高维向量空间中的数值向量的技术。在RAG系统中，Embedding是连接自然语言和向量检索的桥梁。

### Embedding的核心原理

文本Embedding模型（如text-embedding-v2）通过深度学习将文本映射到高维向量空间（例如1536维），使得语义相似的文本在向量空间中距离更近。例如：

- "苹果手机" 和 "iPhone" 的向量距离很近（语义相似）
- "苹果手机" 和 "苹果水果" 的向量距离较远（语义不同）

### 本系统使用的Embedding模型

本系统使用DashScope的text-embedding-v2模型：
- **维度**：1536维向量
- **接口**：OpenAI兼容的Embedding API
- **特点**：支持中英文，对中文语义理解优秀
- **调用方式**：通过APIEmbedder类封装，支持批量嵌入

### Embedding的使用场景

1. **文档索引**：将知识库文档分块后，每个chunk计算Embedding并存入向量数据库
2. **查询向量化**：用户输入的查询文本也计算Embedding，用于与文档向量计算相似度
3. **语义缓存**：通过比较查询Embedding的相似度判断是否命中缓存
4. **重排序**：使用Embedding相似度辅助重排序检索结果

---

## 二、向量数据库（Vector Database）

### 什么是向量数据库

向量数据库是专门为存储和检索高维向量数据而设计的数据库。与传统关系型数据库（如MySQL、SQLite）存储结构化数据不同，向量数据库存储的是非结构化数据的向量表示。

### ChromaDB

本系统使用ChromaDB作为向量数据库：

- **类型**：开源嵌入式向量数据库
- **存储**：本地持久化存储（persist_directory）
- **集合**：使用"rag_documents"集合存储所有文档向量
- **检索**：支持余弦相似度（cosine）、欧氏距离（L2）、内积（IP）等距离度量
- **元数据**：支持存储和过滤文档元数据（source、title、doc_type等）

### ChromaDB的核心操作

```python
# 创建/获取集合
collection = client.get_or_create_collection(
    name="rag_documents",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# 添加文档向量
collection.add(
    ids=["doc_1_chunk_0"],
    embeddings=[vector_1536_dim],
    documents=["文档文本内容"],
    metadatas=[{"source": "file.md", "title": "文档标题"}]
)

# 查询相似向量
results = collection.query(
    query_embeddings=[query_vector],
    n_results=10
)
```

---

## 三、文档分块策略（Chunking）

### 为什么需要分块

LLM的上下文窗口有限，且向量检索需要细粒度的语义匹配。将长文档分成多个小块（chunk），每个块独立计算Embedding，可以提高检索的精确度。

### 本系统的分块配置

- **分块大小（chunk_size）**：800字符
- **重叠大小（chunk_overlap）**：100字符
- **分割器**：LangChain的RecursiveCharacterTextSplitter
- **分割优先级**：优先按段落（\n\n）→ 换行（\n）→ 句子（。！？）→ 字符切分

### 分块的元数据

每个chunk存储以下元数据：
- **source**：源文件路径
- **title**：文档标题
- **doc_id**：文档唯一标识
- **doc_type**：文档类型（md、pdf、xlsx等）
- **chunk_id**：块在文档中的序号

---

## 四、向量检索的核心算法

### 余弦相似度（Cosine Similarity）

计算两个向量夹角的余弦值，范围为[-1, 1]，值越大表示越相似：

similarity = cos(θ) = (A·B) / (|A| × |B|)

余弦相似度只关注向量方向，不关注长度，适合文本语义相似度计算。

### 近似最近邻搜索（ANN）

精确的KNN搜索在高维空间中计算成本很高。向量数据库通常使用ANN算法加速检索：

- **HNSW**（Hierarchical Navigable Small World）：ChromaDB默认使用的算法，构建多层导航图，在精度和速度之间取得良好平衡
- **IVF**（Inverted File Index）：将向量空间划分为多个聚类，查询时只搜索最近的几个聚类
- **PQ**（Product Quantization）：通过量化压缩向量，减少存储和计算成本

---

## 五、RAG系统中Embedding的优化实践

### 1. 批量嵌入
将文档分块后批量计算Embedding（本系统每批10个chunk），减少API调用次数。

### 2. Embedding缓存
缓存已计算的Embedding结果，避免重复计算。本系统在检索缓存层实现了Embedding缓存。

### 3. 查询Embedding优化
使用HyDE（Hypothetical Document Embeddings）技术，先生成假设答案文档，用假设文档的Embedding进行检索，提高检索质量。

### 4. 多向量检索
对同一文档生成多个Embedding（如标题Embedding + 内容Embedding），从不同角度提高召回率。

---

## 总结

向量嵌入和向量数据库是RAG系统的基础设施。本系统使用DashScope text-embedding-v2模型将文本转换为1536维向量，存储在ChromaDB向量数据库中，通过余弦相似度和HNSW索引实现高效的语义检索。结合合理的文档分块策略和Embedding优化实践，系统实现了高质量的知识检索能力。
