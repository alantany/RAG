# AI知识问答系统

这是一个基于Streamlit构建的AI知识问答系统，支持RAG问答、网络搜索问答和AI数据分析功能。

## 功能特性

- **RAG知识问答**: 基于上传文档的检索增强生成问答
- **网络搜索问答**: 基于SerpAPI的实时网络搜索问答
- **AI数据分析**: 自然语言转SQL查询和数据可视化

## 安装与配置

### 1. 环境要求

- Python 3.7+
- pip (Python包管理器)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 环境变量配置

#### 本地开发环境

创建`.env`文件（已在.gitignore中排除）：

```env
# OpenAI API配置
OPENAI_API_KEY=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
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=deepseek/deepseek-chat-v3.1:free

# SerpAPI配置（用于网络搜索）
SERPAPI_KEY=你的SerpAPI密钥
```

#### Streamlit Cloud部署

在Streamlit Cloud的应用设置中添加以下环境变量：

- `OPENAI_API_KEY`: `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`
- `OPENAI_BASE_URL`: `https://openrouter.ai/api/v1`
- `OPENAI_MODEL`: `deepseek/deepseek-chat-v3.1:free`
- `SERPAPI_KEY`: 你的SerpAPI密钥（如果需要网络搜索功能）

### 4. 数据库初始化（可选）

如果需要使用AI数据分析功能的数据库功能：

```bash
python initialize_database.py
```

## 运行应用

### 主应用

```bash
streamlit run app.py
```

### 独立数据分析模块

```bash
streamlit run ai-chart.py
```

## 项目结构

```
RAG/
├── app.py                    # 主应用文件
├── ai-chart.py              # 独立数据分析模块
├── initialize_database.py   # 数据库初始化脚本
├── requirements.txt         # Python依赖
├── .env                     # 环境变量文件（本地）
├── .gitignore              # Git忽略文件
├── indices/                # 文档索引存储目录
└── README.md               # 项目说明文档
```

## 安全说明

- API密钥已通过环境变量管理，不会被提交到代码仓库
- `.env`文件已在`.gitignore`中排除
- 在Streamlit Cloud上手动配置环境变量，确保安全性

## 技术栈

- **前端**: Streamlit
- **AI模型**: OpenRouter API (DeepSeek Chat v3.1)
- **向量搜索**: FAISS + SentenceTransformers
- **数据处理**: Pandas, SQLAlchemy
- **可视化**: Plotly, Pyvis
- **文档处理**: PyPDF2, python-docx

## 开发者

Huaiyuan Tan
