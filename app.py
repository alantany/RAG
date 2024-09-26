"""
AI知识问答系统安装文档

1. 环境要求：
   - Python 3.7+
   - pip (Python包管理器)

2. 创建虚拟环境（可选但推荐）：
   python -m venv venv
   source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate

3. 安装依赖：
   pip install -r requirements.txt

4. requirements.txt文件内容：
   streamlit
   openai
   sentence-transformers
   PyPDF2
   python-docx
   faiss-cpu
   tiktoken
   serpapi
   pandas
   sqlite3  # 通常已包含在Python标准库中

5. 其他依赖：
   - 确保你有有效的OpenAI API密钥
   - 如果使用Google搜索功能，需要有效的SerpAPI密钥

6. 运行应用：
   streamlit run app.py

注意：
- 请确保所有依赖都已正确安装
- 在代码中替换OpenAI API密钥和SerpAPI密钥为你自己的密钥
- 对于大型文件处理，可能需要增加系统内存或使用更强大的硬件
"""

import streamlit as st
import sys
import os

# 设置页面配置必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="AI知识问答系统 - by Huaiyuan Tan",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 添加开发者信息
st.markdown("<h6 style='text-align: right; color: gray;'>开发者: Huaiyuan Tan</h6>", unsafe_allow_html=True)

# 隐藏 Streamlit 默认的菜单、页脚和 Deploy 按钮
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import multiprocessing
import PyPDF2
import docx
import faiss
import tiktoken
import os
import pickle
import numpy as np
import jieba
from collections import Counter
import sqlite3
import pandas as pd
from serpapi import GoogleSearch
import requests
import io
from sqlalchemy import create_engine, inspect
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components

# 初始化
client = OpenAI(
    api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
    base_url="https://api.chatanywhere.tech/v1"
)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 计算token数量
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(string))

# 文档向量化模块
def vectorize_document(file, max_tokens):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = file.getvalue().decode("utf-8")
    
    chunks = []
    current_chunk = ""
    for sentence in text.split('.'):
        if num_tokens_from_string(current_chunk + sentence) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence + '.'
    if current_chunk:
        chunks.append(current_chunk)
    
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(384)  # 384是向量维度,根据实际模型调整
    index.add(vectors)
    return chunks, index

# 新增函数：提取关键词
def extract_keywords(text, top_k=5):
    words = jieba.cut(text)
    word_count = Counter(words)
    # 过滤掉停用词和单个字符
    keywords = [word for word, count in word_count.most_common(top_k*2) if len(word) > 1]
    return keywords[:top_k]

# 新增函数：基于关键词搜索文档
def search_documents(keywords, file_indices):
    relevant_docs = []
    for file_name, (chunks, _) in file_indices.items():
        doc_content = ' '.join(chunks)
        if any(keyword in doc_content for keyword in keywords):
            relevant_docs.append(file_name)
    return relevant_docs

# 修改知识问答模块
def rag_qa(query, file_indices, relevant_docs=None):
    keywords = extract_keywords(query)
    if relevant_docs is None:
        relevant_docs = search_documents(keywords, file_indices)
    
    if not relevant_docs:
        return "没有找到相关文档。请尝试使用不同的关键词。", [], ""

    all_chunks = []
    chunk_to_file = {}
    combined_index = faiss.IndexFlatL2(384)
    
    offset = 0
    for file_name in relevant_docs:
        if file_name in file_indices:
            chunks, index = file_indices[file_name]
            all_chunks.extend(chunks)
            for i in range(len(chunks)):
                chunk_to_file[offset + i] = file_name
            combined_index.add(index.reconstruct_n(0, index.ntotal))
            offset += len(chunks)

    if not all_chunks:
        return "没有找到相关信息。请确保已上传文档。", [], ""

    query_vector = model.encode([query])
    D, I = combined_index.search(query_vector, k=3)
    context = []
    context_with_sources = []
    for i in I[0]:
        if 0 <= i < len(all_chunks):  # 确索引在有效范围内
            chunk = all_chunks[i]
            context.append(chunk)
            file_name = chunk_to_file.get(i, "未知文件")
            context_with_sources.append((file_name, chunk))

    context_text = "\n".join(context)
    
    # 确保总token数不超过4096
    max_context_tokens = 3000  # 为系统消息、查询和其他内容预留更多空间
    while num_tokens_from_string(context_text) > max_context_tokens:
        context_text = context_text[:int(len(context_text)*0.9)]  # 每次减少10%的内容
    
    if not context_text:
        return "没有找到相关信息。", [], ""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一段最相关的原文摘录，以'相关原文：'为前缀。"},
            {"role": "user", "content": f"上下文: {context_text}\n\n问题: {query}\n\n请提供你的回答然后在回答后面附上相关的原文摘录，以'相关原文：'为前缀。"}
        ]
    )
    answer = response.choices[0].message.content
    
    # 更灵活地处理回答格式
    if "相关原文：" in answer:
        answer_parts = answer.split("相关原文：", 1)
        main_answer = answer_parts[0].strip()
        relevant_excerpt = answer_parts[1].strip()
    else:
        main_answer = answer.strip()
        relevant_excerpt = ""
    
    # 如果AI没有提供相关原文，我们从上下文中选择一个
    if not relevant_excerpt and context:
        relevant_excerpt = context[0][:200] + "..."  # 使用第一个上下文的前200个字符
    
    # 找出包含相关原文的文件
    relevant_sources = []
    if relevant_excerpt:
        for file_name, chunk in context_with_sources:
            if relevant_excerpt in chunk:
                relevant_sources.append((file_name, chunk))
                break  # 只添加第一个匹配的文件
    if not relevant_sources and context_with_sources:  # 如果没有找到精确匹配，使用第一个上下文源
        relevant_sources.append(context_with_sources[0])

    return main_answer, relevant_sources, relevant_excerpt

# 保存索引和chunks
def save_index(file_name, chunks, index):
    if not os.path.exists('indices'):
        os.makedirs('indices')
    with open(f'indices/{file_name}.pkl', 'wb') as f:
        pickle.dump((chunks, index), f)
    # 保存文件名到一个列表中
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
    else:
        file_list = []
    if file_name not in file_list:
        file_list.append(file_name)
        with open(file_list_path, 'w') as f:
            f.write('\n'.join(file_list))

# 加载所有保存的索引
def load_all_indices():
    file_indices = {}
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        for file_name in file_list:
            file_path = f'indices/{file_name}.pkl'
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    chunks, index = pickle.load(f)
                file_indices[file_name] = (chunks, index)
    return file_indices

def delete_index(file_name):
    if os.path.exists(f'indices/{file_name}.pkl'):
        os.remove(f'indices/{file_name}.pkl')
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        if file_name in file_list:
            file_list.remove(file_name)
            with open(file_list_path, 'w') as f:
                f.write('\n'.join(file_list))

def main():
    st.markdown("""
    <style>
    .reportview-container .main .block-container{
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    .stColumn {
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AI知识问答系统")

    # 初始化 session state
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "file_indices" not in st.session_state:
        st.session_state.file_indices = load_all_indices()

    # 创建签页
    tab1, tab2, tab3 = st.tabs(["RAG知识问答", "网络搜索问答", "AI数据分析"])

    with tab1:
        st.header("RAG 问答")

        # 文档上传部分
        st.subheader("文档上传")
        
        max_tokens = 4096

        uploaded_files = st.file_uploader("上传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="rag_file_uploader_1")

        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"正在处理文档: {uploaded_file.name}..."):
                    chunks, index = vectorize_document(uploaded_file, max_tokens)
                    st.session_state.file_indices[uploaded_file.name] = (chunks, index)
                    save_index(uploaded_file.name, chunks, index)
                st.success(f"文档 {uploaded_file.name} 已向量化并添加到索引中！")

        # 显示已处理的文件并添加删除按钮
        st.subheader("已处理文档:")
        for file_name in list(st.session_state.file_indices.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {file_name}")
            with col2:
                if st.button("删除", key=f"delete_{file_name}"):
                    del st.session_state.file_indices[file_name]
                    delete_index(file_name)
                    st.success(f"文档 {file_name} 已删除！")
                    st.rerun()

        # 添加关键词搜索功能
        st.subheader("关键词搜索")
        search_keywords = st.text_input("输入关键词（用空格分隔）", key="rag_search_keywords_1")
        if search_keywords:
            keywords = search_keywords.split()
            relevant_docs = search_documents(keywords, st.session_state.file_indices)
            if relevant_docs:
                st.write("相关文档：")
                for doc in relevant_docs:
                    st.write(f"• {doc}")
                st.session_state.relevant_docs = relevant_docs
            else:
                st.write("没有找到相关文档。")
                st.session_state.relevant_docs = None

        # 对话部分
        st.subheader("对话")
        chat_container = st.container()

        with chat_container:
            for i, message in enumerate(st.session_state.rag_messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        st.markdown("**参考来源：**")
                        file_name, _ = message["sources"][0]
                        st.markdown(f"**文件：** {file_name}")
                        if os.path.exists(f'indices/{file_name}.pkl'):
                            with open(f'indices/{file_name}.pkl', 'rb') as f:
                                file_content = pickle.load(f)[0]  # 获取文件内容
                            st.download_button(
                                label="下载源文件",
                                data='\n'.join(file_content),
                                file_name=file_name,
                                mime='text/plain',
                                key=f"download_{i}"
                            )
                    if "relevant_excerpt" in message:
                        st.markdown(f"**相关原文：** <mark>{message['relevant_excerpt']}</mark>", unsafe_allow_html=True)

        # 用户输入
        prompt = st.chat_input("请基于上传的文档提出问题:", key="rag_chat_input_1")

        if prompt:
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            
            if st.session_state.file_indices:
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("正在生成回答..."):
                            try:
                                relevant_docs = st.session_state.get('relevant_docs')
                                response, sources, relevant_excerpt = rag_qa(prompt, st.session_state.file_indices, relevant_docs)
                                st.markdown(response)
                                if sources:
                                    st.markdown("**参考来源：**")
                                    file_name, _ = sources[0]
                                    st.markdown(f"**文件：** {file_name}")
                                    if os.path.exists(f'indices/{file_name}.pkl'):
                                        with open(f'indices/{file_name}.pkl', 'rb') as f:
                                            file_content = pickle.load(f)[0]  # 获取文件内容
                                        st.download_button(
                                            label="下载源文件",
                                            data='\n'.join(file_content),
                                            file_name=file_name,
                                            mime='text/plain',
                                            key=f"download_new_{len(st.session_state.rag_messages)}"
                                        )
                                if relevant_excerpt:
                                    st.markdown(f"**相关原文：** <mark>{relevant_excerpt}</mark>", unsafe_allow_html=True)
                                else:
                                    st.warning("未能提取到精确的相关原文，但找到相关信息。")
                            except Exception as e:
                                st.error(f"生成回答时发生错误: {str(e)}")
                st.session_state.rag_messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "sources": sources,
                    "relevant_excerpt": relevant_excerpt
                })
            else:
                with chat_container:
                    with st.chat_message("assistant"):
                        st.warning("请先上传文档。")

    with tab2:
        st.header("网络搜索问答")

        # 初始化 session state
        if "web_messages" not in st.session_state:
            st.session_state.web_messages = []
        if "web_prompt" not in st.session_state:
            st.session_state.web_prompt = ""
        if "execute_web_query" not in st.session_state:
            st.session_state.execute_web_query = False

        # 创建一个容器来放置对话历史
        web_chat_container = st.container()

        with web_chat_container:
            for message in st.session_state.web_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 用户输入
        web_prompt = st.chat_input("请输入您的问题（如需搜索，请以'搜索'开头）:", key="web_chat_input")

        if web_prompt:
            st.session_state.web_messages.append({"role": "user", "content": web_prompt})
            
            with web_chat_container:
                with st.chat_message("user"):
                    st.markdown(web_prompt)
                with st.chat_message("assistant"):
                    with st.spinner("正在搜索并生成回答..."):
                        try:
                            if web_prompt.lower().startswith("搜索"):
                                response = serpapi_search_qa(web_prompt[2:].strip())  # 去掉"搜索"前缀
                            else:
                                response = direct_qa(web_prompt)
                            st.markdown(response)
                            st.session_state.web_messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"生成回答时发生错误: {str(e)}")


    with tab3:
        st.header("AI数据分析")
        
        def load_data():
            data_source = st.radio("选择数据源", ["Excel文件", "RDBMS数据库"])
            
            if data_source == "Excel文件":
                uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx", "xls"])
                if uploaded_file is not None:
                    df = pd.read_excel(uploaded_file)
                    return df, None, data_source
            else:
                conn = connect_to_database()
                if conn is not None:
                    try:
                        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                        if not tables.empty:
                            selected_table = st.selectbox("选择数据表", tables['name'])
                            if selected_table:
                                df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                                return df, conn, data_source
                        else:
                            st.warning("数据库中没有找到任何表。请确保已经初始化数据库。")
                    except sqlite3.Error as e:
                        st.error(f"查询数据库时出错: {e}")
                else:
                    st.error("无法连接到数据库。请检查数据库文件是否存在。")
            
            return None, None, data_source

        def generate_table_info(df, conn):
            if conn:
                inspector = inspect(create_engine('sqlite:///chinook.db'))
                table_info = "Available tables:\n"
                for table_name in inspector.get_table_names():
                    table_info += f"- {table_name}\n"
                    columns = inspector.get_columns(table_name)
                    table_info += f"  Columns: {', '.join([col['name'] for col in columns])}\n"
            else:
                table_info = f"Table name: data\nColumns: {', '.join(df.columns)}\n"
                table_info += "\n".join([f"{col}: {df[col].dtype}" for col in df.columns])
            return table_info

        def nl_to_sql(nl_query, table_info):
            prompt = f"""
            给定以下表格信息：
            {table_info}
            
            将以下自然语言查询转换为SQL：
            {nl_query}
            
            只返回SQL查询，不要包含任何解释。
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个SQL专家，能够将自然语言查询转换为SQL语句。请用中文回答。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.5,
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # 移除可能的前缀和后缀
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            
            return sql_query

        def execute_sql_query(df, sql_query):
            # 创建一个临时的 SQLite 数据库在内存中
            engine = create_engine('sqlite:///:memory:')
            
            # 将 DataFrame 写入 SQLite 数据库
            df.to_sql('data', engine, index=False)
            
            # 执行 SQL 查询
            result = pd.read_sql_query(sql_query, engine)
            
            return result

        def connect_to_database():
            try:
                conn = sqlite3.connect('chinook.db')
                return conn
            except sqlite3.Error as e:
                st.error(f"连接数据库时出错: {e}")
                return None

        def get_table_relationships(conn):
            cursor = conn.cursor()
            
            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            relationships = []
            
            for table in tables:
                table_name = table[0]
                # 获取表的外键信息
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = cursor.fetchall()
                
                for fk in foreign_keys:
                    from_table = table_name
                    to_table = fk[2]  # 引用的表名
                    from_column = fk[3]  # 外键列名
                    to_column = fk[4]  # 引用的列名
                    
                    relationships.append({
                        'from_table': from_table,
                        'to_table': to_table,
                        'from_column': from_column,
                        'to_column': to_column
                    })
            
            return relationships

        def generate_relationship_graph(relationships):
            net = Network(notebook=True, height="500px", width="100%", bgcolor="#ffffff", font_color="black")
            
            # 添加节点（表）
            tables = set()
            for rel in relationships:
                tables.add(rel['from_table'])
                tables.add(rel['to_table'])
            
            for table in tables:
                net.add_node(table, label=table, title=table, shape="box")
            
            # 添加边（关系）
            for rel in relationships:
                edge_label = f"{rel['from_column']} -> {rel['to_column']}"
                net.add_edge(rel['from_table'], rel['to_table'], 
                             title=edge_label, label=edge_label, arrows='to')
            
            # 配置网络图的物理布局
            net.set_options("""
            var options = {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 200,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {"iterations": 150}
                },
                "edges": {
                    "font": {
                        "size": 12,
                        "align": "middle"
                    },
                    "smooth": {
                        "type": "continuous",
                        "forceDirection": "none"
                    }
                },
                "nodes": {
                    "font": {
                        "size": 16,
                        "face": "Tahoma"
                    },
                    "shape": "box"
                }
            }
            """)
            
            # 生成HTML文件
            net.save_graph("temp_graph.html")
            
            # 读取生成的HTML文件内容
            with open("temp_graph.html", "r", encoding="utf-8") as f:
                html_string = f.read()
            
            return html_string

        # 加载数据
        df, conn, data_source = load_data()

        if df is not None:
            st.success("数据已成功加载！")
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(df.head())
            
            table_info = generate_table_info(df, conn)
            
            # 自然语言查询
            nl_query = st.text_input("请输入您的自然语言查询：")
            
            if nl_query:
                # 使用NL2SQL转换查询
                sql_query = nl_to_sql(nl_query, table_info)
                
                st.write(f"生成的SQL查询：{sql_query}")
                
                try:
                    if conn:
                        result_df = pd.read_sql_query(sql_query, conn)
                    else:
                        result_df = execute_sql_query(df, sql_query)
                    
                    st.write("查询结果：")
                    st.dataframe(result_df)
                    
                    # 分析选项
                    if not result_df.empty:
                        st.subheader("数据可视化")
                        
                        # 创建两列布局
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            analysis_type = st.selectbox("选择分析图表类型", [
                                "柱状图", "折线图", "散点图", "饼图", "箱线图", "热力图", "面积图", "直方图"
                            ])
                            
                            x_column = st.selectbox("选择X轴", result_df.columns)
                            y_column = st.selectbox("选择Y轴", result_df.columns)
                            
                            if analysis_type in ["散点图", "热力图"]:
                                color_column = st.selectbox("选择颜色映射列", result_df.columns)
                            
                            # 图表大小调整
                            chart_width = st.slider("图表宽度", 400, 1200, 800)
                            chart_height = st.slider("图表高度", 300, 900, 500)
                        
                        with col2:
                            if analysis_type == "柱状图":
                                fig = px.bar(result_df, x=x_column, y=y_column, title="柱状图")
                            elif analysis_type == "折线图":
                                fig = px.line(result_df, x=x_column, y=y_column, title="折线图")
                            elif analysis_type == "散点图":
                                fig = px.scatter(result_df, x=x_column, y=y_column, color=color_column, title="散点图")
                            elif analysis_type == "饼图":
                                fig = px.pie(result_df, values=y_column, names=x_column, title="饼图")
                            elif analysis_type == "箱线图":
                                fig = px.box(result_df, x=x_column, y=y_column, title="箱线图")
                            elif analysis_type == "热力图":
                                fig = px.density_heatmap(result_df, x=x_column, y=y_column, z=color_column, title="热力图")
                            elif analysis_type == "面积图":
                                fig = px.area(result_df, x=x_column, y=y_column, title="面积图")
                            elif analysis_type == "直方图":
                                fig = px.histogram(result_df, x=x_column, title="直方图")
                            
                            # 调整图表大小
                            fig.update_layout(width=chart_width, height=chart_height)
                            
                            # 显示图表
                            st.plotly_chart(fig)
                    else:
                        st.write("查询结果为空")
                
                except Exception as e:
                    st.error(f"查询执行错误: {e}")

        else:
            st.info("选择数据源并加载数据")

        # 关闭数据库连接
        if conn:
            conn.close()

        # 数据库结构可视化部分
        if data_source == "RDBMS数据库":
            st.subheader("数据库表关系")

            if st.button("显示数据库表关系"):
                conn = connect_to_database()
                if conn:
                    relationships = get_table_relationships(conn)
                    if relationships:
                        html_string = generate_relationship_graph(relationships)
                        components.html(html_string, height=600, scrolling=True)
                    else:
                        st.info("未找到表之间的关系。")
                    conn.close()

def direct_qa(query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手，能够回答各种问题。请用中文回答。"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content.strip()

def serpapi_search_qa(query, num_results=3):
    params = {
        "engine": "google",
        "q": query,
        "api_key": "04fec5e75c6f477225ce29bc358f4cc7088945d0775e7f75721cd85b36387125",
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    
    if not organic_results:
        return "没有找到相关结果。"
    
    snippets = [result.get("snippet", "") for result in organic_results]
    links = [result.get("link", "") for result in organic_results]
    
    search_results = "\n".join([f"{i+1}. {snippet} ({link})" for i, (snippet, link) in enumerate(zip(snippets, links))])
    prompt = f"""问题: {query}
搜索结果:
{search_results}

请根据上述搜索结果回答问题。如果搜索结果不足以回答问题，请说"根据搜索结果无法回答问题"。"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手，能够根据搜索结果回答问题。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def download_and_create_database():
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open('chinook.db', 'wb') as f:
            f.write(response.content)
        
        print("数据库文件已下载并保存")
        
        conn = sqlite3.connect('chinook.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        if tables:
            print(f"成功创建数据库，包含以下表：{[table[0] for table in tables]}")
        else:
            print("数据库文件已创建，但没有找到任何表")
        conn.close()
    except Exception as e:
        print(f"下载或创建数据库时出错：{e}")

def get_table_info():
    try:
        conn = sqlite3.connect('chinook.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        
        table_info = {}
        for table in tables:
            table_name = table[0]
            c.execute(f"PRAGMA table_info({table_name})")
            columns = c.fetchall()
            table_info[table_name] = [column[1] for column in columns]
        
        conn.close()
        return table_info
    except Exception as e:
        print(f"获取表信息时出错：{e}")
        return {}

def clean_sql_query(sql_query):
    # 移除可能的 Markdown 代码块标记和多余的空白字符
    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
    return ' '.join(sql_query.split())  # 移除多余的空白字符

def nl_to_sql(nl_query):
    table_info = get_table_info()
    table_descriptions = "\n".join([f"表名: {table}\n字段: {', '.join(columns)}" for table, columns in table_info.items()])
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"你是一个SQL专家，能够将自然语言查询转换为SQL语句。数据库包含以下表和字段：\n\n{table_descriptions}"},
            {"role": "user", "content": f"将以下自然语言查询转换为SQL语句：\n{nl_query}\n只返回SQL语句，不要有其他解释。"}
        ]
    )
    return clean_sql_query(response.choices[0].message.content.strip())

def execute_sql(sql_query):
    conn = sqlite3.connect('chinook.db')
    c = conn.cursor()
    try:
        sql_query = clean_sql_query(sql_query)  # 清理 SQL 查询
        c.execute(sql_query)
        results = c.fetchall()
        column_names = [description[0] for description in c.description]
        conn.close()
        return results, column_names
    except sqlite3.Error as e:
        conn.close()
        return f"SQL执行错误: {str(e)}", None

def generate_explanation(nl_query, sql_query, df):
    df_str = df.to_string(index=False, max_rows=5)
    
    prompt = (
        f"自然语言查询: {nl_query}\n"
        f"SQL查询: {sql_query}\n"
        f"查询结果 (前5行):\n"
        f"{df_str}\n\n"
        "请用通俗易懂的语言解释这个查询的结果。解释应该包括：\n"
        "1. 查询的主要目的\n"
        "2. 结果的概述\n"
        "3. 任何有趣或重要的发现\n\n"
        "请确保解释简洁明了，适合非技术人员理解。"
        "在解释中，请用**双星号**将与结果直接相关的重要数字或关键词括起来，以便后续高亮显示。"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个数据分析专家，擅长解释SQL查询结果。"},
            {"role": "user", "content": prompt}
        ]
    )
    explanation = response.choices[0].message.content.strip()
    
    # 将双星号包围的文本转换为HTML的高亮标记
    highlighted_explanation = explanation.replace("**", "<mark>", 1)
    while "**" in highlighted_explanation:
        highlighted_explanation = highlighted_explanation.replace("**", "</mark>", 1)
        highlighted_explanation = highlighted_explanation.replace("**", "<mark>", 1)
    
    return highlighted_explanation

# 运行主应用
if __name__ == "__main__":
    main()
