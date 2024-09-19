import streamlit as st

# 设置页面配置必须是第一个 Streamlit 命令
st.set_page_config(layout="wide", page_title="RAG 知识问答系统")

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
            {"role": "user", "content": f"上下文: {context_text}\n\n问题: {query}\n\n请提供你的回答，然后在回答后面附上相关的原文摘录，以'相关原文：'为前缀。"}
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

    st.title("知识问答系统")

    # 初始化 session state
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "file_indices" not in st.session_state:
        st.session_state.file_indices = load_all_indices()

    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["RAG 问答", "网络搜索问答", "数据库查询"])

    with tab1:
        st.header("RAG 问答")

        # 添加CSS样式
        st.markdown("""
        <style>
        .stColumn {
            padding: 10px;
        }
        .divider {
            border-left: 2px solid #bbb;
            height: 100vh;
            position: absolute;
            left: 50%;
            margin-left: -1px;
            top: 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # 创建左右两列布局
        left_column, divider, right_column = st.columns([2, 0.1, 3])

        with left_column:
            st.header("文档上传")
            
            # 设置最大token数
            max_tokens = 4096

            # 多文件上传 (添加唯一key)
            uploaded_files = st.file_uploader("上传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="rag_file_uploader_1")

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with st.spinner(f"正在处理文档: {uploaded_file.name}..."):
                        chunks, index = vectorize_document(uploaded_file, max_tokens)
                        st.session_state.file_indices[uploaded_file.name] = (chunks, index)
                        save_index(uploaded_file.name, chunks, index)
                    st.success(f"文档 {uploaded_file.name} 已向量化并添加到索引中！")

            # 显示已处理的文件
            st.subheader("已处理文档:")
            for file_name in st.session_state.file_indices.keys():
                st.write(f"• {file_name}")

            # 添加关键词搜索功能
            st.subheader("关键词搜索")
            search_keywords = st.text_input("���入关键词（用空格分隔）", key="rag_search_keywords_1")
            if search_keywords:
                keywords = search_keywords.split()
                relevant_docs = search_documents(keywords, st.session_state.file_indices)
                if relevant_docs:
                    st.write("相关文档：")
                    for doc in relevant_docs:
                        st.write(f"• {doc}")
                    # 存储相关文档到 session state
                    st.session_state.relevant_docs = relevant_docs
                else:
                    st.write("没有找到相关档。")
                    st.session_state.relevant_docs = None

        # 添加垂直分割线
        with divider:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        with right_column:
            # 创建一个容器来放置对话历史
            chat_container = st.container()

            # 显示对话历史
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

            # 创建一个列布局来放置输入框和清除对话按钮
            col1, col2 = st.columns([3, 1])

            # 用户输入
            with col1:
                prompt = st.chat_input("请基于上传的文档提出问题:", key="rag_chat_input_1")

            # 清除对话按钮
            with col2:
                if st.button("清除对话", key="clear_rag_chat_1"):
                    st.session_state.rag_messages = []
                    st.rerun()

            if prompt:
                st.session_state.rag_messages.append({"role": "user", "content": prompt})
                
                if st.session_state.file_indices:
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.chat_message("assistant"):
                            with st.spinner("正在生成回答..."):
                                try:
                                    # 使用保存的相关文档（如果有的话）
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

        # 将输入框和清除按钮放在标题下方
        col1, col2 = st.columns([3, 1])
        with col1:
            web_prompt = st.text_input("请输入您的问题（如需搜索，请以'搜索'开头）:", key="web_chat_input_2")
        with col2:
            if st.button("清除对话", key="clear_web_chat_2"):
                st.session_state.web_messages = []
                st.rerun()

        # 创建一个容器来放置对话历史
        web_chat_container = st.container()

        with web_chat_container:
            for message in st.session_state.web_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

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
        st.header("自然语言数据库查询")

        # 检查数据库文件是否存在
        if not os.path.exists('chinook.db'):
            st.warning("数据库文件不存在，正在尝试下载...")
            with st.spinner("正在下载并创建数据库..."):
                download_and_create_database()
            if os.path.exists('chinook.db'):
                st.success("数据库文件已成功下载！")
            else:
                st.error("无法创建数据库文件。请检查网络连接和文件权限。")

        # 调整列的宽度比例
        col1, col2 = st.columns([1, 1.5])

        with col1:
            nl_query = st.text_input("请输入您的自然语言查询:", key="db_nl_query")

            if nl_query:
                with st.spinner("正在生成SQL并执行查询..."):
                    try:
                        sql_query = nl_to_sql(nl_query)
                        st.code(sql_query, language="sql")
                        
                        results, column_names = execute_sql(sql_query)
                        if isinstance(results, str):
                            st.error(results)
                        else:
                            with col2:
                                st.subheader("查询结果")
                                df = pd.DataFrame(results, columns=column_names)
                                st.dataframe(df)

                                # 生成自然语言解释
                                explanation = generate_explanation(nl_query, sql_query, df)
                                st.subheader("结果解释")
                                st.markdown(explanation, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"查询执行错误: {str(e)}")

            if 'show_db_info' not in st.session_state:
                st.session_state.show_db_info = False
            if 'selected_table' not in st.session_state:
                st.session_state.selected_table = None

            if st.button("显示/隐藏数据库信息"):
                st.session_state.show_db_info = not st.session_state.show_db_info
                st.session_state.selected_table = None  # 重置选中的表

            if st.session_state.show_db_info:
                table_info = get_table_info()
                if not table_info:
                    st.error("无法获取数据库信息。请检查数据库连接。")
                else:
                    st.success(f"成功获取到 {len(table_info)} 个表的信息")
                    
                    # 创建一个动态的列布局来横向排列表名
                    cols = st.columns(4)  # 每行4个表名
                    for i, table in enumerate(table_info.keys()):
                        with cols[i % 4]:
                            if st.button(table, key=f"table_{table}"):
                                st.session_state.selected_table = table

        # 在表名下方显示查询结果
        if st.session_state.selected_table:
            st.subheader(f"表名: {st.session_state.selected_table}")
            results, column_names = execute_sql(f"SELECT * FROM {st.session_state.selected_table} LIMIT 10")
            if isinstance(results, str):
                st.error(results)
            else:
                df = pd.DataFrame(results, columns=column_names)
                st.dataframe(df)

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
    return response.choices[0].message.content.strip()

def execute_sql(sql_query):
    conn = sqlite3.connect('chinook.db')
    c = conn.cursor()
    try:
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