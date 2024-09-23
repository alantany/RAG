import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect, MetaData
import sqlite3
import io
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
import os
from pyvis.network import Network

# 设置页面标题
st.set_page_config(page_title="基于AI的数据分析", layout="wide")

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
    base_url="https://api.chatanywhere.tech/v1"
)

# 主页面标题
st.title("基于AI的数据分析")

# NL2SQL函数
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

# 修改数据库连接函数
def connect_to_database():
    db_path = 'chinook.db'
    if not os.path.exists(db_path):
        st.error(f"数据库文件 {db_path} 不存在。请确保已运行 initialize_database.py 脚本。")
        return None
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        st.error(f"连接数据库时出错: {e}")
        return None

# 修改数据加载函数
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

# 添加这个函数来检查数据库状态
def check_database():
    try:
        conn = sqlite3.connect('chinook.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if tables:
            st.success(f"数据库连接成功！发现 {len(tables)} 个表。")
            for table in tables:
                st.write(f"- {table[0]}")
        else:
            st.warning("数据库连接成功，但没有找到任何表。")
        conn.close()
    except sqlite3.Error as e:
        st.error(f"数据库连接错误: {e}")

# 修改表信息生成函数
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
                        fig = px.density_heatmap(result_df, x=x_column, y=y_column, z=color_column, title="热力���")
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
    st.info("选择数据源并加载数")

# 关闭数据库连接
if conn:
    conn.close()

# 数据导出功能已被移除

# 添加新函数来获取表之间的关系
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
            relationships.append({
                'table': table_name,
                'column': fk[3],
                'references_table': fk[2],
                'references_column': fk[4]
            })
    
    return relationships

# 添加新函数来生成关系图
def generate_relationship_graph(relationships):
    net = Network(height="750px", width="100%", directed=True)
    
    for rel in relationships:
        table = rel['table']
        references_table = rel['references_table']
        label = f"{rel['column']} -> {rel['references_column']}"
        
        net.add_node(table, label=table, shape='box')
        net.add_node(references_table, label=references_table, shape='box')
        net.add_edge(table, references_table, title=label, label=label)
    
    return net.generate_html()

# 修改数据库结构可视化部分
if data_source == "RDBMS数据库":
    st.subheader("数据库表关系")

    if st.button("显示数据库表关系"):
        conn = connect_to_database()
        if conn:
            relationships = get_table_relationships(conn)
            if relationships:
                html = generate_relationship_graph(relationships)
                st.components.v1.html(html, height=750, scrolling=True)
            else:
                st.info("未找到表之间的关系。")
            conn.close()
