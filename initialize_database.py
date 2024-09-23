import sqlite3
import requests

def initialize_database():
    # 连接到SQLite数据库（如果不存在则创建）
    conn = sqlite3.connect('chinook.db')
    cursor = conn.cursor()

    # 从GitHub获取SQL脚本
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    
    if response.status_code == 200:
        sql_script = response.text
        
        # 执行SQL脚本
        try:
            cursor.executescript(sql_script)
            conn.commit()
            print("数据库初始化成功！")
        except sqlite3.Error as e:
            print(f"数据库初始化错误: {e}")
    else:
        print(f"无法从GitHub获取SQL脚本。状态码: {response.status_code}")
    
    conn.close()

if __name__ == "__main__":
    initialize_database()