# Streamlit Cloud 环境变量配置指南

## 在Streamlit Cloud中配置环境变量

当您在Streamlit Cloud上部署应用时，需要在应用设置中添加以下环境变量：

### 必需的环境变量

1. **OPENAI_API_KEY**
   - 值：`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`
   - 描述：OpenRouter API密钥，用于调用DeepSeek模型

2. **OPENAI_BASE_URL**
   - 值：`https://openrouter.ai/api/v1`
   - 描述：OpenRouter API的基础URL

3. **OPENAI_MODEL**
   - 值：`deepseek/deepseek-chat-v3.1:free`
   - 描述：使用的AI模型名称

### 可选的环境变量

4. **SERPAPI_KEY**
   - 值：`04fec5e75c6f477225ce29bc358f4cc7088945d0775e7f75721cd85b36387125`
   - 描述：SerpAPI密钥，用于网络搜索功能（如果不需要网络搜索可以不配置）

## 配置步骤

1. 登录到 [Streamlit Cloud](https://share.streamlit.io/)
2. 找到您的应用
3. 点击应用右侧的设置图标（⚙️）
4. 选择 "Settings"
5. 在 "Secrets" 部分添加上述环境变量
6. 点击 "Save" 保存配置
7. 重新部署应用

## 安全说明

- 这些环境变量不会出现在您的代码仓库中
- Streamlit Cloud会安全地管理这些敏感信息
- 请确保不要在代码中硬编码任何API密钥

## 验证配置

应用启动后，如果配置正确，您应该能看到：
- 模型测试成功的消息
- 所有AI功能正常工作
- 没有关于API密钥的错误信息

如果遇到问题，请检查：
1. 环境变量名称是否正确（区分大小写）
2. 环境变量值是否完整且没有多余的空格
3. API密钥是否有效且未过期
