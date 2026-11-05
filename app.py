import os
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 页面基础配置
st.set_page_config(page_title="多公司银行年报 RAG 检索系统", layout="wide")

st.title("🏦 多公司银行年报 RAG 智能检索与分析系统")
st.markdown("基于多达 10 家主流上市银行年报构建的全局知识库，支持跨公司全景对比与带出处溯源检索。")

# 数据文件夹路径
DATA_DIR = r"D:\HuaweiMoveData\Users\86187\Desktop\ai_agent\data"

@st.cache_resource
def load_all_bank_reports(data_dir):
    all_chunks = []
    if not os.path.exists(data_dir):
        return all_chunks
    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    for filename in files:
        # 兼容处理你的文件名（比如 boc.pdf 或 中国银行.pdf 都能识别成代号）
        name_lower = filename.lower()
        if "cmb" in name_lower or "招商" in name_lower: company_name = "CMB"
        elif "pab" in name_lower or "平安" in name_lower: company_name = "PAB"
        elif "citic" in name_lower or "中信" in name_lower: company_name = "CITIC"
        elif "spdb" in name_lower or "浦发" in name_lower: company_name = "SPDB"
        elif "psbc" in name_lower or "邮储" in name_lower: company_name = "PSBC"
        elif "abc" in name_lower or "农业" in name_lower: company_name = "ABC"
        elif "icbc" in name_lower or "工商" in name_lower: company_name = "ICBC"
        elif "ccb" in name_lower or "建设" in name_lower: company_name = "CCB"
        elif "boc" in name_lower or "中国银行" in name_lower: company_name = "BOC"
        elif "bcm" in name_lower or "交通" in name_lower: company_name = "BCM"
        else: company_name = filename.split(".")[0].upper()

        pdf_path = os.path.join(data_dir, filename)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        all_chunks.append({
                            "company": company_name,
                            "page": page_idx + 1,
                            "content": text
                        })
        except Exception as e:
            pass
    return all_chunks

# 加载知识库
with st.spinner("正在加载 10 家银行年报数据并构建全局知识库，请稍候..."):
    bank_chunks = load_all_bank_reports(DATA_DIR)

st.success(f"🎉 知识库加载完成！已成功接入 {len(bank_chunks)} 个全局知识块。")

# 侧边栏参数调节
st.sidebar.header("⚙️ 检索控制面板")
top_k = st.sidebar.slider("召回文本块数量 (Top-K)", min_value=1, max_value=5, value=3)

# 用户输入框
query = st.text_input("💬 请输入您的查询问题（例如：中国银行的营业收入是多少 / 哪几家银行的不良贷款率最低）：", "中国银行的营业收入是多少")

if st.button("🔍 开始全局智能检索", type="primary"):
    if not bank_chunks:
        st.error("知识库为空，请检查 data 目录下的 PDF 文件！")
    elif not query.strip():
        st.warning("请输入有效的查询问题！")
    else:
        with st.spinner("正在智能匹配与跨公司检索..."):
            # 1. 实体智能识别：看看用户问的是哪家银行
            target_company = None
            bank_keywords = {
                "招商": "CMB", "招行": "CMB",
                "平安": "PAB", 
                "中信": "CITIC",
                "浦发": "SPDB",
                "邮储": "PSBC",
                "农业": "ABC", "农行": "ABC",
                "工商": "ICBC", "工行": "ICBC",
                "建设": "CCB", "建行": "CCB",
                "中国银行": "BOC", "中行": "BOC",
                "交通": "BCM", "交行": "BCM"
            }
            
            for key, code in bank_keywords.items():
                if key in query:
                    target_company = code
                    break
            
            # 2. 如果识别到了特定银行，优先在该银行的知识块中检索
            if target_company:
                filtered_chunks = [c for c in bank_chunks if c['company'] == target_company]
                if not filtered_chunks:
                    filtered_chunks = bank_chunks
            else:
                filtered_chunks = bank_chunks
                
            contents = [chunk['content'] for chunk in filtered_chunks]
            
            # 3. 计算 TF-IDF 相似度
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(contents)
            
            query_vec = vectorizer.transform([query])
            similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            top_indices = similarity.argsort()[-top_k:][::-1]
            
            st.markdown("---")
            st.markdown(f"### 📑 检索结果与溯源出处 {f'(已智能锁定目标银行: {target_company})' if target_company else ''}")
            
            for i, idx in enumerate(top_indices, 1):
                res = filtered_chunks[idx]
                score = similarity[idx]
                
                with st.expander(f"【结果 {i}】 命中公司：{res['company']} | 对应页码：第 {res['page']} 页 | 相似度得分：{score:.4f}"):
                    st.markdown(f"**内容摘要：**")
                    st.text(res['content'][:400] + "...")