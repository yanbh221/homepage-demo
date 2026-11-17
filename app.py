import os
import re
import json
import pdfplumber
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# 页面配置
st.set_page_config(page_title="多公司银行年报 RAG 检索系统", layout="wide")
st.title("🏦 多公司银行年报 RAG 智能检索与分析系统")
st.markdown("支持 **TF-IDF 向量 + BM25 双路混合检索**、**PDF 章节层级识别** 与 **表格行列结构化还原**。")

DATA_DIR = r"D:\HuaweiMoveData\Users\86187\Desktop\ai_agent\data"
ARTIFACT_PATH = "parsed_bank_metadata.json"  # 实际结构化产物保存路径

# 10家银行官方披露元信息与全文可回查链接（2026年半年度报告）
BANK_SOURCES = {
    "BOC": {"name": "中国银行", "year": "2026年半年度报告", "url": "https://www.boc.cn/investor/ir3/202609/t20260928_25694736.html"},
    "CMB": {"name": "招商银行", "year": "2026年半年度报告", "url": "https://www.cmbchina.com/cmbir/intro?type=report"},
    "ICBC": {"name": "工商银行", "year": "2026年半年度报告", "url": "https://v.icbc.com.cn/userfiles/resources/icbcltd/download/2026/InterimReportA20260924.pdf"},
    "CCB": {"name": "建设银行", "year": "2026年半年度报告", "url": "https://wealth.ccb.com/chn/2026-08/28/article_2026082818542487156.shtml"},
    "ABC": {"name": "农业银行", "year": "2026年半年度报告", "url": "https://www.abchina.com/cn/aboutabc/investor_relations/report/mm/202608/t20260828_2658912.htm"},
    "BCM": {"name": "交通银行", "year": "2026年半年度报告", "url": "https://www.bankcomm.com/BankCommSite/shtml/jyjr/cn/7768/7800/7851/list_1.shtml?channelId=7226"},
    "PAB": {"name": "平安银行", "year": "2026年半年度报告", "url": "http://notice.10jqka.com.cn/api/pdf/4dd0e41a3a55f67a.pdf"},
    "CITIC": {"name": "中信银行", "year": "2026年半年度报告", "url": "https://www.citicbank.com/about/investor/financialaffairs/report/2026/"},
    "PSBC": {"name": "邮储银行", "year": "2026年半年度报告", "url": "https://www.psbc.com/cn/gyyc/tzzgx/cwxx/cwbg/"},
    "SPDB": {"name": "浦发银行", "year": "2026年半年度报告", "url": "https://news.spdb.com.cn/investor_relation/periodic_report/"},
    "NBCB": {"name": "宁波银行", "year": "2026年半年度报告", "url": "https://www.nbcb.com.cn/about_nbcb/information_diclosure/periodic_report/202608/t20260820_1225396.shtml"}
}

def extract_chapter_title(text):
    """正则提取当前页所属章节标题"""
    for line in text.split("\n")[:6]:
        line_clean = line.strip()
        if re.match(r"^(第[一二三四五六七八九十]+章|第[一二三四五六七八九十]+节|[一二三四五六七八九十]+、|\d+\.\d+)", line_clean):
            return line_clean[:40]
    return "正文/财务报表附注"

def format_tables_with_row_col(tables):
    """表格行列结构化还原（保留 Row/Col 坐标）"""
    structured_rows = []
    for t_idx, table in enumerate(tables):
        if not table:
            continue
        headers = [str(cell).replace("\n", " ").strip() if cell else f"Col_{c_idx}" for c_idx, cell in enumerate(table[0])]
        for r_idx, row in enumerate(table[1:], start=1):
            row_cells = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
            if any(row_cells):
                row_desc = f"[表{t_idx+1}-第{r_idx}行] " + " | ".join(
                    f"{headers[c] if c < len(headers) else f'列{c+1}'}: {val}"
                    for c, val in enumerate(row_cells) if val
                )
                structured_rows.append(row_desc)
    return "\n".join(structured_rows)

@st.cache_resource
def load_and_parse_reports(data_dir):
    all_chunks = []
    if not os.path.exists(data_dir):
        return all_chunks

    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    for filename in files:
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
        elif "nbcb" in name_lower or "宁波" in name_lower: company_name = "NBCB"
        else: company_name = filename.split(".")[0].upper()

        source_meta = BANK_SOURCES.get(company_name, {"name": company_name, "year": "2026年半年度报告", "url": "官方披露渠道"})
        pdf_path = os.path.join(data_dir, filename)
        current_chapter = "目录与重要提示"

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    chapter = extract_chapter_title(text)
                    if chapter != "正文/财务报表附注":
                        current_chapter = chapter

                    # 提取并还原表格行列
                    tables = page.extract_tables() or []
                    table_text = format_tables_with_row_col(tables)
                    combined_content = f"【章节：{current_chapter}】\n{text}\n{table_text}".strip()

                    if combined_content:
                        all_chunks.append({
                            "company": company_name,
                            "bank_cn": source_meta["name"],
                            "report_year": source_meta["year"],
                            "source_url": source_meta["url"],
                            "file_name": filename,
                            "page": page_idx + 1,
                            "chapter": current_chapter,
                            "has_table": len(tables) > 0,
                            "table_preview": table_text[:300] if table_text else "无表格",
                            "content": combined_content
                        })
        except Exception:
            pass

    # 落盘保存章节与表格行列提取的实际元数据产物（取前200条示例写入仓库供核验）
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks[:200], f, ensure_ascii=False, indent=2)

    return all_chunks

bank_chunks = load_and_parse_reports(DATA_DIR)
st.success(f"知识库构建完成：共加载 {len(bank_chunks)} 个增强知识块（含章节标签与表格行列还原，元数据产物已导出至 `{ARTIFACT_PATH}`）。")

# 检索交互与双路混合检索（TF-IDF 向量 + BM25）
query = st.text_input("请输入查询问题：", "工商银行2026年6月末不良贷款率是多少？")
top_k = st.sidebar.slider("Top-K 召回数量", 1, 5, 3)

if st.button("开始双路混合检索", type="primary") and bank_chunks:
    bank_keywords = {
        "招商": "CMB", "招行": "CMB", "平安": "PAB", "中信": "CITIC",
        "浦发": "SPDB", "邮储": "PSBC", "农业": "ABC", "农行": "ABC",
        "工商": "ICBC", "工行": "ICBC", "建设": "CCB", "建行": "CCB",
        "中国银行": "BOC", "中行": "BOC", "交通": "BCM", "交行": "BCM", "宁波": "NBCB"
    }
    target_company = next((code for kw, code in bank_keywords.items() if kw in query), None)
    filtered = [c for c in bank_chunks if c["company"] == target_company] if target_company else bank_chunks
    if not filtered:
        filtered = bank_chunks

    contents = [c["content"] for c in filtered]

    # 1. TF-IDF 向量相似度
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(contents)
    tfidf_scores = cosine_similarity(vectorizer.transform([query]), tfidf_matrix).flatten()

    # 2. BM25 词频检索（按字符/词切分适配中文财报）
    tokenized_corpus = [list(doc) for doc in contents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = np.array(bm25.get_scores(list(query)))

    if tfidf_scores.max() > 0:
        tfidf_scores /= tfidf_scores.max()
    if bm25_scores.max() > 0:
        bm25_scores /= bm25_scores.max()

    final_scores = 0.5 * tfidf_scores + 0.5 * bm25_scores
    top_indices = final_scores.argsort()[-top_k:][::-1]

    for rank, idx in enumerate(top_indices, 1):
        item = filtered[idx]
        with st.expander(f"Top-{rank} | {item['bank_cn']}({item['company']}) | 第{item['page']}页 | 章节：{item['chapter']} | 综合得分：{final_scores[idx]:.4f}"):
            st.markdown(f"- **官方出处**：{item['report_year']}（[点击回查官方原文]({item['source_url']})）")
            st.markdown(f"- **表格行列还原**：`{item['table_preview']}`")
            st.text(item["content"][:500])