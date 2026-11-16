# -*- coding: utf-8 -*-
"""
网页版本地知识库智能体（Flask 后端）
运行：python 智能体网页版_后端.py   然后浏览器打开 http://localhost:5000
"""
import os, glob, subprocess, sys, re, json, shutil, datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(SCRIPT_DIR, "history_backup")
FAISS_INDEX_DIR = os.path.join(SCRIPT_DIR, "faiss_index")
CACHE_FILE = os.path.join(SCRIPT_DIR, "file_cache.json")

print("=" * 50)
print("  启动网页版智能体（本地知识库 + 联网兜底）")
print("=" * 50)

# 0. 预检查：pip 是否可用
try:
    subprocess.check_call([sys.executable, "-m", "pip", "--version"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    print("  错误：pip 不可用，请先安装 pip 或修复 Python 环境")
    sys.exit(1)

# 1. 扫描当前文件夹的资料文件 
os.makedirs(BACKUP_DIR, exist_ok=True)
raw_files = []
for ext in [".docx", ".doc", ".txt", ".pdf"]:
    for f in glob.glob(os.path.join(SCRIPT_DIR, "*" + ext)) + \
             glob.glob(os.path.join(SCRIPT_DIR, "*" + ext.upper())):
        raw_files.append(f)

FILES =raw_files

if not FILES:
    print("  没找到可读文件，支持 .docx/.doc/.txt/.pdf")
    print(f"  扫描位置：{SCRIPT_DIR}")
else:
    print(f"  去重后，参与建库的文件有 {len(FILES)} 个")

# 2. 文件读取模块
import pdfplumber
from docx import Document

def clean_text(text):
    if not text: return ""
    return '\n'.join([line.strip() for line in text.split('\n') if line.strip()])

def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t: text += t + "\n"
        elif ext == '.docx':
            doc = Document(file_path)
            text = '\n'.join([p.text for p in doc.paragraphs])
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        print(f"  读取文件 {file_path} 失败: {e}")
        return None
    return clean_text(text)

# 3. 查 Ollama 
try:
    r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
except FileNotFoundError:
    print("  没装 Ollama，请先安装并执行 ollama serve")
    sys.exit(1)
if r.returncode != 0:
    print("  Ollama 没运行，先执行 ollama serve")
    sys.exit(1)
print("  Ollama 正常")
lines = [l for l in r.stdout.strip().splitlines() if l.strip() and not l.startswith("NAME")]
MODELS = [l.split()[0] for l in lines]
print(f"  本地模型：{', '.join(MODELS) if MODELS else '（空）'}")

def pick(cands):
    for c in cands:
        if c in MODELS:
            return c
    return None

LLM = pick(["qwen2.5:3b", "qwen2.5"]) or (MODELS[0] if MODELS else None)
EMB="bge-m3"

if not LLM:
    print("  没找到对话模型，先执行 ollama pull qwen2.5:3b")
    sys.exit(1)
print(f"  对话：{LLM}  向量：{EMB}")

# 4. 自动装依赖
MISSING = []
for mod, pkg in [
    ("langchain_community", "langchain-community"),
    ("langchain_ollama", "langchain-ollama"),
    ("langchain_text_splitters", "langchain"),
    ("faiss", "faiss-cpu"),
    ("pypdf", "pypdf"),
    ("duckduckgo_search", "duckduckgo-search"),
    ("flask", "flask"),
    ("flask_cors", "flask-cors"),
]:
    try:
        __import__(mod.replace("-", "_"))
    except ImportError:
        MISSING.append(pkg)
if MISSING:
    print(f"  正在安装依赖：{' '.join(set(MISSING))}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *set(MISSING)])
    except Exception as e:
        print(f"  依赖安装失败：{e}")
        sys.exit(1)

# 5. 建知识库与增量更新
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS

print("  正在处理知识库...")
embeddings = OllamaEmbeddings(model=EMB)
chat = ChatOllama(model=LLM, temperature=0.4, num_ctx=2048, num_predict=1000)

cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)

new_texts = []
new_cache = {}
for f_path in FILES:
    file_name = os.path.basename(f_path)
    mtime = os.path.getmtime(f_path)
    time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    
    if file_name not in cache or cache[file_name] != mtime:
        print(f"  读取新资料/更新：{file_name}")
        text = read_file(f_path)
        if text:
            new_texts.append(f"【来源：{file_name} | 更新时间：{time_str}】\n{text}")
    new_cache[file_name] = mtime

if os.path.exists(FAISS_INDEX_DIR):
    print("  加载本地历史向量库...")
    vs = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    if new_texts:
        print(f"  新增 {len(new_texts)} 份资料，正在入库...")
        parts = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_text("\n\n".join(new_texts))
        vs.add_texts(parts)
        vs.save_local(FAISS_INDEX_DIR)
else:
    print("  首次启动，正在全量构建向量库...")
    if not new_texts:
        new_texts = ["（暂无本地资料，请放入文件）"]
    parts = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_text("\n\n".join(new_texts))
    vs = FAISS.from_texts(parts, embeddings)
    vs.save_local(FAISS_INDEX_DIR)

with open(CACHE_FILE, 'w', encoding='utf-8') as f:
    json.dump(new_cache, f, ensure_ascii=False, indent=2)
print("  知识库就绪")

def web_search(q, n=3):
    try:
        with DDGS() as ddgs:
            res = list(ddgs.text(q, max_results=n))
        if not res:
            return "（网络未找到相关信息）"
        return "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}" for r in res)
    except Exception as e:
        return f"（联网搜索失败：{e}）"

def clean_answer(text):
    if not text: return ""
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'(?m)^\s*#{1,6}\s*', '', text)
    text = re.sub(r'`{1,3}', '', text)
    text = re.sub(r'(?m)^\s*>\s?', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def ask_llm(prompt):
    ans = ""
    try:
        for c in chat.stream([HumanMessage(content=prompt)]):
            if getattr(c, "content", None):
                ans += c.content
    except Exception:
        try:
            resp = chat.invoke([HumanMessage(content=prompt)])
            ans = resp.content if resp else ""
        except Exception as e2:
            ans = f"（模型调用失败：{e2}）"
    return clean_answer(ans)

# 6. HTTP 接口
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=SCRIPT_DIR)
CORS(app)

@app.route("/")
def index():
    return send_from_directory(SCRIPT_DIR, "index.html")

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True) or {}
    q = (data.get("message") or "").strip()
    if not q:
        return jsonify({"reply": "请输入问题。"})

    local_docs = vs.similarity_search(q, k=3)
    local_ctx = "\n".join(d.page_content for d in local_docs)
    local_ok = len(local_ctx) > 50

    web_info = ""
    if not local_ok:
        print("  本地资料不足，正在联网搜索...")
        web_info = web_search(q)

    # 提取来源与时间
    sources = re.findall(r'【来源：(.*?) \| 更新时间：(.*?)】', local_ctx)
    source_str = ""
    if sources:
        unique_sources = list(set([f"{name.strip()}({time.strip()})" for name, time in sources]))
        source_str = "\n\n（来源：" + "、".join(unique_sources) + "）"

   # 4. 组装极简提示词
    prompt = (
        "你是一个严谨的的校园助手。请根据以下资料回答用户问题。\n"
        "【重要要求】：\n"
        "1. 必须用简明扼要的语言进行分点总结，严禁大段复制原文！\n"
        "2. 严禁输出'本地资料总结'、'根据资料显示'、'以下是回答'等任何废话前缀，直接输出干货正文！\n"
        "3. 优先使用资料信息，不足时参考网络。如果没有相关信息，如实说不知道。\n\n"
        f"【本地资料】\n{local_ctx}\n\n"
        f"【网络信息】\n{web_info}\n\n"
        f"问题：{q}\n"
        "请直接分点给出回答（不要任何前缀）："
)

    ans = ask_llm(prompt)

    if local_ok and source_str:
        ans += source_str

    return jsonify({"reply": ans, "used_web": not local_ok})

if __name__ == "__main__":
    print("=" * 50)
    print("  服务已启动：浏览器打开 http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000)