# -*- coding: utf-8 -*-
"""
网页版本地知识库智能体（Flask 后端）
运行：python 智能体网页版_后端.py   然后浏览器打开 http://localhost:5000
"""
import os, glob, subprocess, sys, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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

#  1. 扫描当前文件夹的资料文件 
FILES = []
for ext, name in [(".docx", "Word"), (".doc", "Word旧"), (".txt", "文本"), (".pdf", "PDF")]:
    for f in glob.glob(os.path.join(SCRIPT_DIR, "*" + ext)) + \
             glob.glob(os.path.join(SCRIPT_DIR, "*" + ext.upper())):
        FILES.append((f, name))
if not FILES:
    print("  没找到可读文件，支持 .docx/.doc/.txt/.pdf")
    print(f"  扫描位置：{SCRIPT_DIR}")
else:
    print(f"  发现 {len(FILES)} 个文件")

# 2. 文件读取模块（pdfplumber + docx）
import pdfplumber
from docx import Document

def clean_text(text):
    """清洗文本，去除多余空格和换行"""
    if not text:
        return ""
    return '\n'.join([line.strip() for line in text.split('\n') if line.strip()])

def read_file(file_path):
    """根据后缀名读取文件并返回清洗后的文本"""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
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

print("  开始读取并清洗本地知识库...")
FILES_CONTENT = []
for f, name in FILES:
    text = read_file(f)
    if text:
        FILES_CONTENT.append((f, text))
print(f"  成功加载 {len(FILES_CONTENT)} 个文件的内容")

#  3. 查 Ollama
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
        print("  请手动执行：pip install " + " ".join(set(MISSING)))
        sys.exit(1)

# 5. 建知识库 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS

# 用你原版的方式读取文件内容（pdfplumber + docx）
all_text = []
for fname, ftype in FILES_CONTENT:
    file_name_only=os.path.basename(fname)
    all_text.append(f"【来源：{file_name_only}】\n{ftype}")
    print(f"  已读：{file_name_only}（{len(ftype)} 字）")

full = "\n\n".join(all_text) if all_text else "（暂无本地资料，将主要依赖联网搜索）"
parts = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_text(full)
vs = FAISS.from_texts(parts, OllamaEmbeddings(model=EMB))
chat = ChatOllama(model=LLM, temperature=0.7)
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
    """清洗模型输出：洗掉 *** ** * 以及 ### 等 markdown 装饰符号"""
    if not text:
        return ""
    # 去掉成对的 1~3 个星号，保留其中内容
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    # 去掉 ### 标题符号
    text = re.sub(r'(?m)^\s*#{1,6}\s*', '', text)
    # 去掉 ``` 代码围栏和 ` 行内反引号
    text = re.sub(r'`{1,3}', '', text)
    # 去掉残留的 > 引用符号
    text = re.sub(r'(?m)^\s*>\s?', '', text)
    # 行内多余空格收拢
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def ask_llm(prompt):
    """调用 Ollama 生成回答，兼容不同版本的 streaming API"""
    ans = ""
    try:
        # 方式1：使用 stream（大多数情况）
        for c in chat.stream([HumanMessage(content=prompt)]):
            if getattr(c, "content", None):
                ans += c.content
    except AttributeError:
        # 方式2：stream 不存在，用 invoke
        try:
            resp = chat.invoke([HumanMessage(content=prompt)])
            ans = resp.content if resp else ""
        except Exception as e:
            ans = f"（模型调用失败：{e}）"
    except Exception as e:
        # 方式3：stream 报错，降级为 invoke
        try:
            resp = chat.invoke([HumanMessage(content=prompt)])
            ans = resp.content if resp else ""
        except Exception as e2:
            ans = f"（模型调用失败：{e2}）"
    return clean_answer(ans)   # ← 关键：返回前统一清洗

# 6. 对外提供 HTTP 接口 
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

    # 1. 本地检索
    local_docs = vs.similarity_search(q, k=3)
    local_ctx = "\n".join(d.page_content for d in local_docs)
    local_ok = len(local_ctx) > 50

    # 2. 本地不足则联网
    web_info = ""
    if not local_ok:
        print("  本地资料不足，正在联网搜索...")
        web_info = web_search(q)

    # 3. 提取本地资料里的文件名（用 Python 自动提取，不靠 AI）
    sources = re.findall(r'【来源：(.*?)】', local_ctx)
    source_str = ""
    if sources:
        unique_sources = list(dict.fromkeys(sources))  # 去重，防止同一个文件出现多次
        source_str = "\n\n（来源：" + "、".join(unique_sources) + "）"

    # 4. 组装极简提示词
    prompt = (
        "请根据以下资料回答用户问题。优先使用【本地资料】，不足时再参考【网络信息】。\n"
        "如果都没有相关信息，请如实说不知道。直接输出正文答案即可，不要包含额外的格式说明。\n\n"
        f"【本地资料】\n{local_ctx}\n\n"
        f"【网络信息】\n{web_info}\n\n"
        f"问题：{q}"
    )

    # 5. 生成回答
    ans = ask_llm(prompt)

    # 6. 如果用了本地资料，把刚才 Python 提取好的来源拼上去
    if local_ok and source_str:
        ans += source_str
        
    return jsonify({"reply": ans, "used_web": not local_ok})
if __name__ == "__main__":
    print("=" * 50)
    print("  服务已启动：浏览器打开 http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000)
