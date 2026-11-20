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

FILES = []
_seen_paths = set()
for f in raw_files:
    # Windows 的 glob 不区分大小写，*.pdf 和 *.PDF 会命中同一批文件，这里去重
    key = os.path.normcase(os.path.realpath(f))
    if key in _seen_paths:
        continue
    _seen_paths.add(key)
    FILES.append(f)

if not FILES:
    print("  没找到可读文件，支持 .docx/.doc/.txt/.pdf")
    print(f"  扫描位置：{SCRIPT_DIR}")
else:
    if len(FILES) != len(raw_files):
        print(f"  扫描到 {len(raw_files)} 个路径，去掉重复后 {len(FILES)} 个")
    print(f"  去重后，参与建库的文件有 {len(FILES)} 个")

# 2. 文件读取模块
import pdfplumber
from docx import Document

def clean_text(text):
    if not text: return ""
    return '\n'.join([line.strip() for line in text.split('\n') if line.strip()])

# ---------------------------------------------------------------- PDF 解析
# 同一个 PDF 换引擎结果可能天差地别：pdfplumber 遇到 CID 子集字体会吐乱码，
# PyMuPDF（pdf2docx 底层用的也是它）通常能正确还原；扫描件没有文字层，只能 OCR。
def _char_readable(ch):
    o = ord(ch)
    return (
        ch in "\n\t "
        or 0x20 <= o <= 0x7E                  # ASCII
        or 0x4E00 <= o <= 0x9FFF              # 常用汉字
        or 0x3400 <= o <= 0x4DBF              # 扩展 A
        or 0x3000 <= o <= 0x303F              # 中文标点
        or 0xFF00 <= o <= 0xFFEF              # 全角
        or 0x2010 <= o <= 0x2027
        or 0x00B0 <= o <= 0x00FF
    )


def text_quality(text):
    """0~1，越高越干净。乱码的典型特征是私用区字符(U+E000+)和替换符(U+FFFD)。"""
    s = (text or "").strip()
    if not s:
        return 0.0
    total = len(s)
    readable = sum(1 for ch in s if _char_readable(ch))
    broken = sum(1 for ch in s if ch == "\ufffd" or 0xE000 <= ord(ch) <= 0xF8FF)
    return max(0.0, (readable - broken * 2) / total)


def _pdf_text_pdfplumber(path):
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def _pdf_text_pymupdf(path):
    import pymupdf
    parts = []
    with pymupdf.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text", sort=True))
    return "\n".join(parts)


def _read_docx(path):
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return clean_text("\n".join(parts))


def _sibling_docx(pdf_path):
    """PDF 旁边有没有同名的、不比 PDF 旧的 .docx（手动转好的版本）。"""
    cand = os.path.splitext(pdf_path)[0] + ".docx"
    if not os.path.exists(cand):
        return None
    try:
        if os.path.getmtime(cand) + 1 < os.path.getmtime(pdf_path):
            return None
    except OSError:
        return None
    return cand


# ---- OCR（扫描件专用）：rapidocr 优先，paddleocr 备选，结果都过质量门
OCR_ENABLED = os.environ.get("AI_OCR", "1") not in ("0", "false", "False")
OCR_DPI = int(os.environ.get("AI_OCR_DPI", "200"))
OCR_MAX_PAGES = int(os.environ.get("AI_OCR_MAX_PAGES", "30"))
_ocr_engines_cache = {}


def _load_rapidocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def _read_rapidocr(engine, image):
    result, _ = engine(image)
    return [item[1] for item in (result or [])]


def _load_paddleocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="ch", use_doc_orientation_classify=False,
                     use_doc_unwarping=False, use_textline_orientation=False)


def _read_paddleocr(engine, image):
    res = engine.predict(image)[0]
    texts = res.get("rec_texts") if hasattr(res, "get") else None
    return list(texts or [])


def _available_ocr_engines():
    import importlib.util
    engines = []
    if importlib.util.find_spec("rapidocr_onnxruntime"):
        engines.append(("rapidocr", _load_rapidocr, _read_rapidocr))
    if importlib.util.find_spec("paddleocr"):
        engines.append(("paddleocr", _load_paddleocr, _read_paddleocr))
    pref = os.environ.get("AI_OCR_ENGINE", "").strip().lower()
    if pref.startswith("paddle"):
        engines.sort(key=lambda e: 0 if e[0] == "paddleocr" else 1)
    elif pref.startswith("rapid"):
        engines.sort(key=lambda e: 0 if e[0] == "rapidocr" else 1)
    return engines


def _render_pdf_pages(path, dpi, max_pages):
    import numpy as np
    import pymupdf
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            elif pix.n == 1:
                img = np.repeat(img, 3, axis=2)
            pages.append(img)
    return pages


def ocr_pdf(path):
    """整份 PDF 走 OCR；返回纯文本，失败返回 None。"""
    if not OCR_ENABLED:
        return None
    name = os.path.basename(path)
    engines = _available_ocr_engines()
    if not engines:
        print("  [警告] 没有可用的 OCR 引擎，装一个即可：pip install rapidocr-onnxruntime")
        return None
    try:
        pages = _render_pdf_pages(path, OCR_DPI, OCR_MAX_PAGES)
    except Exception as e:
        print(f"  [警告] {name} 渲染成图片失败：{e}")
        return None
    if not pages:
        return None

    print(f"  [OCR] {name}：{len(pages)} 页 @{OCR_DPI}dpi，开始识别（每页约 1-10 秒）")
    for engine_name, loader, reader in engines:
        try:
            engine = _ocr_engines_cache.get(engine_name) or loader()
            _ocr_engines_cache[engine_name] = engine
        except Exception as e:
            print(f"  [警告] {engine_name} 初始化失败：{e}")
            continue

        page_texts, failed = [], False
        for idx, image in enumerate(pages, 1):
            try:
                lines = reader(engine, image)
            except Exception as e:
                print(f"  [警告] {engine_name} 第 {idx} 页识别失败：{e}")
                failed = True
                break
            page_texts.append("\n".join(lines))
            if len(pages) > 3:
                print(f"         · {idx}/{len(pages)} 页")
        if failed:
            continue

        joined = "\n".join(page_texts)
        quality = text_quality(joined)
        if joined.strip() and quality >= 0.5:
            print(f"  [OCR] {engine_name} 完成，质量 {quality:.2f}")
            return clean_text(joined)
        print(f"  [警告] {engine_name} 识别结果质量过低（{quality:.2f}），换下一个引擎")

    print(f"  [警告] {name} OCR 没能识别出可用文字")
    return None


def read_pdf(path):
    """PDF → 文本：同名 docx → 双引擎解析 → OCR。"""
    name = os.path.basename(path)

    # 1) 同名 docx 优先（本地转换过 / 手动用在线工具转好的）
    sib = _sibling_docx(path)
    if sib:
        try:
            text = _read_docx(sib)
            if text and text_quality(text) >= 0.7:
                print(f"  [提示] {name} 旁边有同名 docx，直接读 {os.path.basename(sib)}")
                return text
        except Exception as e:
            print(f"  [警告] 读取 {os.path.basename(sib)} 失败：{e}")

    # 2) 依次尝试两个引擎，取第一个质量合格的
    tried = []
    best = ("", 0.0, "")
    for engine, fn in (("pdfplumber", _pdf_text_pdfplumber), ("pymupdf", _pdf_text_pymupdf)):
        try:
            text = fn(path)
        except Exception as e:
            print(f"  [警告] {name} 用 {engine} 解析失败：{e}")
            continue
        quality = text_quality(text)
        tried.append(f"{engine}={quality:.2f}/{len(text.strip())}字")
        if quality > best[1]:
            best = (text, quality, engine)
        if quality >= 0.85 and len(text.strip()) >= 20:
            if engine != "pdfplumber":
                print(f"  [提示] {name} pdfplumber 结果偏差较大，已改用 {engine}")
            return clean_text(text)

    # 3) 文字层缺失或质量太差 → OCR
    text, quality, engine = best
    if len(text.strip()) < 20 or quality < 0.85:
        if len(text.strip()) < 20:
            print(f"  [提示] {name} 几乎没有文字层（疑似扫描件），改走 OCR")
        ocr_text = ocr_pdf(path)
        if ocr_text and text_quality(ocr_text) >= max(quality, 0.5):
            return ocr_text

    if len(text.strip()) < 20:
        print(f"  [警告] {name} 提取不到文字，OCR 也没成功")
        return None
    if quality < 0.85:
        print(f"  [警告] {name} 解析质量偏低（{'，'.join(tried)}），结果可能不完整")
    return clean_text(text)


def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            return read_pdf(file_path)
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
    ("pymupdf", "pymupdf"),
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