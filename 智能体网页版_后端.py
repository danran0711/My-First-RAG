# -*- coding: utf-8 -*-
"""
网页版本地知识库智能体（Flask 后端）
运行：python 智能体网页版_后端.py   然后浏览器打开 http://localhost:5000
"""
import os, glob, subprocess, sys, re, json, shutil, datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(SCRIPT_DIR, "history_backup")
FAISS_INDEX_DIR = os.path.join(SCRIPT_DIR, "faiss_index")
STATE_FILE = os.path.join(SCRIPT_DIR, "kb_state.json")
UNANSWERED_FILE = os.path.join(SCRIPT_DIR, "unanswered_questions.jsonl")

# 检索命中阈值：最高相似度低于它就算"未命中"（可用环境变量改，如 set AI_SIM_THRESHOLD=0.55）
SIM_THRESHOLD = float(os.environ.get("AI_SIM_THRESHOLD", "0.5"))
FALLBACK_REPLY = "抱歉，当前知识库中暂无相关信息。您的问题已记录，我们会尽快补充完善。"

print("=" * 50)
print("  启动网页版智能体（本地知识库 · 只依据资料回答）")
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

LLM = (os.environ.get("AI_LLM") or "").strip() or \
    pick(["qwen2.5:7b", "qwen2.5:3b", "qwen2.5"]) or (MODELS[0] if MODELS else None)
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
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.messages import HumanMessage

print("  正在处理知识库...")
embeddings = OllamaEmbeddings(model=EMB)


def make_chat(temperature=0.4, num_predict=1000):
    """每次请求新建一个，避免并发共用同一个对象。"""
    return ChatOllama(model=LLM, temperature=temperature, num_ctx=4096, num_predict=num_predict)



# ---- 增量建库：新增 / 修改 / 删除都会触发重建（原来是只增不删，删掉的文件还能被检索出来）
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def index_files_ok():
    return (os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
            and os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.pkl")))


def file_sig(path):
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def decide(state, sigs):
    """返回 (动作, 原因)，动作 ∈ none / load / rebuild / clear"""
    cur = set(sigs)
    known = set(state.get("files", {})) | set(state.get("failed", {}))
    if not cur:
        if known or os.path.isdir(FAISS_INDEX_DIR):
            return "clear", "资料已全部移除"
        return "none", ""
    if state.get("embed") != EMB:
        return "rebuild", f"向量模型变更（{state.get('embed')} → {EMB}）"
    if state.get("metric") != "cosine":
        return "rebuild", "改用余弦相似度，需要重建一次"
    if not index_files_ok():
        return "rebuild", "索引不存在或残缺"
    if cur - known:
        return "rebuild", "发现新资料：" + "、".join(sorted(cur - known))
    if known - cur:
        return "rebuild", "资料被删除：" + "、".join(sorted(known - cur))
    for name in cur:
        old = state.get("files", {}).get(name) or state.get("failed", {}).get(name)
        if old != sigs[name]:
            return "rebuild", f"{name} 已修改"
    if state.get("files") == {}:
        return "rebuild", "上次没有成功入库的资料"
    return "load", ""


def _save_vectorstore(vs):
    """faiss 的 C++ 写盘打不开含中文的绝对路径，切到目录用相对路径写。"""
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(FAISS_INDEX_DIR)
        vs.save_local(".")
    finally:
        os.chdir(cwd)


def load_index():
    cwd = os.getcwd()
    try:
        os.chdir(FAISS_INDEX_DIR)
        return FAISS.load_local(".", embeddings, allow_dangerous_deserialization=True,
                                distance_strategy=DistanceStrategy.COSINE)
    finally:
        os.chdir(cwd)


def build_index(docs_paths, sigs, reason):
    from langchain_core.documents import Document
    print(f"  重建向量库（{reason}）")
    if os.path.isdir(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR, ignore_errors=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    raw_docs, ok, failed = [], {}, {}
    for p in docs_paths:
        name = os.path.basename(p)
        print(f"  读取资料：{name}")
        text = read_file(p)
        if not text or not text.strip():
            failed[name] = sigs[name]
            continue
        stamp = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
        raw_docs.append(Document(page_content=text, metadata={"source": name, "mtime": stamp}))
        ok[name] = sigs[name]

    vs = None
    if raw_docs:
        chunks = splitter.split_documents(raw_docs)
        print(f"  切分 {len(chunks)} 段，正在向量化...")
        vs = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
        _save_vectorstore(vs)
    else:
        print("  没有可入库的资料")

    save_state({"version": 2, "embed": EMB, "metric": "cosine", "files": ok, "failed": failed})
    return vs


def ensure_kb():
    sigs = {os.path.basename(p): file_sig(p) for p in FILES}
    state = load_state()
    action, reason = decide(state, sigs)

    if action == "none":
        print("  知识库为空（把资料放进目录后重启）")
        return None
    if action == "clear":
        print(f"  清空向量库（{reason}）")
        shutil.rmtree(FAISS_INDEX_DIR, ignore_errors=True)
        save_state({"version": 2, "embed": EMB, "metric": "cosine", "files": {}, "failed": {}})
        return None
    if action == "rebuild":
        vs = build_index(FILES, sigs, reason)
        print(f"  知识库就绪（{getattr(getattr(vs, 'index', None), 'ntotal', 0)} 段）")
        return vs

    print("  加载已有向量库...")
    try:
        vs = load_index()
        print(f"  知识库就绪（{getattr(vs.index, 'ntotal', 0)} 段）")
        return vs
    except Exception as e:
        print(f"  [警告] 向量库加载失败（{e}），改为重建")
        return build_index(FILES, sigs, "加载失败")


vs = ensure_kb()

def clean_answer(text):
    if not text: return ""
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'(?m)^\s*#{1,6}\s*', '', text)
    text = re.sub(r'`{1,3}', '', text)
    text = re.sub(r'(?m)^\s*>\s?', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def stream_llm(prompt):
    """真正逐块产出，/chat 聚合、/chat/stream 直接往前端吐。"""
    chat = make_chat()
    for c in chat.stream([HumanMessage(content=prompt)]):
        piece = getattr(c, "content", None)
        if piece:
            yield piece


def ask_llm(prompt):
    ans = ""
    try:
        for piece in stream_llm(prompt):
            ans += piece
    except Exception:
        try:
            resp = make_chat().invoke([HumanMessage(content=prompt)])
            ans = resp.content if resp else ""
        except Exception as e2:
            ans = f"（模型调用失败：{e2}）"
    return clean_answer(ans)


# ---------------------------------------------------------------- 多轮上下文
MAX_HISTORY_MSGS = 6      # 只带最近 6 条（3 轮）
MAX_HISTORY_CHARS = 400   # 单条最多 400 字，防止把上下文撑爆


def normalize_history(raw):
    """前端传来的历史可能是 bot/user，也可能是 assistant/user；统一成两种角色。"""
    out = []
    if not isinstance(raw, list):
        return out
    for item in raw[-MAX_HISTORY_MSGS:]:
        if not isinstance(item, dict):
            continue
        role = "assistant" if item.get("role") in ("bot", "assistant") else "user"
        content = (item.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content[:MAX_HISTORY_CHARS]})
    return out


def history_to_text(hist):
    return "\n".join(("用户：" if m["role"] == "user" else "助手：") + m["content"] for m in hist)


REWRITE_PROMPT = (
    "你是检索词改写器，不是问答助手。任务只有一个：把【最新】这句话变成一条能独立检索的问题。\n"
    "硬性规则：\n"
    "1. 只输出一行问题本身。禁止回答问题、禁止解释、禁止补充背景知识或数据。\n"
    "2. 不超过 30 个字。\n"
    "3. 原句已经能独立检索时，原样输出，一个字都不要改。\n"
    "4. 只允许补全省略的指代和主题词，不许添加原句和历史里没有的信息。\n\n"
    "示例1（原句完整 → 原样输出）：\n"
    "历史：（无）\n最新：垃圾分类\n改写：垃圾分类\n\n"
    "示例2（追问 → 补全）：\n"
    "历史：\n用户：云南师范大学军训\n助手：军训通常安排在九月。\n最新：那怎么报名\n改写：云南师范大学军训怎么报名\n\n"
    "历史：\n{history}\n\n最新：{q}\n改写（只输出一行问题，30字以内）："
)
# 改写跑偏的判定阈值：超过 50 字，或超过原句 3 倍
MAX_REWRITE_CHARS = 50
MAX_REWRITE_RATIO = 3


def _rewrite_runaway(original, rewritten):
    return (len(rewritten) > MAX_REWRITE_CHARS
            or len(rewritten) > len(original) * MAX_REWRITE_RATIO)


# 明显是追问的说法：改写没生效时，用它判断要不要把上一轮问题拼进去
FOLLOWUP_RE = re.compile(r"^(那|这|它|他|她|那么|还有|再|然后|可是|但|如果)")
# 句子自带指代 → 必须结合历史才能检索
PRONOUN_RE = re.compile(r"(它|他|她|这个|那个|这些|那些|上面|刚才|之前|他们|她们|该)")


def _needs_rewrite(q):
    """只有"追问"才需要改写。自带主题的完整问题直接放行，避免模型画蛇添足。"""
    s = (q or "").strip()
    if not s:
        return False
    return bool(FOLLOWUP_RE.match(s)) or bool(PRONOUN_RE.search(s)) or len(s) <= 6


def _cjk_bigrams(text):
    return {text[i:i + 2] for i in range(len(text) - 1)
            if "\u4e00" <= text[i] <= "\u9fff" and "\u4e00" <= text[i + 1] <= "\u9fff"}


def _keeps_keywords(original, rewritten):
    """改写结果里得留着原句的关键词，否则说明模型把问题改跑了（比如把“病假”丢了）。"""
    grams = _cjk_bigrams(original)
    return not grams or any(g in rewritten for g in grams)


def rewrite_query(q, hist):
    """把“那病假呢？”这类追问补全成能独立检索的问题；完整问题原样放行。"""
    if not hist:
        return q
    if not _needs_rewrite(q):
        print(f"  [改写] {q} 已是完整问题，跳过改写")
        return q
    prompt = REWRITE_PROMPT.format(history=history_to_text(hist), q=q)
    try:
        chat = make_chat(temperature=0.0, num_predict=80)
        raw = "".join(getattr(c, "content", "") or ""
                      for c in chat.stream([HumanMessage(content=prompt)]))
    except Exception as e:
        print(f"  [提示] 问题改写失败，用原问题检索：{e}")
        return q

    new_q = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    new_q = new_q.strip("“”\"'` ").rstrip("。？?！!")

    # 跑偏保险：模型不补全问题，而是把答案写出来的话，这里拦下来
    if new_q and _rewrite_runaway(q, new_q):
        print(f"  [改写-丢弃] 改写跑偏（{len(new_q)} 字，原句 {len(q)} 字）：{new_q[:40]}")
        new_q = ""

    if new_q and new_q == q:
        print(f"  [改写] {q}  ->  原样（已是完整问题）")
        return q

    if not new_q or not _keeps_keywords(q, new_q):
        # 改写没生效/改跑了，但它明显是追问 → 把上一轮问题拼在前面，保证检索词里有上下文
        if FOLLOWUP_RE.match(q) or len(q) <= 6:
            prev = next((m["content"] for m in reversed(hist) if m["role"] == "user"), "")
            if prev:
                merged = f"{prev} {q}".strip()[:120]
                print(f"  [改写-拼接] {q}  ->  {merged}")
                return merged
        return q
    print(f"  [改写] {q}  ->  {new_q}")
    return new_q


def build_answer_prompt(q, hist, local_ctx):
    hist_txt = history_to_text(hist) if hist else "（无）"
    return (
        "你是一个严谨的校园助手。只能依据下面的【参考文档】回答，不得使用你自己的知识补充。\n"
        "【硬性要求】：\n"
        "1. 每条结论都必须能在【参考文档】里找到依据；文档没写的内容一律不答。\n"
        "2. 严禁编造：不臆测、不补全数字/日期/人名、不用常识推断，也不要输出“根据资料显示”这类前缀。\n"
        "3. 参考文档不足以回答时，只回一句“抱歉，当前知识库中暂无相关信息。”，不要给任何猜测。\n"
        "4. 用简明分点作答，严禁大段复制原文，不要输出“本地资料总结/以下是回答”等废话。\n"
        "5. 结合【对话历史】理解“它/这个/上面说的”等指代，但回答里不要复述历史。\n\n"
        f"【对话历史】\n{hist_txt}\n\n"
        f"【参考文档】\n{local_ctx}\n\n"
        f"问题：{q}\n"
        "请直接给出回答（不要任何前缀）："
    )


def search_hits(query, k=3):
    """返回 [(余弦相似度, 文档)]，按相似度降序。索引里存的是归一化向量上的平方 L2。"""
    if vs is None:
        return []
    pairs = vs.similarity_search_with_score(query, k=k)
    hits = [(max(0.0, 1.0 - float(dist) / 2.0), doc) for doc, dist in pairs]
    hits.sort(key=lambda x: -x[0])
    return hits


def describe_sources(hits):
    """把命中的文档去重成「（来源：a(xx)、b(yy)）」。"""
    seen, uniq = set(), []
    for _, doc in hits:
        src = doc.metadata.get("source", "未知")
        stamp = doc.metadata.get("mtime", "")
        if (src, stamp) in seen:
            continue
        seen.add((src, stamp))
        uniq.append(f"{src}({stamp})" if stamp else src)
    return "（来源：" + "、".join(uniq) + "）" if uniq else ""


def gather_context(q, hist):
    """改写问题 → 检索 → 判断是否命中。

    返回 (检索用的问题, 参考文档, 最高相似度, hits)；是否命中有调用方按阈值判断。
    """
    rq = rewrite_query(q, hist)
    hits = search_hits(rq, k=3)
    max_sim = hits[0][0] if hits else 0.0

    if vs is None or max_sim < SIM_THRESHOLD:
        print(f"  [未命中] 最高相似度 {max_sim:.2f} < 阈值 {SIM_THRESHOLD}")
        return rq, "", max_sim, hits

    local_ctx = "\n".join(doc.page_content for _, doc in hits)
    return rq, local_ctx, max_sim, hits


# ---- 未命中问题记录
GREETING_PAT = re.compile(
    r"^(你好|您好|hi|hello|哈喽|在吗|在么|谢谢|多谢|感谢|再见|拜拜|"
    r"早上好|中午好|晚上好|晚安|嗯|哦|好的|ok|收到)[!！。.,，?？~～\s]*$", re.I)
MEANINGFUL_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]")


def is_recordable(question):
    """寒暄、长度 <4、纯数字/标点的问题不记录。"""
    s = (question or "").strip()
    if len(s) < 4:
        return False
    if GREETING_PAT.match(s):
        return False
    if not MEANINGFUL_RE.search(s):
        return False
    return True


def record_unanswered(question, max_sim, hits):
    """把未命中的问题追加到 unanswered_questions.jsonl。"""
    if not is_recordable(question):
        print(f"  [未命中-忽略] {question[:30]}（寒暄/过短/无实义，不记录）")
        return False

    docs, seen = [], set()
    for sim, doc in hits[:3]:
        src = doc.metadata.get("source", "未知")
        if src in seen:
            continue
        seen.add(src)
        docs.append({"source": src, "similarity": round(sim, 4)})
        if len(docs) >= 3:
            break

    item = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question.strip(),
        "max_similarity": round(float(max_sim), 4),
        "retrieved_docs": docs,
        "status": "pending",
    }
    try:
        with open(UNANSWERED_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  [未命中] {question[:30]}（最高相似度 {max_sim:.2f}）已记录")
        return True
    except Exception as e:
        print(f"  [警告] 写入未命中记录失败：{e}")
        return False


# 6. HTTP 接口
from flask import Flask, request, jsonify, send_from_directory, Response

# static_folder=None：不让 Flask 把整个脚本目录当静态目录暴露出去
app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(SCRIPT_DIR, "index.html")


def _sse(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True) or {}
    q = (data.get("message") or "").strip()
    if not q:
        return jsonify({"reply": "请输入问题。"})

    hist = normalize_history(data.get("history"))
    rq, local_ctx, max_sim, hits = gather_context(q, hist)

    if max_sim < SIM_THRESHOLD:
        record_unanswered(q, max_sim, hits)
        return jsonify({"reply": FALLBACK_REPLY, "fallback": True,
                        "query": rq, "max_similarity": round(max_sim, 4)})

    ans = ask_llm(build_answer_prompt(q, hist, local_ctx))
    source_str = describe_sources(hits)
    if source_str:
        ans += "\n\n" + source_str

    return jsonify({"reply": ans, "fallback": False,
                    "query": rq, "max_similarity": round(max_sim, 4)})


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    q = (data.get("message") or "").strip()
    if not q:
        return jsonify({"reply": "请输入问题。"}), 400
    hist = normalize_history(data.get("history"))

    def gen():
        try:
            rq, local_ctx, max_sim, hits = gather_context(q, hist)
            if max_sim < SIM_THRESHOLD:
                record_unanswered(q, max_sim, hits)
                yield _sse({"type": "meta", "fallback": True, "sources": "",
                            "query": rq, "max_similarity": round(max_sim, 4)})
                yield _sse({"type": "token", "text": FALLBACK_REPLY})
                yield _sse({"type": "done"})
                return

            yield _sse({"type": "meta", "fallback": False,
                        "sources": describe_sources(hits),
                        "query": rq, "max_similarity": round(max_sim, 4)})
            for piece in stream_llm(build_answer_prompt(q, hist, local_ctx)):
                yield _sse({"type": "token", "text": piece})
            yield _sse({"type": "done"})
        except Exception as e:
            print(f"  [错误] 流式处理失败：{e}")
            yield _sse({"type": "error", "message": str(e)})

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    print("=" * 50)
    print("  服务已启动：浏览器打开 http://localhost:5000")
    print("=" * 50)
    # threaded=True：多个请求并发处理，不然一个长回答会把其他请求全堵住
    app.run(host="127.0.0.1", port=5000, threaded=True)