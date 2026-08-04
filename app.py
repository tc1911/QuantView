"""QuantView - 本地数据分析助手 (Flask 后端入口)

启动方式:
    conda activate deepanalyze
    pip install flask
    python app.py
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_file
import os
import io
import shutil
import json
import sys
import tempfile
import subprocess
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非 GUI 后端，兼容 WSL / 无头环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import urllib.request
import urllib.error
import time
import re
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from file_processing import (
    ALLOWED_EXTENSIONS,
    allowed_file,
    df_summary,
    extract_hard_numbers_by_sheet,
    extract_hard_numbers_from_bytes,
    read_file,
)


# ── 运行模式配置 ──
DEBUG_MODE = os.environ.get("DEEPANALYZE_DEBUG", "").lower() in ("1", "true", "yes")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_THINKING = os.environ.get("DEEPSEEK_THINKING", "").lower() in ("1", "true", "yes")
# 思考模式用 deepseek-reasoner，普通模式用 deepseek-chat
DEEPSEEK_MODEL = "deepseek-reasoner" if DEEPSEEK_THINKING else "deepseek-chat"

# ── 多节点分布式配置 ──
# DEEPANALYZE_NODES = "node1=http://192.168.1.10:5000,node2=http://192.168.1.11:5000"
# 设置了该变量即为主节点模式：工作表任务轮流分发给加速节点，主节点只生成总览
_NODE_LIST = os.environ.get("DEEPANALYZE_NODES", "").strip()
_NODE_TIMEOUT = float(os.environ.get("DEEPANALYZE_NODE_TIMEOUT", "600"))
_SHEET_MAX_TOKENS = 8192  # 每个工作表任务的最大输出 token 数
# 本地 llama-server 调优参数（仅 GGUF 路径附加，DeepSeek API 不受影响）：
# - chat_template_kwargs.enable_thinking=false：Qwen3 思考模式会泄漏思维链进正文
# - repeat_penalty / repeat_last_n：长输出重复循环防护（默认 1.0=关闭，模型会退化复读）
# 本地模型思考模式开关（默认关：Qwen3 思考会泄漏进正文/拖慢输出；
# 想开启时设 DEEPANALYZE_LOCAL_THINKING=true，前端会把 <think> 块折叠展示）
_ENABLE_LOCAL_THINKING = os.environ.get("DEEPANALYZE_LOCAL_THINKING", "").lower() in ("1", "true", "yes")
# 上下文长度（默认 65536）。同机多实例时建议加速节点调小（如 32768）省内存：
# 32B 模型 64K 上下文 KV 缓存约 16GB，双实例会很紧
_CONTEXT = int(os.environ.get("DEEPANALYZE_CONTEXT", "65536"))
_LOCAL_SERVER_BODY = {
    "chat_template_kwargs": {"enable_thinking": _ENABLE_LOCAL_THINKING},
    "repeat_penalty": 1.15,
    "repeat_last_n": 512,
}
# 采样温度（默认 0.5：报告是确定性任务，低温度降低单次跑飞/对话式输出概率）
_TEMPERATURE = float(os.environ.get("DEEPANALYZE_TEMPERATURE", "0.5"))
# 无头模式（加速节点专用）：不提供 Web 界面，只保留 /analyze/sheet 任务接口与命令行日志
_HEADLESS = os.environ.get("DEEPANALYZE_HEADLESS", "").lower() in ("1", "true", "yes")

if DEBUG_MODE:
    print("=" * 60)
    print(" 调试模式已启用 - 使用 DeepSeek API 代替本地模型")
    if DEEPSEEK_THINKING:
        print(" 思考模式已启用 - 使用 deepseek-reasoner 深度推理")
    if not DEEPSEEK_API_KEY:
        print(" 警告: 未设置 DEEPSEEK_API_KEY 环境变量！")
        print(" 设置方式: export DEEPSEEK_API_KEY=sk-xxxx")
    else:
        print(f" API Key: {DEEPSEEK_API_KEY[:10]}...")
    print("=" * 60)
else:
    pass  # 本地推理仅支持 GGUF（外部 llama-server），无 torch/transformers 依赖
    Llama = None

# ── 中文字体配置 ──
_CHINESE_FONT_PATH = None
_FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _ensure_chinese_font():
    """确保有可用的中文字体，优先级：项目内置 > 系统字体 > 自动下载。

    返回字体文件路径，失败返回 None。
    """
    global _CHINESE_FONT_PATH
    if _CHINESE_FONT_PATH:
        return _CHINESE_FONT_PATH

    # ── 1. 检查项目内置字体目录 ──
    os.makedirs(_FONTS_DIR, exist_ok=True)
    for _fname in os.listdir(_FONTS_DIR):
        _fp = os.path.join(_FONTS_DIR, _fname)
        if _fname.lower().endswith((".ttf", ".otf", ".ttc")) and os.path.isfile(_fp):
            if _verify_font_has_chinese(_fp):
                _CHINESE_FONT_PATH = _fp
                print(f"[字体] 使用内置字体: {_fname}")
                return _CHINESE_FONT_PATH

    # ── 2. 检查系统字体路径 ──
    _SYSTEM_PATHS = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]
    for _path in _SYSTEM_PATHS:
        if os.path.isfile(_path):
            _CHINESE_FONT_PATH = _path
            print(f"[字体] 使用系统字体: {_path}")
            return _CHINESE_FONT_PATH

    # ── 3. Linux: 用 fc-list 查找 ──
    try:
        import subprocess
        _result = subprocess.run(
            ["fc-list", ":lang=zh", "file"],
            capture_output=True, text=True, timeout=5
        )
        for _line in _result.stdout.strip().split("\n"):
            _line = _line.strip()
            if ":" in _line:
                _candidate = _line.split(":", 1)[0].strip()
                if os.path.isfile(_candidate) and _candidate.lower().endswith((".ttf", ".otf", ".ttc")):
                    _CHINESE_FONT_PATH = _candidate
                    print(f"[字体] 使用 fc-list 字体: {_candidate}")
                    return _CHINESE_FONT_PATH
    except Exception:
        pass

    # ── 4. 自动下载内置字体 ──
    try:
        _path = _download_chinese_font()
        if _path:
            _CHINESE_FONT_PATH = _path
            return _CHINESE_FONT_PATH
    except Exception as e:
        print(f"[字体] 下载失败: {e}")

    print("[字体] 警告: 未找到中文字体，图表中文可能显示为方框")
    print("[字体] 手动修复: Ubuntu 执行 sudo apt install fonts-wqy-microhei")
    return None


def _verify_font_has_chinese(font_path):
    """快速验证字体文件是否包含中文字符。"""
    try:
        from matplotlib.ft2font import FT2Font
        _font = FT2Font(font_path)
        # 检查几个常用汉字
        _test_chars = "\u6570\u636e\u5206\u6790\u503c"  # 数据分析值
        for _ch in _test_chars:
            if _font.get_char_index(ord(_ch)) == 0:
                return False
        return True
    except Exception:
        return False


def _download_chinese_font():
    """下载一个开源中文字体到项目 fonts/ 目录。"""
    # 来源: Noto Sans SC Regular (Apache 2.0, ~8MB .otf)
    _url = (
        "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
        "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
    )
    _dest = os.path.join(_FONTS_DIR, "NotoSansCJKsc-Regular.otf")

    print(f"[字体] 正在下载中文字体 (~8MB)...")
    print(f"[字体] 来源: {_url}")

    import urllib.request
    urllib.request.urlretrieve(_url, _dest)

    if os.path.isfile(_dest) and os.path.getsize(_dest) > 100000:  # >100KB
        print(f"[字体] 下载完成: {_dest}")
        return _dest
    else:
        if os.path.isfile(_dest):
            os.remove(_dest)
        raise RuntimeError("下载的文件无效")


def _get_font_prop(size=None):
    """获取中文字体 FontProperties 对象。"""
    _ensure_chinese_font()
    if _CHINESE_FONT_PATH:
        return fm.FontProperties(fname=_CHINESE_FONT_PATH, size=size)
    return None


# 启动时尝试加载字体，并重建 matplotlib 字体缓存
_ensure_chinese_font()
if _CHINESE_FONT_PATH:
    fm.fontManager.addfont(_CHINESE_FONT_PATH)
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

app = Flask(__name__, static_folder="static", static_url_path="")

# 限制上传文件总大小为 200MB（支持多文件）
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# ── 模型发现与选择（仅支持 GGUF）──
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_model = None
_tokenizer = None
MODEL_PATH = None
MODEL_TYPE = "gguf"


def _scan_models():
    """扫描 models/ 目录，返回可用 GGUF 模型列表 {名称: 路径}。"""
    if not os.path.isdir(_MODELS_DIR):
        print(f"[模型] models/ 目录不存在: {_MODELS_DIR}")
        return {}

    _HF_SIGNATURE = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    available = {}

    all_items = os.listdir(_MODELS_DIR)
    all_dirs = [d for d in all_items if os.path.isdir(os.path.join(_MODELS_DIR, d))]
    print(f"[模型] 扫描 models/ 目录: 发现 {len(all_dirs)} 个子目录, {len(all_items)} 个条目")

    for entry in all_dirs:
        full = os.path.join(_MODELS_DIR, entry)

        # 目录内的 GGUF 文件
        gguf_files = [f for f in os.listdir(full) if f.endswith(".gguf")]
        if gguf_files:
            available[entry] = os.path.join(full, gguf_files[0])
            print(f"  ✓ {entry} (GGUF: {gguf_files[0]})")
            continue

        # 仅含 HuggingFace 文件的目录：本版本不支持 HF 路线
        if any(os.path.isfile(os.path.join(full, sig)) for sig in _HF_SIGNATURE):
            print(f"  ✗ {entry} — HuggingFace 模型目录（本版本仅支持 GGUF，请下载 .gguf 文件）")
            continue

        subfiles = os.listdir(full)[:5]
        print(f"  ✗ {entry} — 无 GGUF 文件，目录内容: {subfiles}")

    # ── 扫描 models/ 根目录下的 .gguf 文件 ──
    for item in all_items:
        full = os.path.join(_MODELS_DIR, item)
        if os.path.isfile(full) and item.endswith(".gguf"):
            name = item.replace(".gguf", "")
            if name not in available:
                available[name] = full
                print(f"  ✓ {name} (GGUF 单文件: {item})")

    return available


def _select_model():
    """根据可用模型数量和环境变量选择模型路径。

    优先级: DEEPANALYZE_MODEL 环境变量 > 单模型自动选择 > 多模型手动选择
    """
    if DEBUG_MODE:
        return None, "gguf"

    available = _scan_models()
    if not available:
        print("=" * 60)
        print(" 错误: models/ 目录下未找到任何 GGUF 模型！")
        print(f" 目录: {_MODELS_DIR}")
        print(" 请放入 .gguf 模型文件（如 Qwen3-32B-Q4_K_M.gguf）")
        print(" 使用调试模式可跳过本地模型:")
        print("   DEEPANALYZE_DEBUG=true DEEPSEEK_API_KEY=sk-xxx python app.py")
        print("=" * 60)
        if not DEBUG_MODE:
            exit(1)
        return None, "gguf"

    env_model = os.environ.get("DEEPANALYZE_MODEL", "").strip()
    if env_model:
        if env_model in available:
            return available[env_model], "gguf"
        print("=" * 60)
        print(f" 错误: 指定的模型 '{env_model}' 未找到")
        print(f" 可用模型: {', '.join(available.keys())}")
        print("=" * 60)
        exit(1)

    if len(available) == 1:
        name, path = next(iter(available.items()))
        print(f"[模型] 自动选择唯一可用模型: {name} (GGUF)")
        return path, "gguf"

    # 多个模型，手动选择
    sorted_models = sorted(available.items())
    print("=" * 60)
    print(f" 发现 {len(available)} 个可用模型:")
    for i, (name, _) in enumerate(sorted_models, 1):
        print(f"   [{i}] {name}  (GGUF)")
    print()

    while True:
        try:
            choice = input(f" 请选择模型 [1-{len(sorted_models)}]: ").strip()
            idx = int(choice)
            if 1 <= idx <= len(sorted_models):
                name, path = sorted_models[idx - 1]
                print(f"\n 已选择: {name} (GGUF)")
                print("=" * 60)
                return path, "gguf"
            print(f" 请输入 1 到 {len(sorted_models)} 之间的数字")
        except (ValueError, EOFError):
            print(" 输入无效，请输入数字")
        except KeyboardInterrupt:
            print("\n\n 已取消")
            exit(0)


def get_model_and_tokenizer():
    """懒加载本地模型（GGUF）：启动外部 llama-server 进程。调试模式下跳过。"""
    global _model, _tokenizer, MODEL_PATH

    if DEBUG_MODE:
        return None, None

    if MODEL_PATH is None:
        raise RuntimeError("未配置模型路径，请检查 models/ 目录")

    if _model is None:
        model_name = os.path.basename(MODEL_PATH)
        print("=" * 60)
        print(f"正在加载模型: {model_name} (GGUF)")
        print(f"路径: {MODEL_PATH}")
        print("首次加载需 10-30 秒，请稍候...")
        print("=" * 60)

        # ── GGUF 模型：启动外部 llama-server 进程 ──
        import subprocess
        import time

        # 查找 llama-server 二进制（优先项目目录 llama-server/ 文件夹）
        server_bin = os.environ.get("LLAMA_SERVER_PATH", "")
        if not server_bin:
            _project_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(_project_dir, "llama-server", "llama-server.exe"),
                os.path.join(_project_dir, "llama-server", "llama-server"),
                os.path.join(_project_dir, "llama-server.exe"),
                os.path.join(_project_dir, "llama-server"),
                "llama-server", "llama-server.exe",
                "C:/Users/tc191/llama-cpp/llama-server.exe",
            ]
            for c in candidates:
                if os.path.isfile(c) or shutil.which(c):
                    server_bin = c
                    break

        if not server_bin:
            raise RuntimeError(
                "未找到 llama-server 二进制。请把 llama.cpp 的 llama-server 放到项目目录的\n"
                "llama-server/ 文件夹（即 <项目目录>/llama-server/llama-server），\n"
                "或设置环境变量 LLAMA_SERVER_PATH 指向二进制路径。\n"
                "下载: https://github.com/ggerganov/llama.cpp/releases"
            )

        port = 8080
        # 检查端口是否已被占用，自动递增
        while True:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            occupied = s.connect_ex(("127.0.0.1", port)) == 0
            s.close()
            if not occupied:
                break
            port += 1

        print(f"[GGUF] 启动 llama-server: {server_bin}")
        print(f"[GGUF] 端口: {port}, 模型: {MODEL_PATH}")

        # llama-server 输出写入日志文件（不能用 PIPE：没人读取会堵塞进程，且超时后无法看到原因）
        _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama-server.log")
        _logf = open(_log_path, "a", encoding="utf-8", errors="replace")
        # GPU 显存层数：默认全量卸载；显存不足时可调低（如 DEEPANALYZE_NGL=60/0）
        _ngl = os.environ.get("DEEPANALYZE_NGL", "99")
        _llama_proc = subprocess.Popen(
            [server_bin, "-m", MODEL_PATH, "--port", str(port),
             "-ngl", _ngl, "-c", str(_CONTEXT), "--host", "127.0.0.1",
             # 服务端禁用 Qwen3 思考模式（请求级 chat_template_kwargs 对部分模型无效）
             "--chat-template-kwargs", '{"enable_thinking": %s}' % ("true" if _ENABLE_LOCAL_THINKING else "false")],
            stdout=_logf, stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"[GGUF] llama-server 输出日志: {_log_path}")

        # 等待服务就绪（最多 60 秒）
        print("[GGUF] 等待服务就绪...")
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                import urllib.request as _ur
                r = _ur.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                if r.status == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            _llama_proc.kill()
            # 把 llama-server 自己的最后输出带进错误信息，便于定位原因
            _tail = ""
            try:
                with open(_log_path, encoding="utf-8", errors="replace") as _lf:
                    _tail = "\n".join(_lf.read().splitlines()[-25:])
            except Exception:
                pass
            raise RuntimeError(
                "llama-server 启动超时（60秒内 /health 未就绪）\n"
                f"--- llama-server 最近输出（{_log_path}）---\n{_tail}"
            )

        # 设置全局 API URL，推理代码自动走 API 路径
        global DEEPSEEK_API_URL
        DEEPSEEK_API_URL = f"http://127.0.0.1:{port}/v1/chat/completions"
        _model = f"llama-server:{port}"
        _tokenizer = f"llama-server:{port}"

        # 注册退出清理
        import atexit
        atexit.register(lambda: _llama_proc.kill() if _llama_proc.poll() is None else None)

        print(f"[GGUF] llama-server 已就绪 -> {DEEPSEEK_API_URL}")

    return _model, _tokenizer


# ── 启动时扫描并加载模型 ──
if not DEBUG_MODE:
    print("=" * 60)
    print(" 模型发现")
    print("=" * 60)
    MODEL_PATH, MODEL_TYPE = _select_model()
    if MODEL_PATH:
        print(f" 已选择模型: {os.path.basename(MODEL_PATH)} (GGUF)")
        print(f" 路径: {MODEL_PATH}")
        print("=" * 60)
        get_model_and_tokenizer()
    else:
        print("=" * 60)


@app.route("/")
def index():
    """前端入口页面（无头模式不提供 Web 界面，仅提示）"""
    if _HEADLESS:
        return Response(
            "加速节点模式：无 Web 界面。\n本实例仅提供 /analyze/sheet 任务接口，请由主节点调用。",
            mimetype="text/plain; charset=utf-8",
            status=404,
        )
    resp = app.send_static_file("index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def _send_report_email(question, filenames, full_output, charts=None):
    """分析完成后把报告发到配置的邮箱（异步线程调用；未配置收件人则跳过）。

    环境变量: DEEPANALYZE_MAIL_TO(收件人，逗号分隔) DEEPANALYZE_MAIL_SMTP(服务器)
    DEEPANALYZE_MAIL_PORT(默认465) DEEPANALYZE_MAIL_USER DEEPANALYZE_MAIL_PASS
    DEEPANALYZE_MAIL_FROM(默认同 USER)。465=SSL，其余端口=STARTTLS。
    附件：报告全文 .md + 全部图表 PNG 打包 zip。发送失败只打日志，不影响主流程。
    """
    import smtplib
    import zipfile
    import html as _html
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication

    to = os.environ.get("DEEPANALYZE_MAIL_TO", "").strip()
    if not to:
        return  # 未配置收件人，不启用邮件
    smtp_host = os.environ.get("DEEPANALYZE_MAIL_SMTP", "").strip()
    if not smtp_host:
        print("[邮件] 未配置 DEEPANALYZE_MAIL_SMTP，跳过邮件发送")
        return
    try:
        port = int(os.environ.get("DEEPANALYZE_MAIL_PORT", "465") or 465)
    except ValueError:
        port = 465
    user = os.environ.get("DEEPANALYZE_MAIL_USER", "").strip() or None
    pwd = os.environ.get("DEEPANALYZE_MAIL_PASS", "").strip() or None
    frm = os.environ.get("DEEPANALYZE_MAIL_FROM", "").strip() or user or to.split(",")[0].strip()

    try:
        msg = MIMEMultipart()
        msg["From"] = frm
        msg["To"] = to
        msg["Subject"] = f"【QuantView】数据分析报告已完成 {time.strftime('%Y-%m-%d %H:%M')}"
        body = (
            "<h3>QuantView 数据分析报告已生成</h3>"
            f"<p><b>分析问题：</b>{_html.escape(question)}</p>"
            f"<p><b>分析文件：</b>{_html.escape('、'.join(filenames))}</p>"
            f"<p><b>完成时间：</b>{time.strftime('%Y-%m-%d %H:%M:%S')}</p>"
            "<p>报告全文与图表见附件。</p>"
        )
        msg.attach(MIMEText(body, "html", "utf-8"))

        # 报告全文 .md
        md_part = MIMEText(full_output, "plain", "utf-8")
        md_part.add_header("Content-Disposition", "attachment",
                           filename=("utf-8", "", f"QuantView_报告_{time.strftime('%Y-%m-%d')}.md"))
        msg.attach(md_part)

        # 图表打包 zip
        if charts:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, (title, b64) in enumerate(charts, 1):
                    try:
                        png = base64.b64decode(str(b64).split(",", 1)[-1])
                        zf.writestr(f"图表_{idx:02d}.png", png)
                    except Exception:
                        continue
            zip_buf.seek(0)
            zip_part = MIMEApplication(zip_buf.read(), _subtype="zip")
            zip_part.add_header("Content-Disposition", "attachment",
                                filename=("utf-8", "", f"QuantView_图表_{time.strftime('%Y-%m-%d')}.zip"))
            msg.attach(zip_part)

        if port == 465:
            server = smtplib.SMTP_SSL(smtp_host, port, timeout=30)
        else:
            server = smtplib.SMTP(smtp_host, port, timeout=30)
            try:
                server.starttls()
            except smtplib.SMTPException:
                pass
        try:
            if user:
                server.login(user, pwd)
            server.sendmail(frm, [x.strip() for x in to.split(",") if x.strip()], msg.as_string())
        finally:
            server.quit()
        print(f"[邮件] 已发送分析报告到 {to}")
    except Exception as e:
        print(f"[邮件] 发送失败: {e}")


@app.route("/analyze", methods=["POST"])
def analyze():
    """数据分析接口

    POST multipart/form-data:
        files:  Excel/CSV 文件列表 (xlsx/xls/csv)，支持多文件上传
        question: 用户分析问题

    返回:
        {"result": "分析报告内容"}
        或
        {"error": "错误描述"}
    """
    # 1. 检查文件
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "未检测到上传文件"}), 400

    valid_files = []
    for f in files:
        if f.filename == "":
            continue
        if not allowed_file(f.filename):
            return jsonify({"error": f"不支持的文件格式: {f.filename}，仅支持 xlsx/xls/csv"}), 400
        valid_files.append(f)

    if not valid_files:
        return jsonify({"error": "未选择有效的文件"}), 400

    # 2. 检查问题
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "分析问题不能为空"}), 400

    # 3. 保存文件字节（追问会话与分析共用）
    from werkzeug.datastructures import FileStorage as _FS
    file_buffers = []
    fresh_files = []
    for f in valid_files:
        buf = io.BytesIO()
        f.save(buf)
        data = buf.getvalue()
        file_buffers.append((f.filename, data))
        fresh_files.append(_FS(stream=io.BytesIO(data), filename=f.filename))
    session_id = _new_followup_session(file_buffers)

    # 4. 执行分析
    try:
        output = perform_analysis(fresh_files, question)
        _SESSIONS[session_id]["report"] = output["result"]
        _persist_session(session_id)
        # 完成 → 异步发送邮件（可选，未配置收件人时自动跳过）
        try:
            threading.Thread(target=_send_report_email,
                             args=(question, [f[0] for f in file_buffers],
                                   output["result"], output.get("images", [])),
                             daemon=True).start()
        except Exception:
            pass
        return jsonify({
            "result": output["result"],
            "images": output.get("images", []),
            "mode": output.get("mode", "single"),
            "nodes": output.get("nodes", []),
            "session": session_id,
        })
    except Exception as e:
        return jsonify({"error": f"分析过程出错: {str(e)}"}), 500


def _fig_to_base64(fig):
    """将 matplotlib Figure 转为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _generate_charts_from_json(model_output):
    """从 AI 输出的报告文本中提取 chartjson 并生成图表。

    Returns:
        [(title, base64_img), ...] 列表
    """
    import re as _re
    charts = []
    fp_m = _get_font_prop(size=14)
    fp_sm = _get_font_prop(size=10)

    # 提取全部 ```chartjson ... ``` 块（分布式模式下每个分项各有一个）
    matches = list(_re.finditer(r'```chartjson\s*\n(.*?)\n```', model_output, _re.DOTALL))
    if not matches:
        return charts

    for match in matches:
        raw = match.group(1)
        try:
            chart_specs = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试修复常见错误（对象/数组末尾多余逗号）后重试
            fixed = _re.sub(r',\s*([\]}])', r'\1', raw)
            try:
                chart_specs = json.loads(fixed)
            except json.JSONDecodeError:
                print(f"[图表] chartjson 解析失败，跳过该块: {raw[:120]}")
                continue

        if not isinstance(chart_specs, list):
            continue

        for spec in chart_specs:
            try:
                _render_chart_spec(spec, charts, fp_m, fp_sm)
            except Exception as e:
                print(f"[图表] 生成失败: {e}")
                continue

    return charts


def _render_chart_spec(spec, charts, fp_m, fp_sm):
    """按单个 chartjson spec 生成图表并追加到 charts。"""
    try:
        title = spec.get("title", "图表")
        ctype = spec.get("type", "bar")
        data = spec.get("data", {})
        if not data or not isinstance(data, dict):
            return

        # 模型可能输出字符串数值（"149800"、"24.46%"、"-21,400"），统一强转为 float；
        # 无法转换的项（如"无数据"）直接跳过，全部无法转换则放弃该图
        labels = []
        values = []
        for k, v in data.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                num = float(v)
            else:
                s = str(v).strip().replace("%", "").replace(",", "")
                try:
                    num = float(s)
                except (TypeError, ValueError):
                    continue
            # 过滤 NaN / inf / 非有限值
            if num != num or abs(num) == float("inf"):
                continue
            labels.append(k)
            values.append(num)
        if not values:
            return

        fig, ax = plt.subplots(figsize=(9, 4.5))

        if ctype == "pie":
            # 饼图不支持负值：过滤非正值（如利息收入 -24、投资净流量 -21400）
            keep = [(l, v) for l, v in zip(labels, values) if v > 0]
            if not keep:
                print(f"[图表] 跳过 {title}: 饼图无正值数据")
                return
            labels = [k for k, _ in keep]
            values = [v for _, v in keep]
            colors = plt.cm.Set3(range(len(labels)))
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90,
                textprops={'fontproperties': fp_sm}
            )
            ax.set_title(title, fontproperties=fp_m, fontweight="bold")
        elif ctype == "bar_h":
            # 水平柱状图
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))
            bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontproperties=fp_sm)
            ax.invert_yaxis()
            ax.set_title(title, fontproperties=fp_m, fontweight="bold")
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                        str(val), ha="left", va="center", fontproperties=fp_sm)
        else:
            # 默认竖柱状图
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))
            bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="white")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=25, ha="right", fontproperties=fp_sm)
            ax.set_title(title, fontproperties=fp_m, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                        str(val), ha="center", va="bottom", fontproperties=fp_sm, fontsize=9)

        fig.tight_layout()
        charts.append((title, _fig_to_base64(fig)))
    except Exception as e:
        print(f"[图表] 生成失败 ({title}): {e}")


def _generate_charts(df, prefix=""):
    """智能生成图表：根据数据质量和模式自适应选择图表类型。

    原则：
    - 数据不够的列不画，宁缺毋滥
    - 有对比关系的优先画（如预算vs实际、期末vs年初）
    - 变化大的列比变化小的列更值得画

    Returns:
        [(title, base64_img), ...] 列表
    """
    fp = _get_font_prop()
    fp_s = _get_font_prop(size=11)
    fp_m = _get_font_prop(size=14)
    fp_sm = _get_font_prop(size=9)
    fp_tiny = _get_font_prop(size=8)

    charts = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    label_prefix = f"{prefix} - " if prefix else ""

    # ── 评分：哪些列值得画 ──
    def _col_score(col_name):
        """给数值列打分：非空率、方差、数据范围综合考虑。分数越高越值得画。"""
        s = df[col_name].dropna()
        if len(s) < 3:
            return 0
        n_unique = s.nunique()
        if n_unique <= 1:
            return 0  # 单一值，画了没意义
        cv = s.std() / (abs(s.mean()) + 1)  # 变异系数
        coverage = len(s) / len(df)
        # 高方差 + 高覆盖率 = 高分
        return min(cv, 3) * coverage * 10

    # 筛选有效数值列，过滤掉全是零/null/单一值的
    valid_scores = {c: _col_score(c) for c in numeric_cols}
    good_cols = sorted([c for c, sc in valid_scores.items() if sc >= 1.0],
                       key=lambda c: valid_scores[c], reverse=True)

    # ── 尝试检测"对比对"（如期末vs年初、预算vs实际） ──
    pair_cols = []
    for i, c1 in enumerate(numeric_cols):
        n1 = str(c1).lower()
        for c2 in numeric_cols[i + 1:]:
            n2 = str(c2).lower()
            # 检测常见对比模式
            if (("期末" in n1 and "年初" in n2) or ("年初" in n1 and "期末" in n2) or
                ("预算" in n1 and "实际" in n2) or ("实际" in n1 and "预算" in n2) or
                ("本期" in n1 and "同期" in n2) or ("同期" in n1 and "本期" in n2) or
                ("计划" in n1 and "完成" in n2) or ("完成" in n1 and "计划" in n2)):
                if valid_scores.get(c1, 0) >= 0.5 and valid_scores.get(c2, 0) >= 0.5:
                    pair_cols.append((c1, c2))
    # 只取前 3 对
    pair_cols = pair_cols[:3]

    # ── 图表 A: 对比柱状图（如果有对比对）──
    if pair_cols:
        fig, axes = plt.subplots(1, len(pair_cols), figsize=(5 * len(pair_cols), 4.5))
        if len(pair_cols) == 1:
            axes = [axes]

        for ax, (c1, c2) in zip(axes, pair_cols):
            # 取有数据的行
            pair_data = df[[c1, c2]].dropna()
            if len(pair_data) == 0:
                ax.text(0.5, 0.5, f"{c1} vs {c2}\n(\u6570\u636e\u4e0d\u8db3)", ha="center", va="center",
                        transform=ax.transAxes, fontproperties=fp, fontsize=10, color="#94a3b8")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            # 样本太多时取前 15 条
            sample = pair_data.head(15)
            x = np.arange(len(sample))
            w = 0.35

            ax.bar(x - w / 2, sample[c1], w, label=c1, color="#3b82f6", alpha=0.85)
            ax.bar(x + w / 2, sample[c2], w, label=c2, color="#f59e0b", alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in sample.index], rotation=45, ha="right",
                               fontproperties=fp_sm)
            ax.legend(loc="best", prop=fp_sm)
            ax.grid(axis="y", alpha=0.3)
            for label in ax.get_yticklabels():
                label.set_fontproperties(fp_sm)

        fig.suptitle(f"{label_prefix}\u5173\u952e\u6307\u6807\u5bf9\u6bd4", fontproperties=fp_m, fontweight="bold")
        fig.tight_layout()
        charts.append((f"{label_prefix}\u5173\u952e\u6307\u6807\u5bf9\u6bd4", _fig_to_base64(fig)))

    # ── 图表 B: 高分列直方图（只画真正有价值的前几个） ──
    if good_cols:
        n_show = min(len(good_cols), 4)
        hist_cols = good_cols[:n_show]
        rows = (n_show + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(11, 3 * rows))
        axes = axes.flatten() if n_show > 1 else [axes]

        for i, col in enumerate(hist_cols):
            ax = axes[i]
            col_data = df[col].dropna()
            # 自动推断合适的 bins 数
            n_unique = col_data.nunique()
            bins = max(5, min(25, int(np.sqrt(len(col_data)))))
            ax.hist(col_data, bins=bins, color="#3b82f6", edgecolor="white", alpha=0.85)
            ax.set_title(col, fontproperties=fp_s, fontweight="bold")
            ax.set_xlabel("\u503c", fontproperties=fp)
            ax.set_ylabel("\u9891\u6b21", fontproperties=fp)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(fp_sm)

        for j in range(n_show, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{label_prefix}\u6570\u503c\u5217\u5206\u5e03", fontproperties=fp_m, fontweight="bold", y=1.01)
        fig.tight_layout()
        charts.append((f"{label_prefix}\u6570\u503c\u5217\u5206\u5e03\u76f4\u65b9\u56fe", _fig_to_base64(fig)))

    # ── 图表 C: 排名柱状图（找出变化最大的项目）──
    if len(good_cols) >= 2:
        # 取每个 good_col 的均值，按大小排序
        means = {c: df[c].dropna().mean() for c in good_cols}
        # 只画绝对值最大的前 8 个
        top_items = sorted(means.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        if len(top_items) >= 2:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            names = [t[0] for t in top_items]
            values = [t[1] for t in top_items]

            colors = [("#ef4444" if v < 0 else "#3b82f6") for v in values]
            bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white")

            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontproperties=fp_sm)
            ax.set_xlabel("\u5747\u503c", fontproperties=fp)
            ax.set_title(f"{label_prefix}\u4e3b\u8981\u6307\u6807\u6392\u540d",
                         fontproperties=fp_m, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            ax.invert_yaxis()
            for label in ax.get_xticklabels():
                label.set_fontproperties(fp_sm)

            fig.tight_layout()
            charts.append((f"{label_prefix}\u4e3b\u8981\u6307\u6807\u6392\u540d", _fig_to_base64(fig)))

    # ── 图表 D: 类别列频次（只画 3-20 类的，太多太少都没意义） ──
    if categorical_cols:
        for cat_col in categorical_cols[:2]:  # 最多画 2 个类别列
            value_counts = df[cat_col].value_counts()
            if 3 <= len(value_counts) <= 20:
                fig, ax = plt.subplots(figsize=(11, 4.5))
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(value_counts)))
                bars = ax.barh(range(len(value_counts)), value_counts.values,
                               color=colors, edgecolor="white")

                ax.set_yticks(range(len(value_counts)))
                ax.set_yticklabels(value_counts.index, fontproperties=fp_sm)
                ax.set_title(f"{label_prefix}\u300c{cat_col}\u300d\u5206\u5e03",
                             fontproperties=fp_m, fontweight="bold")
                ax.set_xlabel("\u9891\u6b21", fontproperties=fp)
                ax.grid(axis="x", alpha=0.3)
                ax.invert_yaxis()
                for label in ax.get_xticklabels():
                    label.set_fontproperties(fp_sm)
                fig.tight_layout()
                charts.append((f"{label_prefix}\u300c{cat_col}\u300d\u5206\u5e03", _fig_to_base64(fig)))
                break  # 找到第一个合适的就停

    return charts


def _call_deepseek_api(prompt, max_tokens=16384, extra_body=None):
    """通过 DeepSeek API 进行推理（支持 GGUF llama-server 和 DeepSeek API）。"""
    if "deepseek.com" in DEEPSEEK_API_URL and not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek 官方 API 需要 DEEPSEEK_API_KEY")

    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一位资深的企业经营数据分析师。请仔细分析数据并给出专业、详尽的报告。使用 Markdown 格式输出。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": _TEMPERATURE,
        "max_tokens": max_tokens,
        "top_p": 0.9,
    }
    if extra_body:
        body.update(extra_body)
    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=payload,
        headers=dict(
            {"Content-Type": "application/json"},
            **({"Authorization": f"Bearer {DEEPSEEK_API_KEY}"} if DEEPSEEK_API_KEY else {}),
        ),
        method="POST",
    )

    print("[api] 发送推理请求...")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"DeepSeek API 返回错误 ({e.code}): {body}")
    except Exception as e:
        raise RuntimeError(f"DeepSeek API 请求失败: {str(e)}")

    if "choices" not in result or len(result["choices"]) == 0:
        raise RuntimeError(f"DeepSeek API 返回异常: {json.dumps(result, ensure_ascii=False)[:500]}")

    output = result["choices"][0]["message"]["content"]
    if not output or not output.strip():
        output = "（API 未生成有效回复，请重试）"

    usage = result.get("usage", {})
    print(f"[api] 推理完成 — 输入 {usage.get('prompt_tokens', '?')} tokens, "
          f"输出 {usage.get('completion_tokens', '?')} tokens")

    return output


def _call_deepseek_api_stream(prompt, max_tokens=16384, extra_body=None):
    """流式调用 API（支持 GGUF llama-server 和 DeepSeek API）。

    思考模式下 yield ("think", reasoning_content) 和 ("text", content);
    普通模式下只 yield ("text", content)。
    """
    if "deepseek.com" in DEEPSEEK_API_URL and not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek 官方 API 需要 DEEPSEEK_API_KEY")

    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一位资深的企业经营数据分析师。请仔细分析数据并给出专业、详尽的报告。使用 Markdown 格式输出。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": _TEMPERATURE,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "stream": True,
    }
    if extra_body:
        body.update(extra_body)
    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=payload,
        headers=dict(
            {"Content-Type": "application/json"},
            **({"Authorization": f"Bearer {DEEPSEEK_API_KEY}"} if DEEPSEEK_API_KEY else {}),
        ),
        method="POST",
    )

    print(f"[api] 发送流式推理请求 (模型: {DEEPSEEK_MODEL})...")
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"DeepSeek API 返回错误 ({e.code}): {body}")

    chunk_count = 0
    think_count = 0
    try:
        for line_bytes in resp:
            try:
                line = line_bytes.decode("utf-8").strip()
            except Exception:
                continue

            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})

                    # 思考内容 (deepseek-reasoner)
                    reasoning = delta.get("reasoning_content", "")
                    if reasoning:
                        think_count += 1
                        yield ("think", reasoning)

                    # 正式输出
                    content = delta.get("content", "")
                    if content:
                        chunk_count += 1
                        yield ("text", content)

                except json.JSONDecodeError:
                    continue
    finally:
        # 生成器被 close()（客户端断开取消推理）时也立即断开底层连接，让 llama-server 停止生成
        resp.close()
    print(f"[api] 流式推理完成 — {think_count} 个思考块, {chunk_count} 个文本块")


def _run_inference(prompt, max_tokens=16384, stream=False):
    """统一推理入口，覆盖两种后端（DeepSeek API / GGUF llama-server）。

    stream=False: 返回分析文本字符串。
    stream=True: 返回生成器，产出 ("think"|"text", chunk) 元组。
    """
    if DEBUG_MODE:
        if stream:
            return _call_deepseek_api_stream(prompt, max_tokens=max_tokens)
        return _call_deepseek_api(prompt, max_tokens=max_tokens)

    # GGUF：外部 llama-server 已就绪，走 OpenAI 兼容 API 路径
    get_model_and_tokenizer()
    if stream:
        return _call_deepseek_api_stream(prompt, max_tokens=65536, extra_body=_LOCAL_SERVER_BODY)
    return _call_deepseek_api(prompt, max_tokens=65536, extra_body=_LOCAL_SERVER_BODY)


def _distributed_nodes():
    """解析 DEEPANALYZE_NODES，返回 [{"name", "url"}]；未设置返回 []（单节点模式）。"""
    nodes = []
    for i, entry in enumerate(_NODE_LIST.split(",")):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            name, url = entry.split("=", 1)
            nodes.append({"name": name.strip(), "url": url.strip().rstrip("/")})
        else:
            nodes.append({"name": f"node{i+1}", "url": entry.rstrip("/")})
    return nodes


def _work_nodes():
    """参与工作表分发的节点：主节点自身（url=None，本地推理）+ 加速节点列表。"""
    return [{"name": "主节点", "url": None}] + _distributed_nodes()


def _submit_node_task(ex, node, prompt):
    """按节点类型提交任务：主节点本地推理，加速节点走 HTTP（非流式路径用）。"""
    if node["url"] is None:
        return ex.submit(_run_inference, prompt, _SHEET_MAX_TOKENS)
    return ex.submit(_call_node_sheet, node["url"], prompt)


def _build_sheet_prompt(label, summary, hard, question):
    """构建单个工作表的分项分析 prompt。"""
    parts = [
        f"你是一位资深的企业经营数据分析师。下面是经营数据文件中的工作表「{label}」的完整数据摘要，"
        "请对用户提出的问题进行深入、全面的分析。",
        "",
        "⚠️ 以下【硬数字】为从 Excel 中精确提取的数值，绝对正确，禁止修改，必须原样引用。"
        "每个数字都标注了期别标签（如 2026上半年=7400），使用前请核对期别。",
        "",
    ]
    if hard:
        parts += [
            "=== ❌硬数字-禁止编造-必须引用 ===",
            "⚠️ 以下数值从Excel精确提取，绝对正确，必须原样引用，禁止修改。",
            "⚠️ 每个标签标注了期别（如2026H1=当前报告期），请使用对应[当前报告期]的值。",
            "⚠️ 硬数字行的格式为「指标名(期别=数值)」，如「应收票据(期末=13600)」表示期末应收票据为13600万元。",
            "⚠️ 负号只可能出现在数值本身（如-100表示负值），括号内是期别标签，不是负数；"
            "同一指标通常有'期初'和'期末'两行，必须成对核对后再解读增减方向。",
            "⚠️ 引用硬数字必须保留原始指标名（如净利率），禁止改写成硬数字中未出现的指标名（如毛利率）。",
            "⚠️ 只有当期数据（无历史期标签）的指标严禁编造对比期或变动率，只能陈述当期数值。",
            "⚠️ 报告中出现的任何数值必须能在硬数字中找到完全匹配的行（指标名+期别一致）；"
            "找不到出处的数值禁止填入，一律留空或写'无数据'；禁止挪用其他指标行的数值充当历史期。",
            "⚠️ 对比表的'期初/年初'值必须使用同一指标紧随'期末'的那期（标签为'年初'或'2025H2'），"
            "禁止混用'2025H1'等更早期别的数值。",
            "⚠️ '基准/冲刺'是两档目标值，不是时间序列，禁止当作期初/期末计算变动率；"
            "预算执行应对比实际值与预算目标（如实际销售额 vs 预算基准）。",
            hard,
            "=== 结束 ===",
            "",
        ]
    # 经营预算表专项约束：防止模型编造资产负债表/资金状况科目
    if "预算" in label:
        parts += [
            "⚠️ 本工作表是【经营预算】表，只包含预算科目（预算销售额、预算净利润、工资、年终奖金、股权激励、"
            "社会保险费、住房公积金、福利费、工会经费、研发费用-股权激励、剔除股权激励前等）。",
            "⚠️ 禁止出现资产负债表/资金状况类科目（如应收票据、应付账款、经营现金流净额、固定资产、流动负债等），"
            "这些数据在其他工作表，本表没有；分析时只使用上方硬数字中实际存在的预算科目。",
            "",
        ]
    parts.append(f"数据规模: {summary['num_rows']} 行 × {summary['num_cols']} 列")
    if summary.get("sheet_overview"):
        parts.append(summary["sheet_overview"])
    parts.append("")
    if summary.get("per_sheet_summary"):
        parts += ["--- 预计算汇总 ---", summary["per_sheet_summary"], ""]
    # 有硬数字时不传行级样本，防模型读 raw data 编造
    if not hard:
        parts += [
            "--- 头部数据 ---", summary["head_str"], "",
            "--- 尾部数据 ---", summary["tail_str"], "",
            "--- 文本列值分布 ---", summary["dimensions_str"], "",
            "--- 缺失值 ---", summary["missing_str"], "",
        ]
    parts += [
        f"=== 用户问题 ===\n{question}",
        "",
        "=== 分析要求（强制执行） ===",
        "1. 以本工作表为分析单元，逐板块拆分并独立深挖，每个板块一个 ## 标题，不允许合并或一笔带过。"
        "分析思路必须贴合上方'用户问题'：问题明确指向的板块/指标深挖并优先呈现，与问题无关的指标只放数据表不展开讨论，"
        "小标题、结论与建议围绕问题展开，禁止输出与问题无关的模板化内容。",
        "2. 每个板块必须包含：数据概览表、关键指标对比表、变化分析（⚠️预警标记只用于变动幅度绝对值 >30% 的指标，"
        "不足 30% 时严禁出现'预警'字样；⚠️ 符号本身也只能出现在变动绝对值 >30% 的指标上，"
        "费用、成本、收入、利润的任何指标变动不足 30% 时一律不得加 ⚠️，写完必须自查全文，"
        "发现 <30% 的指标带 ⚠️ 立即删除）、业务解读、策略建议、风险提示。"
        "骨架固定是质量底线，但各小节的内容侧重必须按用户问题调整：问题涉及的业务写深写透，"
        "问题未涉及的维度简写；如果问题指向单一指标（如只看应收账款），该指标的分析必须放在板块最前并给出明确结论。",
        "3. 每条发现必须引用至少 2 个具体数值（硬数字优先），禁止“某些指标”“部分数据”等模糊表述。",
        "4. 数据只有一期时只列一期，严禁编造多期对比；无历史期数据的指标不得计算变动率，只能陈述当期数值；"
        "每个数值必须能在硬数字中找到指标名+期别完全匹配的出处，否则留空写'无数据'，禁止挪用其他指标行的数值充当历史期；"
        "禁止将同一组数字复制粘贴到多个期别列。",
        "5. 硬数字必须原样引用，禁止修改或换算。",
        "6. 输出使用 Markdown，标题以 ## 开头，重点数字加粗。",
        "7. 数据概览表必须列出核心指标（至少3行、至多8行，每行含指标名+数值），不得省略为空。",
        "8. 数据优先、文字精简：表格是主体、文字是辅助。数据概览表必须列出本板块全部核心指标"
        "（不设行数上限，一行一个指标，含跨期列），关键指标对比表列出全部有跨期数据的指标；"
        "禁止为了控制篇幅删减数据行。变化分析/业务解读/策略建议/风险提示每节 1-2 段、每段 ≤100 字，"
        "且每段必须引用至少 1 个具体数值（禁止空泛论述，如'整体表现良好'必须写成'销售额 103700 万元、"
        "环比 {{calc: 2500/101200*100}}%'）；与用户问题无关的板块只给数据表不写正文；"
        "本工作表总字数（含表格）控制在 2000-2800 字，超出的部分优先删文字、不删数据。",
        "9. 变化分析/业务解读/策略建议/风险提示每节各写 1-3 段，禁止每个指标单独一段的逐项式输出；"
        "六个部分必须各自独立成节（'变化分析与业务解读'这类合并写法不合格，需拆开）。",
        "10. 只有当期数据的表，历史期列必须写'无数据'，禁止把当期值复制到历史期列、禁止从其他指标行取值。",
        "11. 变化分析的对比期必须与上方对比表一致（以'2025H2/年初'列为基准），禁止改用'2025H1'等其他列计算变动率。",
        "12. 历史期为'无数据'的指标只陈述当期数值，禁止写出任何增幅/降幅。",
        "13. '预算金额/合同金额/已付金额'是项目执行阶段而非时间序列，禁止计算降幅，应解读为执行进度。",
        "14. 图表数据必须与正文一致：只有当期数据的指标只列当期数值，禁止用0、重复值或其他指标行数值填充缺失期别。",
        "15. 每个板块只输出 1 个【图表数据】区块，禁止重复或近似重复的图表。",
        "16. 本任务是单轮一次性任务，必须一次输出完整分析；禁止向用户提问、禁止'是否继续'式交互、禁止中途停止等待确认。",
        "17. 禁止使用 LaTeX 公式、$...$、$$...$$、\\frac、\\approx 等数学符号、引用块（> 符号）等格式，一律使用纯 Markdown 表格与文本。",
        "18. 禁止用省略号（...）、'依此类推'、'其余板块'等占位内容敷衍；本工作表分析必须完整写出六个小节。",
        "19. 直接输出分析正文，禁止任何前言、思考过程、计划或解释性开头（如'好的''首先''我需要''让我们'等），第一个字必须是正文。",
        "21. 硬数字中真实存在的期别数值（如2025H2、年初）必须被引用使用，禁止把存在的数据标为'无数据'；"
        "资产负债表的'年初'标签就是历史期对比值（等同于其他表的2025H2），必须使用，禁止标为无数据；"
        "同一节内数据概览表与关键指标对比表必须一致，禁止自相矛盾。",
        "22. 同一小节的多个子节必须覆盖不同内容，禁止两个子节使用相同表格、相同数值或相同结论；内容雷同的子节必须合并为一个；每个小节最多 4 个子节，超过即失败。",
        "23. 所有合计、占比、比率必须用报告中引用的数值现场重新计算并注明算式（如'8500+3400+1100+300+6800=20100'、'3800/8500=44.7%'、占比必须注明分母如'8500/19500=43.6%'），禁止给出无法由引用数字推导的结果；"
        "多位加法必须逐项相加并当场验算至少两遍（写完重新加，发现不一致必须修正），每个合计结果必须能由列出的算式精确推导，禁止凭印象填合计值。",
        "24. 分析正文之后输出【图表数据】区块，每个分析模块至少一张图表，数据必须来自硬数字，禁止编造，格式：",
        "```chartjson",
        '[{"title": "图表标题", "type": "bar/pie/line/bar_h", "data": {"指标1": 数值, "指标2": 数值}}]',
        "```",
        "25. 硬数字中含'同比/环比'开头的行（如「同比(1个月内=-0.023)」）是工作表自带的变动率（小数），"
        "必须直接引用并转成百分比（-2.3%），禁止自行重新计算或改写标签；同比=与上年同期（2025H1）比较，"
        "环比=与上一期（2025H2）比较，两者不得互换。",
        "26. 比率/占比/费率/收益率类硬数字是小数（如 费用率=0.083 表示 8.3%，税负率=0.120 表示 12%，"
        "收益率=0.018 表示 1.8%），引用时保持小数原样（0.083）或乘 100 写成百分比（8.3%），"
        "严禁写成'0.083%'这类不乘 100 的格式；自行计算的比率同样必须乘 100（如 22100/184800=12%，不是 0.12%）。",
        "27. 【图表数据】中 data 的键名必须与硬数字指标名一致且引号闭合（如\"碳乙**: 6500\"必须写成"
        "\"碳乙**\": 6500），禁止输出缺闭合引号的键名，否则整块图表解析失败被丢弃。",
        "28. 【图表数据】中 data 的每个值必须是单个数字，禁止数组或对象（如\"六\": [5800, 6200] 会导致整块图表解析失败被丢弃）；"
        "同一指标的多期对比请用多个键（如\"碳乙(2026H1)\": 6500, \"碳乙(2025H2)\": 6000）。",
        "29. 禁止出现硬数字中不存在的科目名及数值。编造科目黑名单（一律禁止出现）：人工费用、办公费、招待费、差旅费、"
        "认证费、检验费、检测费、运输费用；历史期数值（如2025H2=700）只有硬数字中存在才能引用，禁止凭空补历史值；"
        "禁止把其他工作表的数值（如职工教育经费700、区域运费200）挪作本表科目或本表历史期。找不到出处的科目直接不写。",
        "30. 所有需要计算的地方（加总、占比、比率、变动率）一律输出 {{calc: 表达式}} 标记，服务端自动计算并替换为结果——"
        "禁止自己心算数字。表达式只写数字和 + - * / 括号小数点；百分数用 *100（如 {{calc: 149800/420000*100}}% 显示为 43.6%）；"
        "占比必须注明分母（如 {{calc: 8500/19500*100}}%）；标记必须完整闭合（{{calc: ...}}），表达式内禁止出现文字。",
    ]
    return "\n".join(parts)


def _truncate_text(text, limit=3000):
    """按行截断长文本，避免总览 prompt 超长。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0] + "\n（该分项分析较长，已截断）"


def _build_overview_prompt(question, sections):
    """构建主节点总览部分的 prompt。sections: [(label, node_name, text)]"""
    parts = [
        "你是一位资深的企业经营数据分析师。以下是对同一批经营数据按工作表拆分、由多个节点分别完成的分项分析结果。",
        "请基于这些分项分析结果，生成整份报告的【总览与综合结论】部分。",
        "",
        f"=== 用户问题 ===\n{question}",
        "",
        "=== 各分项分析结果 ===",
    ]
    for label, node, text in sections:
        parts.append(f"### {label}（节点：{node}）\n{_truncate_text(text)}")
    parts += [
        "",
        "=== 总览部分要求（强制执行） ===",
        "1. 开篇必须先直接回答用户问题（结论先行，一段话讲清问题的答案），然后跨板块综合分析："
        "各工作表之间的联动关系、整体经营画像、资源错配检测、前瞻预测；分析侧重必须跟随用户问题的指向，"
        "问题涉及的板块重点展开，与问题无关的内容从简。",
        "2. 总评与行动纲领：经营健康度评分（优秀/良好/关注/预警）、TOP 3 亮点、TOP 3 风险、5 项最紧迫工作（按优先级排序，每项标明负责板块）。",
        "3. 需求对照表：| 用户要求 | 对应报告章节 | 数据支撑情况 |，逐条对照用户问题中的每项要求，不允许遗漏。",
        "4. 引用具体数值时必须与分项分析一致，禁止编造；数据无法支撑的维度要明确说明数据局限。",
        "5. 输出使用 Markdown，标题以 ## 开头，重点数字加粗。",
        "6. 末尾输出【图表数据】区块，展示总览关键对比，数据必须来自上述分项分析，禁止编造，格式：",
        "```chartjson",
        '[{"title": "图表标题", "type": "bar/pie/line/bar_h", "data": {"指标1": 数值, "指标2": 数值}}]',
        "```",
        "7. 总览中的占比/进度/增幅必须用引用的数值现场重算并注明算式（如 103700/119700=86.6%），"
        "禁止凭感觉写'约一半'等无法由引用数字推导的表述；比率类数值转百分比必须乘 100（0.120=12%）。",
        "8. 总览中的所有计算（占比/进度/增幅/比率）同样使用 {{calc: 表达式}} 标记，服务端自动计算替换，"
        "禁止自己心算（如 {{calc: 103700/119700*100}}%）。",
        "9. 总览控制在 2000-2800 字以内（含图表区块），数据优先：关键对比用表格呈现，"
        "每段论述必须引用具体数值，禁止逐表复述分项分析内容，只做跨板块整合与结论。",
    ]
    return "\n".join(parts)


def _call_node_sheet(url, prompt, max_tokens=_SHEET_MAX_TOKENS, timeout=_NODE_TIMEOUT):
    """调用加速节点的 /analyze/sheet 端点，返回分析文本。"""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(
        url + "/analyze/sheet",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        eb = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"节点 {url} 返回错误 ({e.code}): {eb[:200]}")
    except Exception as e:
        raise RuntimeError(f"节点 {url} 请求失败: {str(e)}")
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    result = data.get("result", "")
    # 安全阀：节点失控输出（如逐行枚举）时截断，防止挤占主节点上下文与图表提取
    if len(result) > 12000:
        print(f"[分布式] 节点 {url} 输出过长 ({len(result)} 字符)，已截断")
        result = result[:12000] + "\n（该分项输出过长，已截断）"
    return result


# 工作表别名：用户问题中点名这些词时，只分析对应表（未点名则全部分析）
_SHEET_ALIASES = {
    "资产负债、杜邦分析": ["资产负债", "杜邦", "资产", "负债", "权益", "资本结构"],
    "销售及毛利": ["销售及毛利", "销售毛利", "毛利", "销售"],
    "应收账龄": ["应收账龄", "应收", "账龄", "应收款"],
    "采购": ["采购", "物料"],
    "存货": ["存货", "库存"],
    "生产成本": ["生产成本", "产量", "单位成本"],
    "期间费用": ["期间费用", "销售费用", "管理费用", "研发费用", "财务费用", "期间费用率"],
    "人力成本": ["人力成本", "人力", "人工成本", "人工", "薪酬", "工资"],
    "经营预算": ["经营预算", "预算", "目标"],
    "税收": ["税收", "税"],
    "资金状况": ["资金状况", "资金", "现金流", "现金流量"],
    "物流运输": ["物流运输", "物流", "运输"],
    "固定资产投资": ["固定资产投资", "固定资产", "固投"],
}


def _filter_tasks_by_question(tasks, question):
    """用户问题明确点名工作表时只分析对应表；未点名时全部分析（安全默认）。

    匹配方式：工作表名或其别名出现在问题中即命中；"成本/费用/资产"等别名
    可能误命中，但只有问题确实点名了表才启用过滤，未命中任何表则保持全部分析。
    """
    matched = set()
    for t in tasks:
        label = t.get("label", "")
        sname = label.split("·", 1)[1] if "·" in label else label
        aliases = _SHEET_ALIASES.get(sname, [sname])
        if any(a and a in question for a in aliases):
            matched.add(sname)
    if not matched:
        return tasks
    kept = [t for t in tasks
            if (t["label"].split("·", 1)[1] if "·" in t["label"] else t["label"]) in matched]
    if kept:
        print(f"[工作表过滤] 问题点名了 {len(kept)} 张表: {sorted(matched)}，只分析这些表")
    return kept or tasks


def _prepare_distributed_input(valid_files, question):
    """为分布式模式准备任务列表：按工作表拆分 summary 与硬数字。

    Returns:
        {filenames, total_rows, total_cols, dataframes, tasks: [{label, prompt}]}
    """
    import traceback
    try:
        return _prepare_distributed_input_impl(valid_files, question)
    except Exception:
        print(f"[_prepare_distributed ERROR] {traceback.format_exc()}")
        raise


def _prepare_distributed_input_impl(valid_files, question):
    from werkzeug.datastructures import FileStorage as _FS

    total_rows = 0
    total_cols = 0
    filenames = []
    dataframes = []

    # 先保存文件原始数据（FileStorage 流只能读一次）
    file_buffers = []
    fresh_files = []
    for f in valid_files:
        buf = io.BytesIO()
        f.save(buf)
        data = buf.getvalue()
        file_buffers.append((f.filename, data))
        fresh_files.append(_FS(stream=io.BytesIO(data), filename=f.filename))

    tasks = []
    for f in fresh_files:
        filename, suffix, df = read_file(f)
        filenames.append(filename)
        dataframes.append((filename, df))
        total_rows += len(df)
        total_cols += len(df.columns)

        # 每个工作表自己的硬数字
        hard_by_sheet = {}
        try:
            file_bytes = dict(file_buffers)[filename]
            hard_by_sheet = extract_hard_numbers_by_sheet(io.BytesIO(file_bytes))
        except Exception as e:
            print(f"[硬数字] {filename} 提取失败: {e}")

        if "_sheet" in df.columns:
            for sname in df["_sheet"].unique():
                sheet_df = df[df["_sheet"] == sname]
                summary = df_summary(filename, sheet_df)
                hard = hard_by_sheet.get(sname, "")
                label = f"{filename}·{sname}"
                tasks.append({"label": label, "prompt": _build_sheet_prompt(label, summary, hard, question)})
        else:
            # CSV/PDF 等无 sheet 划分的文件，整体作为一个任务
            summary = df_summary(filename, df)
            label = filename
            tasks.append({"label": label, "prompt": _build_sheet_prompt(label, summary, "", question)})

    if not tasks:
        raise RuntimeError("未检测到可分发的工作表数据")

    # 问题点名工作表时只分析被点名的表
    tasks = _filter_tasks_by_question(tasks, question)
    if not tasks:
        raise RuntimeError("未检测到可分发的工作表数据")

    return {
        "filenames": filenames,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "dataframes": dataframes,
        "tasks": tasks,
    }


def _safe_eval_arith(expr):
    """安全计算四则运算表达式（仅数字、+ - * / %、括号、小数点）。

    用 ast 校验节点类型后再求值，杜绝 eval 任意代码（防 prompt 注入）。
    % 视为除以100（如 8500/19500% = 0.0044...）。
    """
    import ast as _ast
    expr = str(expr).replace("%", "/100")
    tree = _ast.parse(expr, mode="eval")
    allowed = (_ast.Expression, _ast.BinOp, _ast.UnaryOp, _ast.Constant,
               _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.USub, _ast.UAdd)
    for node in _ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"不支持的表达式: {expr}")
        if isinstance(node, _ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"非数值: {expr}")
    return eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}})


def _normalize_calc_forms(text):
    """把模型误用的其他 calc 标记格式归一化为 {{calc: 表达式}}：
    {% calc %}表达式{% endcalc %} 与 {% calc: 表达式 %} 两种。
    """
    text = re.sub(r"\{%\s*calc\s*%\}\s*([\s\S]*?)\s*\{%\s*endcalc\s*%\}", lambda m: "{{calc: " + m.group(1).strip() + "}}", text)
    text = re.sub(r"\{%\s*calc\s*[:：]\s*([^%}]+?)\s*%\}", lambda m: "{{calc: " + m.group(1).strip() + "}}", text)
    return text


def _eval_calc_markers(text):
    """把文本中所有 {{calc: 表达式}} 标记替换为计算结果（先归一化 {% calc %} 等变体）。"""
    def _repl(m):
        try:
            val = _safe_eval_arith(m.group(1))
        except Exception:
            return m.group(0)  # 计算失败保留原文（正常不应发生）
        if isinstance(val, float):
            s = f"{val:.2f}".rstrip("0").rstrip(".")
            return "0" if s == "-0" else s
        return str(int(val))
    return re.sub(r"\{\{calc:\s*([^}]+?)\s*\}\}", _repl, _normalize_calc_forms(text))


class _CalcStreamer:
    """流式替换 {{calc: ...}} 标记，支持标记跨文本块拆分。

    feed(text) 返回可输出的部分（不完整标记尾部暂存）；流结束时 flush() 返回剩余尾部。
    """

    def __init__(self):
        self._buf = ""

    def feed(self, text):
        out = self._buf + text
        self._buf = ""
        # 归一化完整闭合的 {% calc %}...{% endcalc %} 与 {% calc: ... %} 变体（不完整的留给尾部暂存）
        out = _normalize_calc_forms(out)
        while True:
            m = re.search(r"\{\{calc:\s*([^}]+?)\s*\}\}", out)
            if not m:
                break
            out = out[:m.start()] + _eval_calc_markers(m.group(0)) + out[m.end():]
        # 尾部可能是未闭合标记的开头（含被截断的前缀 "{{ca" 或完整开头 "{{calc: 8500"、"{%"）：暂存等待下个块补全
        # 注意要匹配 "{{" 整体（rfind("{") 会命中第二个 { 导致 tail 从 "{calc..." 开始）
        idx = out.rfind("{{")
        idx2 = out.rfind("{%")
        if idx2 > idx:
            idx = idx2
        if idx == -1:
            idx = out.rfind("{")
        if idx != -1:
            tail = out[idx:]
            if tail.startswith("{{calc:") or "{{calc:".startswith(tail) or tail.startswith("{%") or "{% calc".startswith(tail):
                self._buf = tail
                out = out[:idx]
        return out

    def flush(self):
        out = self._buf
        self._buf = ""
        return _eval_calc_markers(out)


def _stream_section_worker(q, node, t, stop_event):
    """流式工作表任务线程：把文本块逐个放入队列。

    队列元素: ("chunk", t, node, evt_type, chunk) / ("error", t, node, msg) / ("done", t, node, None)
    首个文本块自带小节标题头（### 工作表「x」分析（节点：y））。
    stop_event：客户端断开/分析中止时置位，worker 在下个文本块边界提前终止，释放模型。
    """
    first = True
    try:
        if node["url"] is None:
            # 主节点本地任务：直接流式推理；中止时关闭推理生成器，释放底层连接
            gen = _run_inference(t["prompt"], max_tokens=_SHEET_MAX_TOKENS, stream=True)
            try:
                for evt, chunk in gen:
                    if stop_event.is_set():
                        break
                    if evt == "think":
                        q.put(("chunk", t, node, ("think", chunk)))
                    else:
                        if first:
                            chunk = f"\n\n### 工作表「{t['label']}」分析（节点：{node['name']}）\n\n" + chunk
                            first = False
                        q.put(("chunk", t, node, ("text", chunk)))
            finally:
                gen.close()
        else:
            # 加速节点任务：SSE 流式读取（旧版节点无流式端点时回退非流式）
            try:
                body = json.dumps({"prompt": t["prompt"], "max_tokens": _SHEET_MAX_TOKENS}).encode("utf-8")
                req = urllib.request.Request(
                    node["url"] + "/analyze/sheet/stream",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=_NODE_TIMEOUT) as resp:
                    for line in resp:
                        if stop_event.is_set():
                            break  # 客户端断开，停止转发该节点任务
                        line = line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if data.get("type") == "text":
                            chunk = data.get("content", "")
                            if first:
                                chunk = f"\n\n### 工作表「{t['label']}」分析（节点：{node['name']}）\n\n" + chunk
                                first = False
                            q.put(("chunk", t, node, ("text", chunk)))
                        elif data.get("type") == "think":
                            q.put(("chunk", t, node, ("think", data.get("content", ""))))
                        elif data.get("type") == "error":
                            raise RuntimeError(data.get("content", "节点流式任务错误"))
            except urllib.error.HTTPError as e:
                if e.code not in (404, 405):
                    raise
                # 加速节点是旧版本（无 /analyze/sheet/stream 端点）→ 回退非流式调用
                print(f"[分布式] 节点 {node['url']} 无流式端点，回退非流式")
                result = _call_node_sheet(node["url"], t["prompt"])
                q.put(("chunk", t, node, ("text", f"\n\n### 工作表「{t['label']}」分析（节点：{node['name']}）\n\n{result}\n")))
        if first:
            # 节点没有产出任何文本（空输出）
            q.put(("chunk", t, node, ("text", f"\n\n### 工作表「{t['label']}」分析（节点：{node['name']}）\n\n（节点未返回有效分析内容）\n")))
    except Exception as e:
        q.put(("error", t, node, str(e)))
    finally:
        q.put(("done", t, node, None))


def _distributed_analysis_events(prep, question, output_parts, stop_event=None):
    """分布式流式分析：并行生成、保序投递 → 主节点生成总览。

    各表在后台并行生成（线程池），但投递严格按任务顺序：
    每张表的文本块连续流式输出完，才开始下一张表——保证最终报告
    各表完整不交错。yield ("think"|"text"|"progress", chunk)，
    所有文本块累计进 output_parts 供图表提取。
    stop_event：客户端断开时置位，worker 提前终止，避免模型空转。
    """
    stop_event = stop_event or threading.Event()
    workers = _work_nodes()
    tasks = prep["tasks"]
    print(f"[分布式] {len(tasks)} 个工作表任务（并行生成、保序投递、动态取任务），分发给 {len(workers)} 个节点（含主节点自身）")
    sections = []
    failures = 0
    done = 0
    total = len(tasks)
    # {{calc: 表达式}} 标记 → 服务端计算结果（流式逐块替换，支持跨块拆分）
    calc_streamer = _CalcStreamer()
    # 每个任务独立的队列 + 任务顺序消费
    task_queues = {id(t): queue.Queue() for t in tasks}

    # 动态取任务（work-stealing）：每节点一个 worker 线程，从共享池取下一个任务，
    # 做完再取——快的节点自动多干活，慢的节点自然少接，负载实时均衡
    task_pool = queue.Queue()
    for i in range(len(tasks)):
        task_pool.put(i)

    def _node_worker(node):
        while not stop_event.is_set():
            try:
                idx = task_pool.get_nowait()
            except queue.Empty:
                return
            t = tasks[idx]
            _stream_section_worker(task_queues[id(t)], node, t, stop_event)

    for node in workers:
        threading.Thread(target=_node_worker, args=(node,), daemon=True).start()

    deliver_cnt = 0
    t0_consume = time.time()
    for t in tasks:
        print(f"[分布式] 开始投递任务 {done + 1}/{total}：{t['label'][:24]} @ {time.strftime('%H:%M:%S')}")
        q = task_queues[id(t)]
        node = None
        parts = []
        while True:
            # 超时保护：worker 异常死亡导致队列永久静默时，把该任务标记失败继续投递，
            # 避免主生成器无限阻塞（客户端永远等不到 done）
            try:
                kind, t2, node2, arg = q.get(timeout=900)
            except queue.Empty:
                failures += 1
                block = f"\n\n### 工作表「{t['label']}」分析（节点：未知）\n\n⚠️ 该任务超时未返回（worker 异常），已跳过\n"
                output_parts.append(block)
                yield ("text", block)
                break
            if node is None:
                node = node2
            if kind == "chunk":
                evt, chunk = arg
                if evt == "think":
                    yield ("think", chunk)
                else:
                    chunk = calc_streamer.feed(chunk)
                    output_parts.append(chunk)
                    parts.append(chunk)
                    deliver_cnt += 1
                    if deliver_cnt % 100 == 0:
                        print(f"[分布式] 已投递 {deliver_cnt} 块（{t['label'][:16]}）@ {time.strftime('%H:%M:%S')}")
                    yield ("text", chunk)
            elif kind == "error":
                failures += 1
                block = f"\n\n### 工作表「{t2['label']}」分析（节点：{node2['name']}）\n\n⚠️ 该任务分析失败：{arg}\n"
                output_parts.append(block)
                yield ("text", block)
                break
            else:  # done
                break
        if parts and node is not None:
            sections.append((t["label"], node["name"], "".join(parts).strip()))
        done += 1
        print(f"[分布式] 任务完成 {done}/{total}：{t['label'][:24]}（本任务 {len(parts)} 字符，累计投递 {deliver_cnt} 块，耗时 {time.time()-t0_consume:.0f}s）")
        yield ("progress", {"done": done, "total": total})

    # ── 主节点生成总览 ──
    intro = "\n\n---\n\n## 总览与综合结论（主节点生成）\n\n"
    output_parts.append(intro)
    yield ("text", intro)
    ov_gen = _run_inference(_build_overview_prompt(question, sections), max_tokens=16384, stream=True)
    try:
        for evt, chunk in ov_gen:
            if stop_event.is_set():
                break  # 客户端断开：提前终止，释放模型
            if evt == "think":
                yield ("think", chunk)
            else:
                chunk = calc_streamer.feed(chunk)
                output_parts.append(chunk)
                yield ("text", chunk)
    finally:
        # 无论正常结束还是客户端断开，都关闭推理生成器，避免模型空转占用
        try:
            ov_gen.close()
        except BaseException:
            pass

    # 冲刷尾部暂存（未闭合标记的兜底替换）
    tail = calc_streamer.flush()
    if tail:
        output_parts.append(tail)
        yield ("text", tail)

    if failures:
        note = f"\n\n⚠️ 共有 {failures} 个工作表任务失败，详细原因见对应小节。\n"
        output_parts.append(note)
        yield ("text", note)


def _perform_analysis_distributed(prep, question):
    """非流式分布式分析：并行分发 → 收集 → 主节点总览 → 拼装完整报告。"""
    workers = _work_nodes()
    tasks = prep["tasks"]
    print(f"[分布式] {len(tasks)} 个工作表任务，分发给 {len(workers)} 个节点（含主节点自身）")
    sections = []
    failures = 0

    with ThreadPoolExecutor(max_workers=len(workers)) as ex:
        fut_map = {}
        for i, t in enumerate(tasks):
            node = workers[i % len(workers)]
            fut = _submit_node_task(ex, node, t["prompt"])
            fut_map[fut] = (t, node)
        for fut in as_completed(fut_map):
            t, node = fut_map[fut]
            try:
                section = fut.result().strip()
            except Exception as e:
                failures += 1
                sections.append((t["label"], node["name"], f"（该任务分析失败：{str(e)}）"))
                continue
            if not section:
                section = "（节点未返回有效分析内容）"
            sections.append((t["label"], node["name"], section))

    overview = _run_inference(_build_overview_prompt(question, sections))

    body_parts = []
    for label, node, section in sections:
        body_parts.append(f"### 工作表「{label}」分析（节点：{node}）\n\n{section}\n")
    body_parts.append(f"## 总览与综合结论（主节点生成）\n\n{overview.strip()}\n")
    if failures:
        body_parts.append(f"⚠️ 共有 {failures} 个工作表任务失败，详细原因见对应小节。\n")
    full = _eval_calc_markers("\n".join(body_parts))

    all_charts = _generate_charts_from_json(full)
    if not all_charts:
        for fname, df in prep["dataframes"]:
            prefix = fname if len(prep["dataframes"]) > 1 else ""
            try:
                all_charts.extend(_generate_charts(df, prefix=prefix))
            except Exception as e:
                print(f"[图表] {fname} 图表生成失败: {e}")

    report_lines = [
        "=" * 60,
        "QuantView 数据分析报告（分布式）",
        "=" * 60,
        "",
        f"分析文件 ({len(prep['filenames'])} 个): {', '.join(prep['filenames'])}",
        f"分析问题: {question}",
        f"数据总量: {prep['total_rows']} 行 × {prep['total_cols']} 列",
        f"节点数: {len(workers)} 个（{', '.join(n['name'] for n in workers)}），工作表任务数: {len(tasks)}",
        "",
        "-" * 40,
        "AI 分析正文",
        "-" * 40,
        "",
        full,
        "",
        "-" * 40,
        "报告生成完毕",
        "-" * 40,
    ]
    return {
        "result": "\n".join(report_lines),
        "images": all_charts,
        "mode": "distributed" if _distributed_nodes() else "single",
        "nodes": [n["name"] for n in _distributed_nodes()],
    }


@app.route("/analyze/sheet", methods=["POST"])
def analyze_sheet():
    """加速节点的工作表分析端点：接收 {"prompt", "max_tokens"}，返回 {"result": 分析文本}。"""
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "缺少 prompt"}), 400
    try:
        max_tokens = int(data.get("max_tokens", _SHEET_MAX_TOKENS))
    except (TypeError, ValueError):
        max_tokens = _SHEET_MAX_TOKENS
    # 控制台日志：任务标签（工作表名）、开始/完成时间与耗时
    m = re.search(r"工作表「([^」]+)」", prompt[:300])
    label = m.group(1) if m else "未命名任务"
    print(f"[节点任务] 收到: {label} ({len(prompt)} 字符) @ {time.strftime('%H:%M:%S')}")
    t0 = time.time()
    try:
        result = _run_inference(prompt, max_tokens=max_tokens)
        print(f"[节点任务] 完成: {label} 耗时 {time.time()-t0:.1f}s 输出 {len(result)} 字符")
        return jsonify({"result": result})
    except Exception as e:
        print(f"[节点任务] 失败: {label} 耗时 {time.time()-t0:.1f}s 错误: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/sheet/stream", methods=["POST"])
def analyze_sheet_stream():
    """加速节点的流式工作表分析端点：SSE 推送 {"type": "text"|"think"|"done", ...}。"""
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "缺少 prompt"}), 400
    try:
        max_tokens = int(data.get("max_tokens", _SHEET_MAX_TOKENS))
    except (TypeError, ValueError):
        max_tokens = _SHEET_MAX_TOKENS
    m = re.search(r"工作表「([^」]+)」", prompt[:300])
    label = m.group(1) if m else "未命名任务"
    print(f"[节点任务] 收到(流式): {label} ({len(prompt)} 字符) @ {time.strftime('%H:%M:%S')}")
    t0 = time.time()

    def generate():
        gen = _run_inference(prompt, max_tokens=max_tokens, stream=True)
        try:
            for evt_type, chunk in gen:
                if evt_type == "think":
                    yield f"data: {json.dumps({'type': 'think', 'content': chunk}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"
            print(f"[节点任务] 完成(流式): {label} 耗时 {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[节点任务] 失败(流式): {label} 耗时 {time.time()-t0:.1f}s 错误: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # 客户端断开（GeneratorExit）时关闭推理生成器，释放模型避免空转
            try:
                gen.close()
            except BaseException:
                pass
            try:
                yield "data: {\"type\": \"done\"}\n\n"
            except BaseException:
                pass  # 客户端断开（GeneratorExit）时优雅关闭

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 二次追问会话（像聊天一样继续提问） ──
_MAX_SESSIONS = 20
_SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")


def _session_path(session_id):
    return os.path.join(_SESSIONS_DIR, session_id + ".json")


def _persist_session(session_id):
    """把会话写入磁盘（服务重启后追问记录仍在，可直接继续）。"""
    sess = _SESSIONS.get(session_id)
    if not sess:
        return
    try:
        os.makedirs(_SESSIONS_DIR, exist_ok=True)
        with open(_session_path(session_id), "w", encoding="utf-8") as f:
            json.dump({
                "session": session_id,
                "context": sess.get("context", ""),
                "turns": sess.get("turns", []),
                "created": sess.get("created", time.time()),
                "report": sess.get("report", ""),
            }, f, ensure_ascii=False)
    except Exception as e:
        print(f"[追问] 会话持久化失败: {e}")


def _load_followup_sessions():
    """启动时从磁盘恢复追问会话，只保留最近 _MAX_SESSIONS 个。"""
    out = {}
    try:
        if os.path.isdir(_SESSIONS_DIR):
            for fn in os.listdir(_SESSIONS_DIR):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(_SESSIONS_DIR, fn), encoding="utf-8") as f:
                        d = json.load(f)
                    if d.get("session"):
                        out[d["session"]] = {
                            "context": d.get("context", ""),
                            "turns": d.get("turns", []),
                            "created": d.get("created", 0),
                            "report": d.get("report", ""),
                        }
                except Exception:
                    continue
    except Exception:
        pass
    for sid in sorted(out, key=lambda k: out[k].get("created", 0))[:max(0, len(out) - _MAX_SESSIONS)]:
        out.pop(sid)
    return out


_SESSIONS = _load_followup_sessions()


def _build_followup_context(file_buffers):
    """构建追问会话的数据背景：硬数字 + 每文件摘要（控制总长度）。"""
    from werkzeug.datastructures import FileStorage as _FS
    parts = []
    hard = extract_hard_numbers_from_bytes(file_buffers)
    if hard:
        parts.append("=== 硬数字（从Excel精确提取，禁止编造） ===\n" + "\n".join(hard))
    for fname, data in file_buffers:
        try:
            filename, suffix, df = read_file(_FS(stream=io.BytesIO(data), filename=fname))
            s = df_summary(filename, df)
            parts.append(f"=== 文件 {filename} 摘要 ==="
                         f"\n数据规模: {s['num_rows']} 行 × {s['num_cols']} 列"
                         f"\n{s.get('sheet_overview', '')}"
                         f"\n{s.get('per_sheet_summary', '')}"
                         f"\n--- 文本列值分布 ---\n{s.get('dimensions_str', '')}"
                         f"\n--- 缺失值 ---\n{s.get('missing_str', '')}")
        except Exception as e:
            print(f"[追问] 文件摘要失败 {fname}: {e}")
    ctx = "\n\n".join(parts)
    # 控制总长度：27B 模型大上下文预填充很慢，追问只需核对硬数字，截掉冗余逐表统计
    if len(ctx) > 10000:
        ctx = ctx[:10000].rsplit("\n", 1)[0] + "\n（上下文过长已截断）"
    return ctx


def _new_followup_session(file_buffers):
    """创建追问会话（内存 + 磁盘持久化，最多保留 _MAX_SESSIONS 个）。"""
    session_id = os.urandom(8).hex()
    try:
        ctx = _build_followup_context(file_buffers)
    except Exception as e:
        print(f"[追问] 上下文构建失败: {e}")
        ctx = ""
    _SESSIONS[session_id] = {"context": ctx, "turns": [], "created": time.time()}
    if len(_SESSIONS) > _MAX_SESSIONS:
        oldest = min(_SESSIONS, key=lambda k: _SESSIONS[k]["created"])
        _SESSIONS.pop(oldest, None)
        try:
            os.remove(_session_path(oldest))
        except OSError:
            pass
    _persist_session(session_id)
    return session_id


def _split_report_sections(report):
    """把报告按章节标题切分为 [(标题 or None, 正文), ...]，None 表示报告头部（不参与修改）。"""
    sections = []
    cur_h = None
    cur_body = []
    for ln in (report or "").split("\n"):
        if re.match(r"^#{2,3} ", ln):
            sections.append((cur_h, "\n".join(cur_body)))
            cur_h = ln
            cur_body = []
        else:
            cur_body.append(ln)
    sections.append((cur_h, "\n".join(cur_body)))
    return sections


def _apply_report_edit(report, reply):
    """从追问回复中提取【报告修改】块并应用到报告对应章节。

    Returns:
        (新报告文本, 被修改章节标题 or None)
    """
    m = re.search(r"【报告修改】\s*\n*修改章节[:：]\s*([^\n]+?)\s*\n*新内容[:：]\s*\n?(.*)", reply, re.S)
    if not m:
        return report, None
    target = re.sub(r"\s+", "", m.group(1))
    new_body = m.group(2).strip()
    if not target:
        return report, None
    sections = _split_report_sections(report)
    hit = None
    for i, (h, body) in enumerate(sections):
        if not h:
            continue
        hn = re.sub(r"\s+", "", h)
        if target in hn or hn in target:
            sections[i] = (h, new_body)
            hit = h.strip()
            break
    if hit is None:
        return report, None
    parts = []
    for h, body in sections:
        parts.append(f"{h}\n{body}" if h else body)
    return "\n".join(parts), hit


def _build_followup_prompt(sess):
    """追问 prompt：数据背景 + 之前的报告 + 对话历史 + 最新问题。"""
    question = sess["turns"][-1][1]
    # 编辑类问题需要完整报告才能准确改章节；普通问题只给开头摘要，控制预填充长度
    edit_hints = ("修改", "调整", "补充", "加上", "加入", "删除", "去掉", "删掉", "更新", "重写", "替换", "换成", "改")
    if any(k in question for k in edit_hints):
        report_part = sess.get("report") or "（无）"
    else:
        report_part = (sess.get("report") or "（无）")[:4000]
    parts = [
        "你是一位资深的企业经营数据分析师。这是一次【追问对话】：用户基于之前已完成的报告，"
        "针对某个具体问题进行补充提问、质疑、要求解释或要求修改报告。",
        "铁律：",
        "1. 只回答用户的最新问题，禁止重新逐表分析，禁止重复或改写之前的报告内容；",
        "2. 禁止输出报告结构：禁止'### 工作表'标题、禁止【图表数据】区块、禁止逐板块罗列表格；",
        "3. 引用具体数值时以硬数字为准，禁止编造；无数据支撑的明确说明；",
        "4. 用户指出之前回答有误或不到位时，先认错纠正，再核对数据作答；",
        "5. 回答用 Markdown，简洁直接，不超过 800 字。",
        "6. 如果用户要求修改/更新报告内容（如'把结论改成…'、'在风险提示里补充…'、'删掉某段'），"
        "先写一句话确认，然后在回答末尾输出一个【报告修改】块（不要省略任何字段）：",
        "【报告修改】",
        "修改章节: <要修改章节的标题，必须与报告中的章节标题一致，例如 ## 总览与综合结论（主节点生成）>",
        "新内容:",
        "<该章节的完整新正文，不含标题行。如果原章节末尾有```chartjson```代码块，必须原样保留在新内容末尾>",
        "7. 用户没有要求修改报告时，禁止输出【报告修改】块，正常回答即可。",
        "",
        "=== 数据背景（仅供核对数值，不要重新展开分析） ===",
        sess["context"] or "（无数据背景）",
        "",
        "=== 之前的分析报告（修改需求时给完整报告，否则仅供引用，不要重述） ===",
        report_part,
        "",
        "=== 对话历史 ===",
    ]
    for role, content in sess["turns"][:-1]:
        who = "用户" if role == "user" else "助手"
        parts.append(f"{who}：{content[:600]}")
    parts.append("")
    parts.append(f"=== 最新问题 ===\n{sess['turns'][-1][1]}")
    return "\n".join(parts)


@app.route("/analyze/followup", methods=["POST"])
def analyze_followup():
    """二次追问：基于会话的数据背景 + 对话历史流式回答（SSE）。"""
    data = request.get_json(force=True, silent=True) or {}
    session_id = (data.get("session") or "").strip()
    question = (data.get("question") or "").strip()
    print(f"[追问] /analyze/followup 收到问题: {question[:80]} (session={session_id[:8]})")
    if not session_id or not question:
        return jsonify({"error": "缺少 session 或 question"}), 400
    sess = _SESSIONS.get(session_id)
    if not sess:
        return jsonify({"error": "会话已失效（服务可能已重启），请重新发起分析"}), 404

    sess["turns"].append(("user", question))
    # 历史截断：保留最近 8 轮
    if len(sess["turns"]) > 12:
        sess["turns"] = sess["turns"][-12:]
    _persist_session(session_id)

    def generate():
        output = []
        gen = _run_inference(_build_followup_prompt(sess), max_tokens=8192, stream=True)
        try:
            for evt, chunk in gen:
                if evt == "think":
                    yield f"data: {json.dumps({'type': 'think', 'content': chunk}, ensure_ascii=False)}\n\n"
                else:
                    output.append(chunk)
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # 客户端断开（GeneratorExit）/异常时关闭推理生成器，释放模型避免空转占用
            try:
                gen.close()
            except BaseException:
                pass
            if output:
                reply_text = "".join(output)
                # 应用【报告修改】块：更新会话报告、推送 report_edit 事件，回复文本剔除修改块
                new_report, section_hit = _apply_report_edit(sess.get("report", ""), reply_text)
                if section_hit:
                    sess["report"] = new_report
                    try:
                        yield f"data: {json.dumps({'type': 'report_edit', 'new_report': new_report, 'section': section_hit}, ensure_ascii=False)}\n\n"
                    except BaseException:
                        pass  # 客户端断开（GeneratorExit）
                    reply_text = re.sub(r"【报告修改】[\s\S]*$", "", reply_text).strip()
                sess["turns"].append(("assistant", reply_text))
                if len(sess["turns"]) > 12:
                    sess["turns"] = sess["turns"][-12:]
                _persist_session(session_id)
            try:
                yield "data: {\"type\": \"done\"}\n\n"
            except BaseException:
                pass  # 客户端断开（GeneratorExit）时优雅关闭

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 轮询式分析（任务在后台线程执行，前端轮询取增量事件） ──
# 背景：超长 SSE 连接在部分浏览器/系统环境下 ~5 分钟静默断流（服务端写阻塞、worker 继续空算），
# 轮询方式每次都是短连接，从根上规避该问题。
_JOBS = {}


def _job_worker(job_id, prep, question, session_id, distributed, stop_event):
    job = _JOBS[job_id]
    model_output_parts = []
    try:
        job["events"].append({"type": "meta", "mode": "distributed" if distributed else "single",
                              "nodes": [n["name"] for n in _distributed_nodes()],
                              "split": True, "session": session_id})
        header = "\n".join([
            "=" * 60, "QuantView 数据分析报告", "=" * 60, "",
            f"分析文件 ({len(prep['filenames'])} 个): {', '.join(prep['filenames'])}",
            f"分析问题: {question}",
            f"数据总量: {prep['total_rows']} 行 × {prep['total_cols']} 列",
            "", "-" * 40, "AI 分析正文", "-" * 40, "",
        ])
        job["events"].append({"type": "text", "content": header})
        for evt_type, chunk in _distributed_analysis_events(prep, question, model_output_parts, stop_event):
            if stop_event.is_set():
                break
            if evt_type == "think":
                job["events"].append({"type": "think", "content": chunk})
            elif evt_type == "progress":
                job["events"].append({"type": "progress", "done": chunk["done"], "total": chunk["total"]})
            else:
                job["events"].append({"type": "text", "content": chunk})
        if not stop_event.is_set():
            full_output = "".join(model_output_parts)
            full_output = _eval_calc_markers(full_output)
            if not full_output or not full_output.strip():
                full_output = "（模型未生成有效回复，请重试）"
            _SESSIONS[session_id]["report"] = full_output
            _persist_session(session_id)
            job["events"].append({"type": "text", "content": "\n\n" + "-" * 40 + "\n报告生成完毕\n" + "-" * 40})
            all_charts = _generate_charts_from_json(full_output)
            if not all_charts:
                for fname, df in prep["dataframes"]:
                    prefix = fname if len(prep["dataframes"]) > 1 else ""
                    try:
                        all_charts.extend(_generate_charts(df, prefix=prefix))
                    except Exception as e:
                        print(f"[图表] {fname} 图表生成失败: {e}")
            if all_charts:
                job["events"].append({"type": "charts", "images": all_charts})
            try:
                threading.Thread(target=_send_report_email,
                                 args=(question, prep["filenames"], full_output, all_charts),
                                 daemon=True).start()
            except Exception:
                pass
    except Exception as e:
        import traceback
        print(f"[Job Error] {traceback.format_exc()}")
        job["error"] = str(e)
    finally:
        stop_event.set()
        job["done"] = True
        job["session"] = session_id


@app.route("/analyze/start", methods=["POST"])
def analyze_start():
    """轮询式分析启动：后台线程执行分析，返回 job id，前端轮询 /analyze/poll 取增量事件。"""
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "未检测到上传文件"}), 400
    valid_files = []
    for f in files:
        if f.filename == "":
            continue
        if not allowed_file(f.filename):
            return jsonify({"error": f"不支持的文件格式: {f.filename}"}), 400
        valid_files.append(f)
    if not valid_files:
        return jsonify({"error": "未选择有效的文件"}), 400
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "分析问题不能为空"}), 400
    print(f"[分析] /analyze/start 收到问题: {question[:80]}")

    from werkzeug.datastructures import FileStorage as _FS
    file_buffers = []
    fresh_files = []
    for f in valid_files:
        buf = io.BytesIO()
        f.save(buf)
        data = buf.getvalue()
        file_buffers.append((f.filename, data))
        fresh_files.append(_FS(stream=io.BytesIO(data), filename=f.filename))
    valid_files = fresh_files

    distributed = bool(_distributed_nodes())
    try:
        prep = _prepare_distributed_input(valid_files, question)
    except Exception as e:
        return jsonify({"error": f"数据准备失败: {str(e)}"}), 500

    session_id = _new_followup_session(file_buffers)
    job_id = os.urandom(8).hex()
    stop_event = threading.Event()
    _JOBS[job_id] = {"events": [], "done": False, "error": None, "session": None,
                     "created": time.time(), "stop_event": stop_event}
    # 清理旧任务（最多保留 8 个）
    if len(_JOBS) > 8:
        for old in sorted(_JOBS, key=lambda k: _JOBS[k]["created"])[:len(_JOBS) - 8]:
            _JOBS.pop(old, None)
    threading.Thread(target=_job_worker,
                     args=(job_id, prep, question, session_id, distributed, stop_event),
                     daemon=True).start()
    return jsonify({"job": job_id})


@app.route("/analyze/poll", methods=["GET"])
def analyze_poll():
    """轮询取增量事件：?job=xxx&from=N → {from, done, error, events:[...]}"""
    job_id = request.args.get("job", "")
    job = _JOBS.get(job_id)
    if not job:
        return jsonify({"error": "任务不存在或已过期"}), 404
    try:
        from_idx = int(request.args.get("from", 0) or 0)
    except (TypeError, ValueError):
        from_idx = 0
    events = job["events"][from_idx:]
    return jsonify({
        "job": job_id,
        "from": from_idx + len(events),
        "done": job["done"],
        "error": job["error"],
        "session": job["session"],
        "events": events,
    })


@app.route("/analyze/cancel", methods=["POST"])
def analyze_cancel():
    """取消后台分析任务（置位 stop_event，worker 提前终止释放模型）。"""
    data = request.get_json(silent=True) or {}
    job_id = data.get("job") or request.form.get("job", "")
    job = _JOBS.get(job_id)
    if job:
        job["stop_event"].set()
        print(f"[分析] 任务 {job_id[:8]} 已取消")
    return jsonify({"ok": True})


@app.route("/analyze/stream", methods=["POST"])
def analyze_stream():
    """流式分析接口 — SSE 逐字推送模型输出。"""
    # 1. 校验文件
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "未检测到上传文件"}), 400

    valid_files = []
    for f in files:
        if f.filename == "":
            continue
        if not allowed_file(f.filename):
            return jsonify({"error": f"不支持的文件格式: {f.filename}"}), 400
        valid_files.append(f)

    if not valid_files:
        return jsonify({"error": "未选择有效的文件"}), 400

    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "分析问题不能为空"}), 400
    print(f"[分析] /analyze/stream 收到问题: {question[:80]}")

    # 2. 保存文件字节（追问会话与分析共用，FileStorage 流只能读一次）
    from werkzeug.datastructures import FileStorage as _FS
    file_buffers = []
    fresh_files = []
    for f in valid_files:
        buf = io.BytesIO()
        f.save(buf)
        data = buf.getvalue()
        file_buffers.append((f.filename, data))
        fresh_files.append(_FS(stream=io.BytesIO(data), filename=f.filename))
    valid_files = fresh_files

    # 3. 准备分析输入（一律按工作表拆分任务；无加速节点时主节点自己逐表处理）
    distributed = bool(_distributed_nodes())
    try:
        prep = _prepare_distributed_input(valid_files, question)
    except Exception as e:
        return jsonify({"error": f"数据准备失败: {str(e)}"}), 500

    # 追问会话（含数据背景）
    session_id = _new_followup_session(file_buffers)

    def generate():
        model_output_parts = []
        # 客户端断开/分析中止时置位，后台 worker 提前终止，避免模型空转
        stop_event = threading.Event()
        try:
            # 推送模式标识（单节点/分布式）+ 追问会话 id，供前端展示徽标
            # split=true：一律按工作表逐表处理（单节点=主节点自己逐表）
            meta = {"type": "meta", "mode": "distributed" if distributed else "single",
                    "nodes": [n["name"] for n in _distributed_nodes()],
                    "split": True,
                    "session": session_id}
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            # 推送报告头
            header = "\n".join([
                "=" * 60,
                "QuantView 数据分析报告",
                "=" * 60,
                "",
                f"分析文件 ({len(valid_files)} 个): {', '.join(prep['filenames'])}",
                f"分析问题: {question}",
                f"数据总量: {prep['total_rows']} 行 × {prep['total_cols']} 列",
                "",
                "-" * 40,
                "AI 分析正文",
                "-" * 40,
                "",
            ])
            yield f"data: {json.dumps({'type': 'text', 'content': header}, ensure_ascii=False)}\n\n"

            # 流式推理：逐表处理（有加速节点时并行分发 + 保序投递；单节点时主节点自己逐表）
            for evt_type, chunk in _distributed_analysis_events(prep, question, model_output_parts, stop_event):
                if evt_type == "think":
                    yield f"data: {json.dumps({'type': 'think', 'content': chunk}, ensure_ascii=False)}\n\n"
                elif evt_type == "progress":
                    yield f"data: {json.dumps({'type': 'progress', 'done': chunk['done'], 'total': chunk['total']}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"

            full_output = "".join(model_output_parts)
            full_output = _eval_calc_markers(full_output)  # 兜底清理残留 calc 标记
            if not full_output or not full_output.strip():
                full_output = "（模型未生成有效回复，请重试）"

            # 报告存入追问会话，供"接着问/修正"时引用
            _SESSIONS[session_id]["report"] = full_output
            _persist_session(session_id)

            # 推送报告尾
            footer = "\n\n" + "-" * 40 + "\n报告生成完毕\n" + "-" * 40
            yield f"data: {json.dumps({'type': 'text', 'content': footer}, ensure_ascii=False)}\n\n"

            # 生成并推送图表：优先 AI 指定，回退自动
            all_charts = _generate_charts_from_json(full_output)
            if not all_charts:
                for fname, df in prep["dataframes"]:
                    prefix = fname if len(prep["dataframes"]) > 1 else ""
                    try:
                        charts = _generate_charts(df, prefix=prefix)
                        all_charts.extend(charts)
                    except Exception as e:
                        print(f"[图表] {fname} 图表生成失败: {e}")

            if all_charts:
                yield f"data: {json.dumps({'type': 'charts', 'images': all_charts}, ensure_ascii=False)}\n\n"

            # 分析完成 → 异步发送邮件（可选，未配置收件人时自动跳过）
            try:
                threading.Thread(target=_send_report_email,
                                 args=(question, prep["filenames"], full_output, all_charts),
                                 daemon=True).start()
            except Exception:
                pass

        except Exception as e:
            import traceback
            print(f"[Stream Error] {traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # 无论正常完成还是客户端断开，都通知 worker 停止，释放模型
            stop_event.set()
            try:
                yield "data: {\"type\": \"done\"}\n\n"
            except BaseException:
                pass  # 客户端断开（GeneratorExit）时优雅关闭

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def perform_analysis(files, question):
    """执行数据分析 —— 按工作表逐表处理（无加速节点时主节点自己处理）+ 总览 + 图表。

    Returns:
        {"result": 文本报告, "images": [(标题, base64), ...], "mode", "nodes"}
    """
    prep = _prepare_distributed_input(files, question)
    return _perform_analysis_distributed(prep, question)


@app.route("/export/docx", methods=["POST"])
def export_docx():
    """将 HTML 报告导出为 Word 文档（调用独立脚本）。"""
    data = request.get_json(force=True)
    if not data or not (data.get("text") or data.get("html")):
        return jsonify({"error": "缺少报告内容"}), 400

    html_content = data.get("html", data.get("text", ""))
    title = data.get("title", "QuantView 分析报告")

    # 优先用独立脚本
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_docx.py")
    if os.path.isfile(script):
        import tempfile
        import subprocess
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", encoding="utf-8", delete=False) as f_in:
            f_in.write(html_content)
            html_path = f_in.name
        out_path = html_path.replace(".html", ".docx")
        try:
            result = subprocess.run(
                [sys.executable, script, html_path, out_path, "--title", title],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or result.stdout)
            return send_file(
                out_path,
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                as_attachment=True,
                download_name=f'QuantView_{title[:30]}.docx'
            )
        finally:
            try: os.unlink(html_path)
            except: pass
            try: os.unlink(out_path)
            except: pass

    return jsonify({"error": "Word 导出脚本未找到，请安装 export_docx.py"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("QuantView - 本地数据分析助手")
    print("=" * 60)
    print()
    _port = int(os.environ.get("DEEPANALYZE_PORT", "5000"))
    if _HEADLESS:
        print(f" 运行模式: 加速节点（无头，无 Web 界面）")
        print(f" 任务接口: POST http://localhost:{_port}/analyze/sheet")
    elif _NODE_LIST:
        print(f" 运行模式: 分布式主节点（加速节点: {_NODE_LIST}）")
        print(" 前端页面: http://localhost:%d（加速节点页面对应端口单独访问）" % _port)
    else:
        print(f" 运行模式: 单节点")
        print(" 前端页面: http://localhost:%d/" % _port)
    if not _HEADLESS:
        print(f" 分析接口: POST http://localhost:{_port}/analyze")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    # 端口可用 DEEPANALYZE_PORT 覆盖（多实例/分布式同机部署时需要）
    app.run(host="0.0.0.0", port=_port, debug=False)
