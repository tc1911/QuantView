"""DeepAnalyze - 本地数据分析助手 (Flask 后端入口)

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
_LOCAL_SERVER_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "repeat_penalty": 1.15,
    "repeat_last_n": 512,
}
# 硬数字进入 prompt 的最大字符数（默认 12000，覆盖全部 11 张表约需 8500）
# 太小会导致资金状况/经营预算等表被截断，模型误判"无数据"
_HARD_NUMS_LIMIT = int(os.environ.get("DEEPANALYZE_HARD_NUMS_LIMIT", "12000"))
# 采样温度（默认 0.5：报告是确定性任务，低温度降低单次跑飞/对话式输出概率）
_TEMPERATURE = float(os.environ.get("DEEPANALYZE_TEMPERATURE", "0.5"))

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
    # torch / transformers / llama-cpp 均按需导入
    _HAS_TORCH = False
    _HAS_LLAMA_CPP = False
    AutoModelForCausalLM = AutoTokenizer = TextIteratorStreamer = None
    torch = None
    Thread = None
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

# ── 模型发现与选择 ──
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_model = None
_tokenizer = None
MODEL_PATH = None
MODEL_TYPE = "hf"  # "hf" 或 "gguf"


def _scan_models():
    """扫描 models/ 目录，返回可用模型列表 {名称: (路径, 类型)}。

    类型: "hf" = HuggingFace, "gguf" = GGUF 格式
    """
    if not os.path.isdir(_MODELS_DIR):
        print(f"[模型] models/ 目录不存在: {_MODELS_DIR}")
        return {}

    _HF_SIGNATURE = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "model-00001-of-",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    available = {}

    # ── 扫描子目录（HF 格式 + 目录内的 GGUF 文件）──
    all_items = os.listdir(_MODELS_DIR)
    all_dirs = [d for d in all_items if os.path.isdir(os.path.join(_MODELS_DIR, d))]

    print(f"[模型] 扫描 models/ 目录: 发现 {len(all_dirs)} 个子目录, {len(all_items)} 个条目")

    for entry in all_dirs:
        full = os.path.join(_MODELS_DIR, entry)
        matched = ""
        model_type = "hf"

        # 检查 HF 签名
        for sig in _HF_SIGNATURE:
            if sig.endswith("-"):
                if any(f.startswith(sig) for f in os.listdir(full)):
                    matched = sig
                    break
            elif os.path.isfile(os.path.join(full, sig)):
                matched = sig
                break

        if matched:
            available[entry] = (full, "hf")
            print(f"  ✓ {entry} (HF, 匹配: {matched})")
            continue

        # 检查目录内的 GGUF 文件
        gguf_files = [f for f in os.listdir(full) if f.endswith(".gguf")]
        if gguf_files:
            gguf_path = os.path.join(full, gguf_files[0])
            available[entry] = (gguf_path, "gguf")
            print(f"  ✓ {entry} (GGUF: {gguf_files[0]})")
            continue

        subfiles = os.listdir(full)[:5]
        print(f"  ✗ {entry} — 无模型签名，目录内容: {subfiles}")

    # ── 扫描 models/ 根目录下的 .gguf 文件 ──
    for item in all_items:
        full = os.path.join(_MODELS_DIR, item)
        if os.path.isfile(full) and item.endswith(".gguf"):
            name = item.replace(".gguf", "")
            if name not in available:
                available[name] = (full, "gguf")
                print(f"  ✓ {name} (GGUF 单文件: {item})")

    return available


def _select_model():
    """根据可用模型数量和环境变量选择模型路径。

    优先级: DEEPANALYZE_MODEL 环境变量 > 单模型自动选择 > 多模型报错提示
    """
    if DEBUG_MODE:
        return None

    available = _scan_models()
    if not available:
        print("=" * 60)
        print(" 错误: models/ 目录下未找到任何模型！")
        print(f" 目录: {_MODELS_DIR}")
        print(" 请确保模型文件夹内包含以下任一文件:")
        print("   config.json / model.safetensors / pytorch_model.bin / tokenizer.json")
        print(" 使用调试模式可跳过本地模型:")
        print("   DEEPANALYZE_DEBUG=true DEEPSEEK_API_KEY=sk-xxx python app.py")
        print("=" * 60)
        # 调试模式下不退出，其他模式退出
        if not DEBUG_MODE:
            exit(1)
        return None

    env_model = os.environ.get("DEEPANALYZE_MODEL", "").strip()
    if env_model:
        if env_model in available:
            return available[env_model]  # (path, mtype)
        else:
            print("=" * 60)
            print(f" 错误: 指定的模型 '{env_model}' 未找到")
            print(f" 可用模型: {', '.join(available.keys())}")
            print("=" * 60)
            exit(1)

    if len(available) == 1:
        name, (path, mtype) = next(iter(available.items()))
        type_tag = "GGUF" if mtype == "gguf" else "HF"
        print(f"[模型] 自动选择唯一可用模型: {name} ({type_tag})")
        return path, mtype

    # 多个模型，手动选择
    sorted_models = sorted(available.items())
    print("=" * 60)
    print(f" 发现 {len(available)} 个可用模型:")
    for i, (name, (_, mtype)) in enumerate(sorted_models, 1):
        type_tag = "GGUF" if mtype == "gguf" else "HF"
        print(f"   [{i}] {name}  ({type_tag})")
    print()

    while True:
        try:
            choice = input(f" 请选择模型 [1-{len(sorted_models)}]: ").strip()
            idx = int(choice)
            if 1 <= idx <= len(sorted_models):
                name, (path, mtype) = sorted_models[idx - 1]
                type_tag = "GGUF" if mtype == "gguf" else "HF"
                print(f"\n 已选择: {name} ({type_tag})")
                print("=" * 60)
                return path, mtype
            print(f" 请输入 1 到 {len(sorted_models)} 之间的数字")
        except (ValueError, EOFError):
            print(" 输入无效，请输入数字")
        except KeyboardInterrupt:
            print("\n\n 已取消")
            exit(0)


def get_model_and_tokenizer():
    """懒加载本地模型。调试模式下跳过，返回 (model, tokenizer)。

    HF 模型: tokenizer 是 AutoTokenizer, model 是 AutoModel
    GGUF 模型: tokenizer 是 Llama 对象（自带 tokenize）, model 也是同一个 Llama
    调用方统一使用 (model, tokenizer) 接口。
    """
    global _model, _tokenizer, MODEL_PATH, MODEL_TYPE

    if DEBUG_MODE:
        return None, None

    if MODEL_PATH is None:
        raise RuntimeError("未配置模型路径，请检查 models/ 目录")

    if _model is None:
        model_name = os.path.basename(MODEL_PATH)
        print("=" * 60)
        print(f"正在加载模型: {model_name}")
        print(f"类型: {'GGUF' if MODEL_TYPE == 'gguf' else 'HuggingFace'}")
        print(f"路径: {MODEL_PATH}")
        print("首次加载需 10-30 秒，请稍候...")
        print("=" * 60)

        if MODEL_TYPE == "gguf":
            # ── GGUF 模型：启动外部 llama-server 进程 ──
            import subprocess
            import time

            # 查找 llama-server 二进制
            server_bin = os.environ.get("LLAMA_SERVER_PATH", "")
            if not server_bin:
                # 自动查找：项目目录、常见路径
                candidates = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama-server.exe"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama-server"),
                    "llama-server", "llama-server.exe",
                    "C:/Users/tc191/llama-cpp/llama-server.exe",
                ]
                for c in candidates:
                    if os.path.isfile(c) or shutil.which(c):
                        server_bin = c
                        break

            if not server_bin:
                raise RuntimeError(
                    "未找到 llama-server 二进制。请设置环境变量 LLAMA_SERVER_PATH\n"
                    "或下载 llama.cpp 到项目目录: https://github.com/ggerganov/llama.cpp/releases"
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

            _llama_proc = subprocess.Popen(
                [server_bin, "-m", MODEL_PATH, "--port", str(port),
                 "-ngl", "99", "-c", "65536", "--host", "127.0.0.1",
                 # 服务端禁用 Qwen3 思考模式（请求级 chat_template_kwargs 对部分模型无效）
                 "--chat-template-kwargs", '{"enable_thinking": false}'],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )

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
                raise RuntimeError("llama-server 启动超时")

            # 设置全局 API URL，推理代码自动走 API 路径
            global DEEPSEEK_API_URL
            DEEPSEEK_API_URL = f"http://127.0.0.1:{port}/v1/chat/completions"
            _model = f"llama-server:{port}"
            _tokenizer = f"llama-server:{port}"

            # 注册退出清理
            import atexit
            atexit.register(lambda: _llama_proc.kill() if _llama_proc.poll() is None else None)

            print(f"[GGUF] llama-server 已就绪 -> {DEEPSEEK_API_URL}")

        else:
            # ── HuggingFace 模型加载（按需导入）──
            global _HAS_TORCH
            if not _HAS_TORCH:
                import torch as _t
                from transformers import AutoModelForCausalLM as _AM, AutoTokenizer as _AT, TextIteratorStreamer as _TS
                from threading import Thread as _Th
                globals().update(torch=_t, AutoModelForCausalLM=_AM, AutoTokenizer=_AT,
                                 TextIteratorStreamer=_TS, Thread=_Th)
                _HAS_TORCH = True
            try:
                _tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                )
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except (ValueError, ImportError) as e:
                msg = str(e)
                if "does not recognize this architecture" in msg or "not supported" in msg:
                    print("=" * 60)
                    print(" 模型架构不支持！transformers 版本过旧。")
                    print(f" 当前模型: {model_name}")
                    print(" 请执行: pip install --upgrade transformers")
                    print(" 或选择其他可用模型。")
                    print("=" * 60)
                    raise RuntimeError(f"模型架构不支持，请升级 transformers: {msg}")
                raise

            print("HF 模型加载完成。")
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token

    return _model, _tokenizer


# ── 启动时扫描并加载模型 ──
if not DEBUG_MODE:
    print("=" * 60)
    print(" 模型发现")
    print("=" * 60)
    MODEL_PATH, MODEL_TYPE = _select_model()
    if MODEL_PATH:
        type_tag = "GGUF" if MODEL_TYPE == "gguf" else "HF"
        print(f" 已选择模型: {os.path.basename(MODEL_PATH)} ({type_tag})")
        print(f" 路径: {MODEL_PATH}")
        print("=" * 60)
        get_model_and_tokenizer()
    else:
        print("=" * 60)


@app.route("/")
def index():
    """前端入口页面"""
    resp = app.send_static_file("index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


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

    # 3. 执行分析
    try:
        output = perform_analysis(valid_files, question)
        return jsonify({
            "result": output["result"],
            "images": output.get("images", []),
            "mode": output.get("mode", "single"),
            "nodes": output.get("nodes", []),
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
        try:
            chart_specs = json.loads(match.group(1))
        except json.JSONDecodeError:
            print("[图表] chartjson 解析失败，跳过该块")
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

    print("[DeepSeek API] 发送推理请求...")
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
    print(f"[DeepSeek API] 推理完成 — 输入 {usage.get('prompt_tokens', '?')} tokens, "
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

    print(f"[DeepSeek API] 发送流式推理请求 (模型: {DEEPSEEK_MODEL})...")
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"DeepSeek API 返回错误 ({e.code}): {body}")

    chunk_count = 0
    think_count = 0
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

    resp.close()
    print(f"[DeepSeek API] 流式推理完成 — {think_count} 个思考块, {chunk_count} 个文本块")


def _run_inference(prompt, max_tokens=16384, stream=False):
    """统一推理入口，覆盖三种后端（DeepSeek API / GGUF / HF）。

    stream=False: 返回分析文本字符串。
    stream=True: 返回生成器，产出 ("think"|"text", chunk) 元组。
    """
    if DEBUG_MODE:
        if stream:
            return _call_deepseek_api_stream(prompt, max_tokens=max_tokens)
        return _call_deepseek_api(prompt, max_tokens=max_tokens)

    model, tokenizer = get_model_and_tokenizer()

    if MODEL_TYPE == "gguf":
        if isinstance(model, str) and model.startswith("llama-server:"):
            # 外部 llama-server → 走 API 路径（max_tokens 对齐本地模型）
            if stream:
                return _call_deepseek_api_stream(prompt, max_tokens=65536, extra_body=_LOCAL_SERVER_BODY)
            return _call_deepseek_api(prompt, max_tokens=65536, extra_body=_LOCAL_SERVER_BODY)

        # ── 内嵌 llama-cpp-python ──
        def _gguf_llama_stream():
            for chunk in model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "你是财务分析师。逐表深度分析，每个表输出150字以上。引用预计算汇总中的具体数字。不要概括，要详细。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=65536, temperature=_TEMPERATURE, top_p=0.9,
                stream=True, chat_template_kwargs={"enable_thinking": False},
                repeat_penalty=1.15, repeat_last_n=512):
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield ("text", text)
        if stream:
            return _gguf_llama_stream()
        try:
            result = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "直接输出分析报告，用 Markdown。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=65536, temperature=_TEMPERATURE, top_p=0.9,
                chat_template_kwargs={"enable_thinking": False},
                repeat_penalty=1.15, repeat_last_n=512)
            output = result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"GGUF 推理出错: {str(e)}")
        if not output or not output.strip():
            output = "（模型未生成有效回复，请重试）"
        return output

    # ── HuggingFace 模型推理 (transformers) ──
    try:
        if stream:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=8192).to(model.device)

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=4096,
                temperature=_TEMPERATURE,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                streamer=streamer,
            )
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            def _hf_stream():
                for token_text in streamer:
                    yield ("text", token_text)
                thread.join()
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return _hf_stream()

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=4096, temperature=_TEMPERATURE, top_p=0.9,
                                     do_sample=True, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        model_output = tokenizer.decode(gen_ids, skip_special_tokens=True)
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not model_output or not model_output.strip():
            model_output = "（模型未生成有效回复，请重试）"
        return model_output
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("CUDA 显存不足。请尝试减少文件数据量。")
    except Exception as e:
        raise RuntimeError(f"模型推理出错: {str(e)}")


def _prepare_analysis_input(valid_files, question):
    """准备分析所需的 prompt 和数据框（供 /analyze 和 /analyze/stream 共用）。"""
    import traceback
    try:
        return _prepare_analysis_input_impl(valid_files, question)
    except Exception:
        print(f"[_prepare ERROR] {traceback.format_exc()}")
        raise

def _prepare_analysis_input_impl(valid_files, question):
    total_rows = 0
    total_cols = 0
    filenames = []
    summaries = []
    dataframes = []

    # 先保存文件原始数据（FileStorage 流只能读一次）
    from werkzeug.datastructures import FileStorage as _FS
    file_buffers = []
    fresh_files = []
    for f in valid_files:
        buf = io.BytesIO()
        f.save(buf)
        data = buf.getvalue()
        file_buffers.append((f.filename, data))
        fresh_files.append(_FS(stream=io.BytesIO(data), filename=f.filename))

    for f in fresh_files:
        try:
            filename, suffix, df = read_file(f)
        except Exception as e:
            raise RuntimeError(f"文件 {f.filename} 读取失败: {str(e)}")

        filenames.append(filename)
        dataframes.append((filename, df))

        try:
            summary = df_summary(filename, df)
        except Exception as e:
            raise RuntimeError(f"数据摘要生成失败: {str(e)}")
        summaries.append(summary)

        total_rows += summary["num_rows"]
        total_cols += summary["num_cols"]

    # ── 硬数字速查：用保存的原始字节重新读取（FileStorage 已被消耗）──
    hard_nums_str = ""
    try:
        hard_nums = extract_hard_numbers_from_bytes(file_buffers)
        if hard_nums:
            hard_nums_str = "\n".join(hard_nums)
            if len(hard_nums_str) > _HARD_NUMS_LIMIT:
                hard_nums_str = hard_nums_str[:_HARD_NUMS_LIMIT].rsplit("\n", 1)[0] + "\n（已截断）"
    except Exception:
        pass

    # 构造提示词（硬数字放最前面）
    prompt_parts = [
        "你是一位资深的企业经营数据分析师。下面是一个经营数据文件的完整数据摘要，"
        "请对用户提出的问题进行深入、全面的分析。",
        "",
        "⚠️ 以下【硬数字】为从 Excel 中精确提取的数值，绝对正确，禁止修改，必须原样引用。",
        "每个数字都标注了期别标签（如 2026上半年=7400），使用前请核对期别。",
        "",
    ]
    if hard_nums_str:
        prompt_parts.append("=== ❌硬数字-禁止编造-必须引用 ===\n"
                           "⚠️ 以下数值从Excel精确提取，绝对正确，必须原样引用，禁止修改。\n"
                           "⚠️ 每个标签标注了期别（如2026H1=当前报告期），请使用对应[当前报告期]的值。\n"
                           "⚠️ 硬数字行的格式为「指标名(期别=数值)」，如「应收票据(期末=13600)」表示期末应收票据为13600万元。\n"
                           "⚠️ 负号只可能出现在数值本身（如-100表示负值），括号内是期别标签，不是负数；"
                           "同一指标通常有'期初'和'期末'两行，必须成对核对后再解读增减方向。\n"
                           "⚠️ 如同时出现'期初'和'期末'，请使用期末值（期末=当前报告期末的余额）。\n"
                           "⚠️ 引用硬数字必须保留原始指标名（如净利率），禁止改写成硬数字中未出现的指标名（如毛利率）。\n"
                           "⚠️ 只有当期数据（无历史期标签）的指标严禁编造对比期或变动率，只能陈述当期数值。\n"
                           "⚠️ 报告中出现的任何数值必须能在硬数字中找到完全匹配的行（指标名+期别一致）；"
                           "找不到出处的数值禁止填入，一律留空或写'无数据'；禁止挪用其他指标行的数值充当历史期。\n"
                           "⚠️ 对比表的'期初/年初'值必须使用同一指标紧随'期末'的那期（标签为'年初'或'2025H2'），"
                           "禁止混用'2025H1'等更早期别的数值。\n"
                           "⚠️ '基准/冲刺'是两档目标值，不是时间序列，禁止当作期初/期末计算变动率；"
                           "预算执行应对比实际值与预算目标（如实际销售额 vs 预算基准）。\n"
                           + hard_nums_str + "\n=== 结束 ===")
        prompt_parts.append("")
    prompt_parts.append(f"共 {len(valid_files)} 个文件，总计 {total_rows} 行数据。")
    prompt_parts.append("")

    for i, s in enumerate(summaries, 1):
        prompt_parts.append(f"=== 文件 {i}: {s['filename']} ===")
        prompt_parts.append(f"数据规模: {s['num_rows']} 行 × {s['num_cols']} 列")
        if s.get("sheet_overview"):
            prompt_parts.append(s["sheet_overview"])
        prompt_parts.append("")

        # 有硬数字时只传 sheet_overview，其余全跳过，防模型读 raw data 编造
        if not hard_nums_str:
            if s.get("per_sheet_summary"):
                prompt_parts.append("--- 预计算汇总 ---")
                prompt_parts.append(s['per_sheet_summary'])
                prompt_parts.append("")
            prompt_parts.append("--- 头部数据 ---")
            prompt_parts.append(s['head_str'])
            prompt_parts.append("")
            prompt_parts.append("--- 尾部数据 ---")
            prompt_parts.append(s['tail_str'])
            prompt_parts.append("")
            prompt_parts.append("--- 文本列值分布 ---")
            prompt_parts.append(s['dimensions_str'])
            prompt_parts.append("")
            prompt_parts.append("--- 缺失值 ---")
            prompt_parts.append(s['missing_str'])
            prompt_parts.append("")

    # 铁律放在用户问题正前方，服从度最高（放在长文末尾会被忽略）
    prompt_parts.append("""【铁律 - 违反任何一条视为分析失败】
1. 数据概览表必须列出核心指标（至少 3 行、至多 8 行，每行含指标名+数值），不得省略为空。
2. 变化分析/业务解读/策略建议/风险提示每节各写 1-3 段，禁止每个指标单独一段的模板化、逐项式输出。
3. 整份报告（不含图表数据区块）控制在 6000 字以内。
4. 每个数值必须有硬数字出处（指标名+期别完全匹配），无出处的写"无数据"。
5. 硬数字中不存在的期别或指标不得出现。
6. 只有当期数据的表，历史期列必须写"无数据"，禁止把当期值复制到历史期列、禁止从其他指标行取值。
7. 图表数据不得用 0、重复值或其他指标行的数值填充缺失期别。
8. 禁止连续重复输出相同或雷同的内容（如同一指标行重复列出多次）。
9. 变化分析的对比期必须与上方对比表一致（以"2025H2/年初"列为基准），禁止改用"2025H1"等其他列计算变动率。
10. 历史期为"无数据"的指标只陈述当期数值，禁止写出任何增幅/降幅。
11. "预算金额/合同金额/已付金额"是项目执行阶段而非时间序列，禁止计算降幅，应解读为执行进度。
12. 必须分析数据中的每一个工作表，每个工作表一个板块，禁止省略任何一张；确实无法分析的表也要写一小节说明原因。
13. 每个板块只输出 1 个【图表数据】区块，禁止重复或近似重复的图表。
14. 本任务是单轮一次性任务，必须在一次输出中生成完整报告；禁止向用户提问、禁止"是否继续"式交互、禁止中途停止等待确认。
15. 禁止使用 LaTeX 公式、$...$、引用块（> 符号）等格式，一律使用纯 Markdown 表格与文本。
16. 禁止用省略号（...）、"依此类推"、"其余板块"等占位内容敷衍；每个板块必须完整写出数据概览、关键对比、变化分析、业务解读、策略建议、风险提示六个小节。
17. 板块清单中的行号范围必须来自数据摘要中的真实行号，无法确定时写"—"，禁止编造行号区间。
""")
    prompt_parts.append("")
    prompt_parts.append(f"=== 用户问题 ===\n{question}")
    prompt_parts.append("")
    prompt_parts.append("""=== 需求对照检查（强制执行） ===

在开始分析报告之前，你务必先做以下自查，并在报告末尾列出对照结果：

1. 从用户问题中逐条提取所有分析要求（如：氟化学品市场销售影响、资产规模变化、投融资情况、现金流、成本费用、预算执行、盈利质量、风险识别、发展目标与解决方案等）。
2. 对每条要求，确认：数据中是否有直接数据支撑？如果没有直接数据，应从哪些相关指标推断？
3. 报告输出时，每一条用户要求都必须有对应的分析段落，不允许遗漏。即使数据不完全，也要明确说明"当前数据在XX方面存在局限，基于可用的XX指标进行推断"。
4. 报告末尾附一个对照表：| 用户要求 | 对应报告章节 | 数据支撑情况 |

常见的容易遗漏维度警告（必须覆盖，每个都要独立成板块）：

1. 成本费用分析（强制执行）—— 即便是推断也必须独立成章：
   - 从净利率变动反推成本结构变化（净利率下降说明成本/费用增速>收入增速）
   - 从应付账款激增推断采购成本与账期策略
   - 从存货变动推断生产成本走势
   - 从货币资金/利息收支推断财务费用
   - 明确标注"数据局限：无直接利润表明细，基于XX指标推算"但不能因此跳过
2. 预算执行（如有预算vs实际数据）
3. 现金流分析（经营活动/投资/筹资现金流，或从应收应付存货变动推断）
4. 投融资活动（借款变化、权益变动、资本开支）

=== 分析协议（必须严格遵守） ===

你的分析方式：不是把整份数据混在一起笼统地讲，而是【逐节拆分、逐节深挖】。
把数据当成一本经营报告书，每个业务板块是独立的一章，每一章都要独立、完整、深入地分析。

═══════════════════════════════════════
第一步：拆分数据，划定分析单元
═══════════════════════════════════════

扫描全部数据，识别数据中自然存在的【独立业务板块/独立表格】。一个板块的特征：
- 由空行/标题行/汇总行分隔
- 列名发生变化
- 主题切换（如从"资产"切换到"收入"）
- 数据结构发生变化（如从明细表切换到汇总表）

列出所有识别到的板块，给出每个板块的名称和范围（行号），【不允许合并不同板块】。

数据来自 Excel 的多个工作表（见上述工作表概览）。分析时必须以【工作表】为基本单元，
逐表分析。每个工作表都是独立的数据板块，不允许合并或遗漏。

提示：如果数据包含"生产成本""期间费用""经营预算""资金状况""销售及毛利"等表，
这些表里的数据就是成本费用和现金流的直接数据，必须完整引用，不允许说"无直接数据"。

示例输出格式：
| 序号 | 板块名称 | 行号范围 | 主要内容 |
|------|----------|----------|----------|
| 1    | 资产结构 | 第1-30行 | 货币资金、结构性存款…… |
| 2    | 成本费用分析 | （推断板块） | 从净利率+应付+存货反推成本结构…… |
| 3    | 销售收入 | 第32-80行 | 各产品线收入明细…… |
（有多少板块就列多少，不遗漏任何一个）

═══════════════════════════════════════
第二步：逐板块独立分析（每个板块一个 ## 大标题）
═══════════════════════════════════════

对第一步列出的【每一个板块】，独立输出一节完整分析，不允许合并、不允许一笔带过。
每个板块的分析必须是一个完整、自足的段落，包含以下全部内容：

【板块内必须包含】
1. 数据概览表：该板块涉及哪些指标、多少行、数据完整度
2. 关键指标对比表：期末vs期初/预算vs实际/各分类明细，用表格展示
3. 变化分析：各指标变动率、绝对值变化，标注异常波动（变动 >30% 的标红预警）
4. 业务解读（强制执行）：
   - 这个板块的数据变化反映了什么经营状况？
   - 数据之间的关系说明了什么？（如：费用增长快于收入增长→盈利能力承压）
   - 对整体经营的传导影响是什么？
5. 策略建议（强制执行）：
   - 针对这个板块，具体应该采取什么管理动作？
   - 是否需要调整预算、优化流程、加强管控？
6. 风险提示：该板块存在的数据质量问题（缺失、异常值）和经营风险

═══════════════════════════════════════
第三步：跨板块综合分析
═══════════════════════════════════════

在所有板块独立分析完成后，串起来看全局：

- 板块之间的联动关系：A板块的变化如何影响B板块？
  （如：销售收入增长但应收账款也在增长→可能存在回款周期拉长风险）
- 整体经营画像：这家企业的优势在哪个板块？短板在哪里？
- 资源错配检测：是否有板块投入大产出小？是否有高价值板块投入不足？
- 前瞻预测：基于各板块趋势，预测下阶段的整体经营走势

═══════════════════════════════════════
第四步：总评与行动纲领
═══════════════════════════════════════

- 经营健康度评分（优秀/良好/关注/预警）
- TOP 3 亮点 和 TOP 3 风险
- 5 项最紧迫工作（按优先级排序，每项标明负责板块）

═══════════════════════════════════════
输出格式要求
═══════════════════════════════════════

- 报告标题：## 板块名称
- 数据：表格呈现，**重点数字加粗**
- 每条发现必须有 >2 个具体数值引用
- 不允许出现"某些指标""部分数据"等模糊表述—必须说出具体列名和数值
- 如果数据只有一期，表格只列一期，严禁编造多期对比（编造比缺失更严重）
- 无历史期数据的指标不得计算变动率，只能陈述当期数值
- 表格中的每个数值必须有硬数字出处（指标名+期别完全匹配），找不到出处的留空写"无数据"，禁止挪用其他指标行的数值充当历史期
- 对比表中"期初/年初"值必须来自同一指标标签为"年初"或"2025H2"的行，禁止使用"2025H1"等其他期别的数值
- "基准/冲刺"是两档目标值而非时间序列，禁止计算变动率；预算执行应对比实际值与预算目标
- 图表数据必须与正文一致：只有当期数据的指标只列当期数值，禁止用 0 或占位值填充缺失期别
- 引用硬数字必须保留原始指标名（如净利率），禁止改写成硬数字中未出现的指标名（如毛利率）
- 数据概览表只列核心指标（最多 8 行），禁止逐行或逐行号区间罗列，数据量大时用"共N行"概括
- 整份报告（不含图表数据区块）控制在 6000 字以内，禁止枚举式、重复式输出
- 禁止将同一组数字复制粘贴到多个期别列中
- 每个板块的分析必须独立完整，读者可以只看一个板块而不需要参考其他部分
- 报告末尾必须附上【需求对照表】，格式：| 用户要求 | 对应章节 | 数据支撑情况 |

- 在需求对照表之后，输出一个【图表数据】区块。每个分析模块至少生成一张图表，展示该模块的关键对比数据。数据必须来自上方的硬数字，禁止编造。以 JSON 格式列出：
```chartjson
[
  {"title": "图表标题", "type": "bar/pie/line/bar_h", "data": {"指标1": 数值, "指标2": 数值}}
]
```""")

    prompt = "\n".join(prompt_parts)

    return {
        "filenames": filenames,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "dataframes": dataframes,
        "prompt": prompt,
    }


# ══════════════════════════════════════════════════════════════
# 多节点分布式分析
# 主节点（设置了 DEEPANALYZE_NODES）把各工作表任务轮流分发给加速节点，
# 各节点的 AI 分别完成分项分析，最后由主节点的 AI 生成总览，拼成完整报告。
# ══════════════════════════════════════════════════════════════

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
        "1. 以本工作表为分析单元，逐板块拆分并独立深挖，每个板块一个 ## 标题，不允许合并或一笔带过。",
        "2. 每个板块必须包含：数据概览表、关键指标对比表、变化分析（变动 >30% 标注预警）、业务解读、策略建议、风险提示。",
        "3. 每条发现必须引用至少 2 个具体数值（硬数字优先），禁止“某些指标”“部分数据”等模糊表述。",
        "4. 数据只有一期时只列一期，严禁编造多期对比；无历史期数据的指标不得计算变动率，只能陈述当期数值；"
        "每个数值必须能在硬数字中找到指标名+期别完全匹配的出处，否则留空写'无数据'，禁止挪用其他指标行的数值充当历史期；"
        "禁止将同一组数字复制粘贴到多个期别列。",
        "5. 硬数字必须原样引用，禁止修改或换算。",
        "6. 输出使用 Markdown，标题以 ## 开头，重点数字加粗。",
        "7. 数据概览表必须列出核心指标（至少3行、至多8行，每行含指标名+数值），不得省略为空。",
        "8. 本工作表分析控制在 1500-3000 字，禁止枚举、重复或模板化输出。",
        "9. 变化分析/业务解读/策略建议/风险提示每节各写 1-3 段，禁止每个指标单独一段的逐项式输出。",
        "10. 只有当期数据的表，历史期列必须写'无数据'，禁止把当期值复制到历史期列、禁止从其他指标行取值。",
        "11. 变化分析的对比期必须与上方对比表一致（以'2025H2/年初'列为基准），禁止改用'2025H1'等其他列计算变动率。",
        "12. 历史期为'无数据'的指标只陈述当期数值，禁止写出任何增幅/降幅。",
        "13. '预算金额/合同金额/已付金额'是项目执行阶段而非时间序列，禁止计算降幅，应解读为执行进度。",
        "14. 图表数据必须与正文一致：只有当期数据的指标只列当期数值，禁止用0、重复值或其他指标行数值填充缺失期别。",
        "15. 每个板块只输出 1 个【图表数据】区块，禁止重复或近似重复的图表。",
        "16. 本任务是单轮一次性任务，必须一次输出完整分析；禁止向用户提问、禁止'是否继续'式交互、禁止中途停止等待确认。",
        "17. 禁止使用 LaTeX 公式、$...$、引用块（> 符号）等格式，一律使用纯 Markdown 表格与文本。",
        "18. 禁止用省略号（...）、'依此类推'、'其余板块'等占位内容敷衍；本工作表分析必须完整写出六个小节。",
        "19. 分析正文之后输出【图表数据】区块，每个分析模块至少一张图表，数据必须来自硬数字，禁止编造，格式：",
        "```chartjson",
        '[{"title": "图表标题", "type": "bar/pie/line/bar_h", "data": {"指标1": 数值, "指标2": 数值}}]',
        "```",
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
        "1. 跨板块综合分析：各工作表之间的联动关系、整体经营画像、资源错配检测、前瞻预测。",
        "2. 总评与行动纲领：经营健康度评分（优秀/良好/关注/预警）、TOP 3 亮点、TOP 3 风险、5 项最紧迫工作（按优先级排序，每项标明负责板块）。",
        "3. 需求对照表：| 用户要求 | 对应报告章节 | 数据支撑情况 |，逐条对照用户问题中的每项要求，不允许遗漏。",
        "4. 引用具体数值时必须与分项分析一致，禁止编造；数据无法支撑的维度要明确说明数据局限。",
        "5. 输出使用 Markdown，标题以 ## 开头，重点数字加粗。",
        "6. 末尾输出【图表数据】区块，展示总览关键对比，数据必须来自上述分项分析，禁止编造，格式：",
        "```chartjson",
        '[{"title": "图表标题", "type": "bar/pie/line/bar_h", "data": {"指标1": 数值, "指标2": 数值}}]',
        "```",
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

    return {
        "filenames": filenames,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "dataframes": dataframes,
        "tasks": tasks,
    }


def _distributed_analysis_events(prep, question, output_parts):
    """分布式流式分析：并行分发工作表任务 → 逐个产出小节 → 主节点生成总览。

    yield ("think"|"text", chunk)，并把所有文本块累计进 output_parts 供图表提取。
    """
    nodes = _distributed_nodes()
    tasks = prep["tasks"]
    print(f"[分布式] {len(tasks)} 个工作表任务，分发给 {len(nodes)} 个节点")
    sections = []
    failures = 0
    done = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        fut_map = {}
        for i, t in enumerate(tasks):
            node = nodes[i % len(nodes)]
            fut = ex.submit(_call_node_sheet, node["url"], t["prompt"])
            fut_map[fut] = (t, node)

        for fut in as_completed(fut_map):
            t, node = fut_map[fut]
            try:
                section = fut.result().strip()
            except Exception as e:
                failures += 1
                block = f"\n\n### 工作表「{t['label']}」分析（节点：{node['name']}）\n\n⚠️ 该任务分析失败：{str(e)}\n"
                output_parts.append(block)
                yield ("text", block)
                done += 1
                yield ("progress", {"done": done, "total": total})
                continue
            if not section:
                section = "（节点未返回有效分析内容）"
            sections.append((t["label"], node["name"], section))
            block = f"\n\n### 工作表「{t['label']}」分析（节点：{node['name']}）\n\n{section}\n"
            output_parts.append(block)
            yield ("text", block)
            done += 1
            yield ("progress", {"done": done, "total": total})

    # ── 主节点生成总览 ──
    intro = "\n\n---\n\n## 总览与综合结论（主节点生成）\n\n"
    output_parts.append(intro)
    yield ("text", intro)
    for evt, chunk in _run_inference(_build_overview_prompt(question, sections), max_tokens=16384, stream=True):
        if evt == "think":
            yield ("think", chunk)
        else:
            output_parts.append(chunk)
            yield ("text", chunk)

    if failures:
        note = f"\n\n⚠️ 共有 {failures} 个工作表任务失败，详细原因见对应小节。\n"
        output_parts.append(note)
        yield ("text", note)


def _perform_analysis_distributed(prep, question):
    """非流式分布式分析：并行分发 → 收集 → 主节点总览 → 拼装完整报告。"""
    nodes = _distributed_nodes()
    tasks = prep["tasks"]
    print(f"[分布式] {len(tasks)} 个工作表任务，分发给 {len(nodes)} 个节点")
    sections = []
    failures = 0

    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        fut_map = {}
        for i, t in enumerate(tasks):
            node = nodes[i % len(nodes)]
            fut = ex.submit(_call_node_sheet, node["url"], t["prompt"])
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
    full = "\n".join(body_parts)

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
        "DeepAnalyze 数据分析报告（分布式）",
        "=" * 60,
        "",
        f"分析文件 ({len(prep['filenames'])} 个): {', '.join(prep['filenames'])}",
        f"分析问题: {question}",
        f"数据总量: {prep['total_rows']} 行 × {prep['total_cols']} 列",
        f"节点数: {len(nodes)} 个（{', '.join(n['name'] for n in nodes)}），工作表任务数: {len(tasks)}",
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
        "mode": "distributed",
        "nodes": [n["name"] for n in nodes],
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
    try:
        result = _run_inference(prompt, max_tokens=max_tokens)
        return jsonify({"result": result})
    except Exception as e:
        print(f"[节点任务错误] {e}")
        return jsonify({"error": str(e)}), 500


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

    # 2. 准备分析输入（分布式模式按工作表拆分任务）
    distributed = bool(_distributed_nodes())
    try:
        if distributed:
            prep = _prepare_distributed_input(valid_files, question)
        else:
            prep = _prepare_analysis_input(valid_files, question)
    except Exception as e:
        return jsonify({"error": f"数据准备失败: {str(e)}"}), 500

    def generate():
        model_output_parts = []
        try:
            # 推送模式标识（单节点/分布式），供前端展示徽标
            meta = {"type": "meta", "mode": "distributed" if distributed else "single",
                    "nodes": [n["name"] for n in _distributed_nodes()]}
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            # 推送报告头
            header = "\n".join([
                "=" * 60,
                "DeepAnalyze 数据分析报告",
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

            # 流式推理（分布式：并行分发工作表任务 + 主节点总览）
            if distributed:
                for evt_type, chunk in _distributed_analysis_events(prep, question, model_output_parts):
                    if evt_type == "think":
                        yield f"data: {json.dumps({'type': 'think', 'content': chunk}, ensure_ascii=False)}\n\n"
                    elif evt_type == "progress":
                        yield f"data: {json.dumps({'type': 'progress', 'done': chunk['done'], 'total': chunk['total']}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"
            else:
                for evt_type, chunk in _run_inference(prep["prompt"], stream=True):
                    if evt_type == "think":
                        yield f"data: {json.dumps({'type': 'think', 'content': chunk}, ensure_ascii=False)}\n\n"
                    else:
                        model_output_parts.append(chunk)
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk}, ensure_ascii=False)}\n\n"

            full_output = "".join(model_output_parts)
            if not full_output or not full_output.strip():
                full_output = "（模型未生成有效回复，请重试）"

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

        except Exception as e:
            import traceback
            print(f"[Stream Error] {traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            yield "data: {\"type\": \"done\"}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def perform_analysis(files, question):
    """执行数据分析 —— 读取多个文件 + 调用模型/API 生成综合报告 + 图表。

    Returns:
        {"result": 文本报告, "images": [(标题, base64), ...]}
    """
    # 分布式模式：按工作表分发到加速节点，主节点生成总览
    if _distributed_nodes():
        prep = _prepare_distributed_input(files, question)
        return _perform_analysis_distributed(prep, question)

    prep = _prepare_analysis_input(files, question)
    prompt = prep["prompt"]

    # ── 调用模型推理 ──
    model_output = _run_inference(prompt)

    # ── 生成图表：优先 AI 指定，回退自动 ──
    all_charts = _generate_charts_from_json(model_output)
    if not all_charts:
        for fname, df in prep["dataframes"]:
            prefix = fname if len(prep["dataframes"]) > 1 else ""
            try:
                charts = _generate_charts(df, prefix=prefix)
                all_charts.extend(charts)
            except Exception as e:
                print(f"[图表] {fname} 图表生成失败: {e}")

    # ── 拼装最终报告 ──
    report_lines = [
        "=" * 60,
        "DeepAnalyze 数据分析报告",
        "=" * 60,
        "",
        f"分析文件 ({len(files)} 个): {', '.join(prep['filenames'])}",
        f"分析问题: {question}",
        f"数据总量: {prep['total_rows']} 行 × {prep['total_cols']} 列",
        "",
        "-" * 40,
        "AI 分析正文",
        "-" * 40,
        "",
        model_output.strip(),
        "",
        "-" * 40,
        "报告生成完毕",
        "-" * 40,
    ]

    return {
        "result": "\n".join(report_lines),
        "images": all_charts,
        "mode": "single",
        "nodes": [],
    }


@app.route("/export/docx", methods=["POST"])
def export_docx():
    """将 HTML 报告导出为 Word 文档（调用独立脚本）。"""
    data = request.get_json(force=True)
    if not data or not (data.get("text") or data.get("html")):
        return jsonify({"error": "缺少报告内容"}), 400

    html_content = data.get("html", data.get("text", ""))
    title = data.get("title", "DeepAnalyze 分析报告")

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
                download_name=f'DeepAnalyze_{title[:30]}.docx'
            )
        finally:
            try: os.unlink(html_path)
            except: pass
            try: os.unlink(out_path)
            except: pass

    return jsonify({"error": "Word 导出脚本未找到，请安装 export_docx.py"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("DeepAnalyze - 本地数据分析助手")
    print("=" * 60)
    print()
    _port = int(os.environ.get("DEEPANALYZE_PORT", "5000"))
    if _NODE_LIST:
        print(f" 运行模式: 分布式主节点（加速节点: {_NODE_LIST}）")
        print(" 前端页面: http://localhost:%d（加速节点页面对应端口单独访问）" % _port)
    else:
        print(f" 运行模式: 单节点")
        print(" 前端页面: http://localhost:%d/" % _port)
    print(f" 分析接口: POST http://localhost:{_port}/analyze")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    # 端口可用 DEEPANALYZE_PORT 覆盖（多实例/分布式同机部署时需要）
    app.run(host="0.0.0.0", port=_port, debug=False)
