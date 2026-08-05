"""QuantView - 文件处理模块

负责上传文件的读取与解析（Excel/CSV/PDF → DataFrame）、
数据摘要生成（_df_summary 系列）与硬数字提取（_extract_hard_numbers 系列）。

该模块不依赖 Flask，只依赖 pandas（pdfplumber 为可选依赖）。
"""

import io
import os
import re
import traceback

import pandas as pd

# pdfplumber 为可选依赖，仅在处理 PDF 时需要
try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False
    pdfplumber = None

ALLOWED_EXTENSIONS = {"xlsx", "xls", "csv", "pdf"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file(file):
    """读取单个上传文件为 DataFrame。

    Returns:
        (filename, suffix, df) 三元组
    """
    filename = file.filename or "unknown"
    suffix = os.path.splitext(filename)[1].lower()
    file_bytes = file.read()

    if suffix == ".pdf":
        if not _HAS_PDFPLUMBER:
            raise RuntimeError("PDF 文件需要 pdfplumber 库，请执行: pip install pdfplumber")
        # PDF: 用 pdfplumber 提取所有页面的表格，合并为单个 DataFrame
        all_tables = []
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    for ti, table in enumerate(tables):
                        if not table or len(table) == 0:
                            continue
                        header = [str(h or f"列{i}") for i, h in enumerate(table[0])]
                        # 去重列名：同名列加后缀 _1, _2 ...
                        seen = {}
                        unique_header = []
                        for h in header:
                            if h in seen:
                                seen[h] += 1
                                unique_header.append(f"{h}_{seen[h]}")
                            else:
                                seen[h] = 0
                                unique_header.append(h)

                        data = table[1:] if len(table) > 1 else []
                        df_t = pd.DataFrame(data, columns=unique_header)
                        df_t.insert(0, "_pdf_page", page_num)
                        df_t.insert(1, "_pdf_table", ti + 1)
                        all_tables.append(df_t)

            if not all_tables:
                raise RuntimeError("PDF 中未检测到表格数据，请确认文件包含可提取的表格")

            df = pd.concat(all_tables, ignore_index=True, sort=False)
            print(f"[PDF] 从 {filename} 提取了 {len(all_tables)} 个表格，共 {len(df)} 行")
        except Exception as e:
            raise RuntimeError(f"PDF 解析失败: {str(e)}")
    else:
        buf = io.BytesIO(file_bytes)
        try:
            if suffix == ".csv":
                df = pd.read_csv(buf)
            else:
                # Excel: 读取所有工作表，合并为单个 DataFrame
                xls = pd.ExcelFile(buf)
                sheet_dfs = []
                for sheet_name in xls.sheet_names:
                    sdf = pd.read_excel(xls, sheet_name=sheet_name)
                    # 添加工作表名列
                    sdf.insert(0, "_sheet", sheet_name)
                    sheet_dfs.append(sdf)
                df = pd.concat(sheet_dfs, ignore_index=True, sort=False)
                sheet_count = len(sheet_dfs)
                total_rows = len(df)
                print(f"[Excel] 读取了 {sheet_count} 个工作表，合并后共 {total_rows} 行")
        finally:
            buf.close()

    return filename, suffix, df


def df_summary(filename, df):
    """生成单个 DataFrame 的增强数据摘要，让模型看到数据的全貌。"""
    try:
        return _df_summary_impl(filename, df)
    except Exception:
        print(f"[_df_summary ERROR] {filename}: {traceback.format_exc()}")
        raise


def _df_summary_impl(filename, df):
    num_rows, num_cols = df.shape
    columns_str = ", ".join(df.columns.astype(str))

    # ── 工作表概览（多 sheet Excel 时关键）──
    sheet_overview = ""
    if "_sheet" in df.columns:
        sheet_counts = df["_sheet"].value_counts()
        sheet_lines = ["数据来自多个工作表："]
        for sname, scount in sheet_counts.items():
            sheet_lines.append(f"  [{sname}]: {scount} 行")
        sheet_overview = "\n".join(sheet_lines)

    # ── 列级概览：类型、非空数、唯一值数 ──
    col_overview_lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = int(df[col].notna().sum())
        nunique = int(df[col].nunique())
        col_overview_lines.append(
            f"  {col}: dtype={dtype}, 非空={non_null}/{num_rows}, 唯一值数={nunique}"
        )
    col_overview_str = "\n".join(col_overview_lines)

    # ── 分 Sheet 采样 ──
    head_rows = []
    tail_rows = []

    per_sheet_head = 5
    per_sheet_tail = 2

    if "_sheet" in df.columns:
        for sname in df["_sheet"].unique():
            sdf = df[df["_sheet"] == sname]
            if len(sdf) == 0:
                continue
            try:
                if per_sheet_head > 0:
                    h = sdf.head(per_sheet_head).to_string(index=True)
                    head_rows.append(f"--- [{sname}] ---\n{h}")
            except Exception:
                pass
            try:
                if per_sheet_tail > 0:
                    t = sdf.tail(per_sheet_tail).to_string(index=True)
                    tail_rows.append(f"--- [{sname}] 尾部 ---\n{t}")
            except Exception:
                pass
    else:
        try:
            head_rows.append(df.head(30).to_string(index=True))
        except Exception:
            head_rows.append("(无法生成预览)")
        try:
            tail_rows.append(df.tail(10).to_string(index=True))
        except Exception:
            tail_rows.append("(无法生成尾部预览)")

    head_str = "\n\n".join(head_rows) if head_rows else "（已省略行级数据，见下方统计）"
    tail_str = "\n\n".join(tail_rows) if tail_rows else ""

    # 中间抽样（每隔 num_rows/15 取一行，最多 20 行）
    try:
        step = max(1, num_rows // 15)
        mid_indices = list(range(0, num_rows, step))[:20]
        mid_sample = df.iloc[mid_indices].to_string(index=True)
    except Exception:
        mid_sample = "(无法生成中间抽样)"

    # ── 文本列的值分布 + 行号范围（确保模型看到完整数据维度）──
    text_dimensions = {}
    text_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in text_cols:
        val_counts = df[col].value_counts()
        n_unique = len(val_counts)
        lines = []

        if 2 <= n_unique <= 80:
            # 列出所有唯一值、频次及行号范围
            for val in val_counts.index:
                indices = df.index[df[col] == val].tolist()
                row_range = f"第{indices[0]}-{indices[-1]}行" if len(indices) > 1 else f"第{indices[0]}行"
                lines.append(f"  '{val}': {len(indices)}次, {row_range}")
            text_dimensions[col] = f"共{n_unique}个唯一值:\n" + "\n".join(lines)
        elif n_unique > 80:
            text_dimensions[col] = f"（共{n_unique}个唯一值，按频次Top 20）:\n" + val_counts.head(20).to_string()
        else:
            text_dimensions[col] = f"唯一值: {list(val_counts.index)}"

    dims_lines = []
    for col, info in text_dimensions.items():
        dims_lines.append(f"【{col}】\n{info}")
    dimensions_str = "\n\n".join(dims_lines) if dims_lines else "（无文本分类列）"

    # ── 每个工作表独立统计（分 sheet 计算，避免混合失真）──
    per_sheet_summary = ""
    stats_str = ""
    if "_sheet" in df.columns:
        sheets = df["_sheet"].unique()
        sheet_summaries = []

        for sname in sheets:
            sdf = df[df["_sheet"] == sname]
            lines = [f"【{sname}】({len(sdf)}行)"]

            # 文本列
            for col in sdf.columns:
                if col == "_sheet":
                    continue
                vals = sdf[col].dropna()
                if len(vals) > 0 and vals.dtype == "object":
                    unique_count = vals.nunique()
                    lines.append(f"  标签列[{col}]: {unique_count}个唯一值")
                    break

            # 数值列：describe + sums
            for col in sdf.select_dtypes(include=["number"]).columns:
                if sdf[col].notna().sum() > 0:
                    try:
                        # 强制转数值，跳过非数字行
                        numeric_vals = pd.to_numeric(sdf[col], errors="coerce")
                        if numeric_vals.notna().sum() > 0:
                            lines.append(f"  [{col}]: 有效值={int(numeric_vals.notna().sum())}, 合计={numeric_vals.sum():.2f}, 均值={numeric_vals.mean():.2f}")
                    except Exception:
                        pass
            sheet_summaries.append("\n".join(lines))

        per_sheet_summary = "\n\n".join(sheet_summaries)
        stats_str = per_sheet_summary
    else:
        # 无 _sheet 列时用全局统计
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            stats_str = df[numeric_cols].describe().to_string()
            sums = df[numeric_cols].sum()
            stats_str = stats_str + "\n合计:\n" + sums.to_string()
        else:
            stats_str = "（无数值列）"

    # ── 缺失值 ──
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        missing_pct = (missing_cols / num_rows * 100).round(1)
        missing_str = "\n".join([f"  {col}: {cnt} ({pct}%)" for col, cnt, pct in
                                  zip(missing_cols.index, missing_cols.values, missing_pct)])
    else:
        missing_str = "无缺失值"

    return {
        "filename": filename,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns_str": columns_str,
        "sheet_overview": sheet_overview,
        "col_overview_str": col_overview_str,
        "per_sheet_summary": per_sheet_summary,
        "head_str": head_str,
        "mid_sample_str": mid_sample,
        "tail_str": tail_str,
        "dimensions_str": dimensions_str,
        "stats_str": stats_str,
        "missing_str": missing_str,
    }


def extract_hard_numbers_from_bytes(file_buffers):
    """从 Excel 文件字节中提取关键数值。"""
    hard_nums = []
    for _, data in file_buffers:
        try:
            hard_nums.extend(extract_hard_numbers_core(io.BytesIO(data)))
        except Exception:
            pass
    return hard_nums


def extract_hard_numbers_by_sheet(stream):
    """按工作表拆分硬数字，返回 {工作表名: 硬数字文本}。

    复用 extract_hard_numbers_core 的输出（每段以【工作表名】开头），
    按行解析分组。供分布式逐表分析使用。
    """
    result = {}
    current = None
    parts = []
    for line in "\n".join(extract_hard_numbers_core(stream)).splitlines():
        m = re.match(r"^【(.+)】$", line.strip())
        if m:
            if current is not None and parts:
                result[current] = "\n".join(parts)
            current = m.group(1)
            parts = [line]
        elif current is not None:
            parts.append(line)
    if current is not None and parts:
        result[current] = "\n".join(parts)
    return result


def extract_hard_numbers_core(stream):
    hard_nums = []
    _sum_kw = ["合计", "总计", "总负债", "总资产", "所有者权益", "负债合计", "权益合计", "净资产",
               "负债和所有者", "资产总计", "负债总计", "所有者权益合计",
               "流动资产合计", "非流动资产合计", "流动负债合计", "非流动负债合计",
               "经营活动", "投资活动", "筹资活动",
               "投资活动净流量", "筹资活动净流量", "资金净流量",
               "企业所得税", "增值税", "免抵调库", "实缴税收",
               "资金余额", "期末现金余额", "期末银行汇票余额",
               "内贸回款", "外贸回款", "采购付款",
               "销售额", "净利润", "毛利",
               "净资产收益率", "销售净利率", "总资产周转率", "权益乘数"]
    def _norm_period(raw):
        """标准化期别标签，正则匹配常见格式。"""
        import re
        # 预算类必须优先于"上半年/下半年"（否则"2026下半年预算"会被截成 2026H2 丢失预算含义）
        m = re.match(r'(\d{4})(?:年)?(?:下|上)半年预算', raw)
        if m: return f'{m.group(1)}H{"2" if "下" in raw else "1"}预'
        m = re.match(r'(\d{4})(?:年)?全年(?:预算)?', raw)
        if m: return f'{m.group(1)}全年预'
        # 上半年/下半年
        m = re.match(r'(\d{4})上半年', raw)
        if m: return f'{m.group(1)}H1'
        m = re.match(r'(\d{4})下半年', raw)
        if m: return f'{m.group(1)}H2'
        # Q1-Q4
        m = re.match(r'(\d{4})年Q([1-4])', raw)
        if m: return f'{m.group(1)}Q{m.group(2)}'
        m = re.match(r'(\d{4})年(\d{1,2})月', raw)
        if m: return f'{m.group(1)}M{m.group(2)}'
        # 英文 Q1 2024
        m = re.match(r'Q([1-4])\s*(\d{4})', raw)
        if m: return f'{m.group(2)}Q{m.group(1)}'
        # 半年度实际 / 年度实际
        m = re.match(r'(\d{4})年(?:半)?年度实际', raw)
        if m: return f'{m.group(1)}实'
        # 7-9月预算 / 10-12月预算
        m = re.match(r'(\d{4})年(\d{1,2})-(\d{1,2})月预算', raw)
        if m: return f'{m.group(2)}-{m.group(3)}月预'
        # 期末 / 年初 variants
        if '期末' in raw: return '期末'
        if '年初' in raw: return '年初'
        if '期初' in raw: return '期初'
        # 预算/合同/已付
        if raw in ('基准', '冲刺', '预算金额', '合同金额', '已付金额', '预算', '合同', '已付'):
            return raw
        # 预算目标类
        if '预算' in raw: return '预算'
        if '实际' in raw: return '实际'
        # 兜底：截断过长标签
        return raw[:8] if len(raw) > 8 else raw
    _skip_hdrs = {"环比", "同比", "变动", "相比年初变动", "相比年初", "变动率",
                  "合同执行率", "合同付款率", "费用率", "税负率", "占比",
                  "资产结构", "项目名称", "项目", "指标", "科目", "税种", "名称", "类别",
                  "负债和所有", "负债和所有者权益结构", "单位", "产品", "物料", "原材料"}
    try:
        xls = pd.ExcelFile(stream)
        for sname in xls.sheet_names:
                try:
                    sdf = pd.read_excel(xls, sheet_name=sname)
                except Exception:
                    continue
                lines = []
                col_periods = []
                # 账龄表有 8 个账龄段列 + 合计列（共 10 列数据），其余表最多读 7 列
                _max_col = 11 if "账龄" in sname else 8
                # 首行"单位：X"会被 pandas 吃成列名，需从列名里取初始单位
                cur_unit = ""
                for _cu in list(sdf.columns)[:5]:
                    if isinstance(_cu, str) and _cu.strip().startswith("单位："):
                        _u = _cu.strip().replace("单位：", "").strip()
                        if _u and len(_u) <= 10:
                            cur_unit = _u
                        break
                for _hri in range(min(3, len(sdf))):
                    hdr_parts = []
                    for c in list(sdf.columns)[1:_max_col]:
                        v = sdf.iloc[_hri][c]
                        if pd.notna(v):
                            vs = str(v).strip()
                            if vs and 2 <= len(vs) <= 12:
                                if any(sk in vs for sk in _skip_hdrs):
                                    hdr_parts.append("__SKIP__")
                                else:
                                    hdr_parts.append(_norm_period(vs))
                            else:
                                hdr_parts.append(None)
                        else:
                            hdr_parts.append(None)
                    valid = [h for h in hdr_parts if h is not None and h != "__SKIP__"]
                    # 拒绝含数字的行（数据值被误认为表头）
                    if len(valid) >= 2 and not any(h.replace("-","").replace(".","").isdigit() for h in valid):
                        col_periods = hdr_parts
                        last_pd = None
                        for _i in range(len(col_periods)):
                            if col_periods[_i] is not None and col_periods[_i] != "__SKIP__":
                                last_pd = col_periods[_i]
                            elif col_periods[_i] is None and last_pd is not None:
                                col_periods[_i] = last_pd
                        break
                # 兜底：如未从数据行找到表头，用列名作为期别标签
                if not col_periods:
                    hdr_parts = []
                    for c in list(sdf.columns)[1:_max_col]:
                        vs = str(c).strip()
                        if vs and 2 <= len(vs) <= 12 and not vs.startswith("Unnamed"):
                            if any(sk in vs for sk in _skip_hdrs):
                                hdr_parts.append("__SKIP__")
                            else:
                                hdr_parts.append(_norm_period(vs))
                        else:
                            hdr_parts.append(None)
                    valid = [h for h in hdr_parts if h is not None and h != "__SKIP__"]
                    if len(valid) >= 2:
                        col_periods = hdr_parts
                        last_pd = None
                        for _i in range(len(col_periods)):
                            if col_periods[_i] is not None and col_periods[_i] != "__SKIP__":
                                last_pd = col_periods[_i]
                            elif col_periods[_i] is None and last_pd is not None:
                                col_periods[_i] = last_pd
                # 转置表布局检测：行=期别、列=指标名（如应收账龄表：行 2026上半年/2025下半年，
                # 列 1个月内/1-2个月…；周转天数表：列 锂电/氟化学/合并）。当检测到的列期别不是
                # 期别标签、而行标签为期别时，按转置方式提取（列头当指标名、行标签当期别），
                # 否则模型会把账龄段/板块当"期别"、把期别当"指标名"，导致标签错位
                _is_pd = lambda s: bool(re.match(r"^\d{4}(?:上|下)半年$", s)) or s in ("同比", "环比")
                _transposed = False
                _col_headers = [None] * (_max_col - 1)
                if col_periods and not any(_is_pd(p) for p in col_periods if p):
                    for _ri in range(min(3, len(sdf))):
                        _lb = ""
                        for _c in sdf.columns[:5]:
                            _v = sdf.iloc[_ri][_c]
                            if pd.notna(_v) and isinstance(_v, str) and _v.strip():
                                _lb = _v.strip()
                                break
                        if _is_pd(_lb):
                            _transposed = True
                            break
                    # 捕获首个表头行的原始列名（转置模式用列头当指标名）
                    for _ri in range(min(3, len(sdf))):
                        _hd = []
                        for _c in list(sdf.columns)[1:_max_col]:
                            _v = sdf.iloc[_ri][_c]
                            if pd.notna(_v) and isinstance(_v, str) and _v.strip() \
                                    and not _v.strip().replace(".", "").replace("-", "").isdigit():
                                _hd.append(_v.strip())
                            else:
                                _hd.append(None)
                        if sum(1 for h in _hd if h) >= 2:
                            _col_headers = _hd
                            break
                seen = set()
                last_section = ""
                section_context = ""  # 段落标题（用于合计行上下文）
                # 跳过标签：非数据行（表头、单位、纯栏目名等）
                _skip_labels = {"单位：万元", "单位：元", "单位", "项目", "指标", "科目",
                               "资产结构", "负债和所有者权益结构", "杜邦分析",
                               "产品销售", "销售额变动", "毛利额变动", "产量", "成本构成",
                               "销售费用", "管理费用", "研发费用", "财务费用",
                               "人力资源", "预算目标", "实缴税收", "税金及附加",
                               "项目名称", "运输类型", "投资类别", "存货类别",
                               "账龄区间", "费用项目", "人力项目", "资金项目",
                               "成本项", "投资指标", "账龄", "物料/合计",
                               "产品线", "业务线", "税种", "物料",
                               "指标名称", "序号", "行号",
                               "2026下半年", "2026全年", "上半年", "下半年",
                               "期初", "内陆运输", "出口运输", "返桶运输",
                               "运输费量", "运输费", "运输单价", "里程",
                               "合同签订率", "合同执行率", "合同付款率",
                               "投资合计",
                               "*****中心建设项目", "项目合计", "总计",
                               "资金收益率", "理财产品", "政府补助专项资金"}
                # 通用标签：跨板块重复出现、需要带板块前缀区分的指标名
                # （固定资产投资两个项目块、期间费用销售/管理两大块、人力预算块等）
                _generic = {"合计", "总计", "氟化学", "锂电", "原材料", "库存商品",
                            "半成品", "产成品", "发出商品及其他", "其他",
                            "建筑工程费", "设备购置费", "安装费", "材料费", "主要材料费",
                            "人工费用", "办公费用", "招待费用", "差旅费用", "咨询费用",
                            "折旧费用", "车辆费用", "租赁费用", "摊销费用", "包装费用",
                            "运输费用", "产品认证费",
                            "工资", "年终奖金", "股权激励", "社会保险费", "住房公积金",
                            "福利费", "工会经费", "职工教育经费", "其他费用",
                            # 生产成本表：氟化学/锂电两套同名成本明细，必须带板块前缀区分
                            #（否则锂电的直接材料 73000 会被同名去重丢弃，模型只能引用氟化学的 23300）
                            "直接材料", "直接人工", "制造费用", "直接折旧", "燃料动力", "直接包材"}
                for _ri in range(min(len(sdf), 80)):
                    row = sdf.iloc[_ri]
                    label = ""
                    for c in sdf.columns[:5]:
                        v = row[c]
                        if pd.notna(v) and isinstance(v, str) and v.strip():
                            label = v.strip()
                            break
                    if not label:
                        continue
                    # 行内单位单元格："单位：X"可能出现在任意列（块标题行携带或独立行），
                    # 用于区分同名指标的量（吨）/金额（万元）/单价（万元/吨）
                    for _c in list(sdf.columns)[:_max_col]:
                        _v = row[_c]
                        if isinstance(_v, str) and _v.strip().startswith("单位："):
                            _u = _v.strip().replace("单位：", "").strip()
                            if _u and len(_u) <= 10:
                                cur_unit = _u
                            break
                    # ── 转置表（行=期别、列=指标名）提取 ──
                    if _transposed:
                        # 表头行（无数值、≥2 个文本列）→ 更新列名（同一张表内各数据块表头不同）
                        _has_num = False
                        _txts = []
                        for _c in list(sdf.columns)[1:_max_col]:
                            _v = row[_c]
                            if pd.notna(_v):
                                _vs = str(_v).strip()
                                if not _vs:
                                    continue
                                try:
                                    float(_vs)
                                    _has_num = True
                                except (ValueError, TypeError):
                                    if 2 <= len(_vs) <= 14:
                                        _txts.append((_c, _vs))
                        if not _has_num and len(_txts) >= 2:
                            _col_headers = [None] * (_max_col - 1)
                            _ci = 0
                            for _c in list(sdf.columns)[1:_max_col]:
                                _v = row[_c]
                                if pd.notna(_v) and isinstance(_v, str) and _v.strip() and 2 <= len(_v.strip()) <= 14:
                                    _col_headers[_ci] = _v.strip()
                                _ci += 1
                            continue
                        if _is_pd(label):
                            # 行标签=期别（2026上半年/2025下半年/同比/环比），列头=指标名
                            for _ci, _c in enumerate(list(sdf.columns)[1:_max_col]):
                                if _ci >= len(_col_headers) or not _col_headers[_ci]:
                                    continue
                                _v = row[_c]
                                if pd.notna(_v):
                                    try:
                                        _nv = float(_v)
                                    except (ValueError, TypeError):
                                        continue
                                    _hk = f"{cur_unit}·{_col_headers[_ci]}·{label}"
                                    if _hk in seen:
                                        continue
                                    seen.add(_hk)
                                    if abs(_nv) < 0.01 and _nv != 0:
                                        continue
                                    if _nv == int(_nv):
                                        _vs = str(int(_nv))
                                    else:
                                        _vs = f"{_nv:.4f}".rstrip("0").rstrip(".")
                                    _u_sfx = ""
                                    if cur_unit and cur_unit not in ("万元", "元"):
                                        _u_sfx = f"·{cur_unit}"
                                    lines.append(f"  {_col_headers[_ci]}{_u_sfx}({label}={_vs})")
                            continue
                        # 非期别行（块标题/单位行）交给常规逻辑处理
                    # 表头行重检测：整表共用一份 col_periods 会误标中途的新表头块
                    #（如税收表"税种|2026下半年预算|2025下半年实际|同比"、固投第二个项目块），
                    # 无数值且含 ≥2 个期别词的文本行视为新表头，更新列期别
                    _row_has_num = False
                    _hdrs = []
                    for _c in list(sdf.columns)[1:_max_col]:
                        _v = row[_c]
                        if pd.notna(_v):
                            _vs = str(_v).strip()
                            if not _vs:
                                continue
                            try:
                                float(_vs)
                                _row_has_num = True
                            except (ValueError, TypeError):
                                if 2 <= len(_vs) <= 14 and not _vs.replace(".", "").replace("-", "").isdigit():
                                    if any(k in _vs for k in ("预算", "实际", "上半年", "下半年", "年度", "同比", "环比", "期末", "年初", "期初", "月", "合计", "总计")):
                                        _hdrs.append(_norm_period(_vs))
                    if not _row_has_num and len(_hdrs) >= 2:
                        col_periods = [None] * (_max_col - 1)
                        _ci = 0
                        for _c in list(sdf.columns)[1:_max_col]:
                            _v = row[_c]
                            if pd.notna(_v):
                                _vs = str(_v).strip()
                                if _vs and 2 <= len(_vs) <= 14 and not _vs.replace(".", "").replace("-", "").isdigit():
                                    if any(k in _vs for k in ("预算", "实际", "上半年", "下半年", "年度", "同比", "环比", "期末", "年初", "期初", "月", "合计", "总计")):
                                        if _vs in ("同比", "环比"):
                                            col_periods[_ci] = "__SKIP__"
                                        else:
                                            col_periods[_ci] = _norm_period(_vs)
                            _ci += 1
                        _last = None
                        for _i in range(len(col_periods)):
                            if col_periods[_i] is not None and col_periods[_i] != "__SKIP__":
                                _last = col_periods[_i]
                            elif col_periods[_i] is None and _last is not None:
                                col_periods[_i] = _last
                    # 跳过非数据标签（记录为段落标题）；"单位："开头的是纯单位注释行
                    #（如"单位：万元/天/吨"），不是指标，禁止提取成硬数字，也不更新段落上下文；
                    # 但单位本身要记录（cur_unit），用于区分同名指标的量（吨）/金额（万元）/单价
                    if label in _skip_labels or label.startswith("单位："):
                        if label.startswith("单位："):
                            _u = label.replace("单位：", "").strip()
                            if _u and len(_u) <= 10:
                                cur_unit = _u
                        else:
                            section_context = label
                            last_section = label
                        continue
                    # 去重：通用标签（氟化学/原材料等）按"段落·标签"去重，同名指标出现在
                    # 不同板块时全部保留（如 氟化学库存·原材料 与 锂电库存·原材料）；
                    # 其余标签按"单位·标签"去重，使同名指标在不同单位块（吨/万元/单价）下
                    # 全部保留（如 物流 产品运输 的 51100吨 与 900万元）
                    _period_like = bool(re.match(r"^\d{4}(?:上|下)半年$", label)) or label in ("同比", "环比")
                    if label in _generic or _period_like:
                        dedup_key = f"{last_section}·{label}"
                    else:
                        dedup_key = f"{cur_unit}·{label}"
                    if dedup_key in seen:
                        last_section = label
                        continue
                    # 提取数值
                    val_pairs = []
                    seen_periods = set()
                    for ci, c in enumerate(list(sdf.columns)[1:_max_col]):
                        if ci < len(col_periods):
                            pd_label = col_periods[ci]
                            if pd_label is None or pd_label == "__SKIP__":
                                continue
                        else:
                            continue
                        v = row[c]
                        if pd.notna(v):
                            try:
                                nv = float(v)
                                if abs(nv) < 0.01 and nv != 0 and "率" not in label:
                                    continue
                                if nv == int(nv):
                                    vs = str(int(nv))
                                elif 0 < abs(nv) < 1 and "率" in label:
                                    vs = f"{nv*100:.2f}%"
                                else:
                                    vs = f"{nv:.4f}".rstrip("0").rstrip(".")
                                    if "." in vs and len(vs.split(".")[1]) > 3:
                                        vs = f"{nv:.3f}"
                                if pd_label not in seen_periods:
                                    val_pairs.append(f"{pd_label}={vs}")
                                    seen_periods.add(pd_label)
                            except (ValueError, TypeError):
                                pass
                    # 块标题行（如"氟化学内销账龄分布"）可能带 1 个杂散数值（残留比例/合计），
                    # 视为段落标题而非指标：更新 section_context 并跳过，防止污染数据
                    _title_kw = ("分布", "对比", "结构", "明细", "清单", "汇总")
                    if (not val_pairs) or (len(val_pairs) == 1 and any(k in label for k in _title_kw)):
                        # 无数值的标签行视为段落标题（如"净利润变动""销售类型"），
                        # 同时更新 section_context——否则上下文会卡在上一个跳过标记
                        # （如"单位：万元"），导致后续所有合计行共用错误前缀、无法区分
                        # 长文本/提示语（备注、警示）不作为段落上下文，防止污染前缀
                        if len(label) <= 12 and not any(ch in label for ch in "⚠，。！？："):
                            section_context = label
                            last_section = label
                        continue
                    seen.add(dedup_key)
                    # 通用标签带板块前缀（如 销售额变动·氟化学），避免同名指标歧义；
                    # 资金状况等表的板块行（期初/投资活动等）不是指标名，只对通用标签生效，
                    # 其余标签保持裸标签，防止"期初·经营活动"这类污染
                    ctx = ""
                    if (label in _generic and section_context
                            and section_context != label
                            and section_context not in ("合计", "总计")):
                        ctx = f"{section_context}·"
                    display_label = label
                    if label == "资金余额":
                        display_label = "期末资金余额"
                    # 预算类标签加前缀防止和实际值混淆
                    if any(p in ("基准", "冲刺") for p in col_periods if p):
                        if label in ("销售额", "净利润", "毛利", "销售收入"):
                            display_label = f"预算{label}"
                    # 非万元单位（吨/天/万元吨等）附加单位后缀，防止量/金额/单价混淆
                    unit_suffix = ""
                    if cur_unit and cur_unit not in ("万元", "元"):
                        unit_suffix = f"·{cur_unit}"
                    full_label = f"{ctx}{display_label}{unit_suffix}"
                    is_multiperiod = any(
                        p and (p[0].isdigit() or p in ("期末", "年初", "期初"))
                        for p in col_periods
                    ) if col_periods else False
                    # 仅税收类 sheet 限制多期（防模型混淆税种列），其余保留全部期别
                    is_tax_sheet = "税" in sname
                    if is_multiperiod and is_tax_sheet:
                        if val_pairs:
                            lines.append(f"  {full_label}({val_pairs[0]})")
                    else:
                        for vp in val_pairs[:7]:
                            lines.append(f"  {full_label}({vp})")
                # 合计/比率/内外销/板块明细行优先保留，其余按序截断（每 sheet 最多 80 行）
                if len(lines) > 160:
                    priority = [l for l in lines if any(k in l for k in
                                ("合计", "总计", "收益率", "周转率", "乘数", "净利率", "费用率",
                                 "税负率", "内销", "外销", "氟化学", "锂电", "账龄",
                                 "预算", "冲刺", "金额", "单价", "量·"))]
                    other = [l for l in lines if l not in priority]
                    lines = priority + other[:(160 - len(priority))]
                if len(lines) > 1:
                    hdr = f"【{sname}】"
                    # 资金状况：强调期末值
                    if "资金" in sname and any("期末资金余额" in l for l in lines):
                        hdr += "\n  ⚠️ 期末资金余额为当前报告期期末值，不要使用期初值"
                    hard_nums.append(hdr + "\n" + "\n".join(lines))
    except Exception as e:
        print(f"[HardNums] Error in sheet: {e}")
    return hard_nums
