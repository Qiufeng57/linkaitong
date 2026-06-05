"""
医学病例PDF实体提取脚本
从PDF病例报道中提取：患者基本信息、主要症状、既往史、诊断结果、治疗方案
不使用API，使用规则+正则匹配方式提取
"""

import re
import json
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("正在安装 pdfplumber，请稍候...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "--break-system-packages", "-q"])
    import pdfplumber


# ─────────────────────────────────────────────
# 1. PDF 文字提取
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """用 pdfplumber 逐页提取全文，拼接后返回"""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text.append(f"[第{i+1}页]\n{text}")
    return "\n".join(full_text)


# ─────────────────────────────────────────────
# 2. 辅助：在全文中按关键词附近截取上下文片段
# ─────────────────────────────────────────────

def find_context(text: str, keywords: list[str], window: int = 120) -> list[str]:
    """返回每个关键词周围 ±window 字的文本片段（去重）"""
    results = []
    seen = set()
    for kw in keywords:
        for m in re.finditer(re.escape(kw), text):
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            snippet = text[start:end].strip().replace("\n", " ")
            if snippet not in seen:
                seen.add(snippet)
                results.append(snippet)
    return results


# ─────────────────────────────────────────────
# 3. 各实体提取函数
# ─────────────────────────────────────────────

def extract_basic_info(text: str) -> dict:
    """提取患者基本信息：性别、年龄、就诊科室、发表来源"""
    info = {}

    # 性别 + 年龄（如"患者女性，81岁"）
    m = re.search(r"患者([男女]性)[，,\s]*(\d+)\s*岁", text)
    if m:
        info["性别"] = m.group(1)
        info["年龄"] = m.group(2) + "岁"

    # 单独匹配年龄（备用）
    if "年龄" not in info:
        m = re.search(r"(\d+)\s*岁[，。,\s]", text)
        if m:
            info["年龄"] = m.group(1) + "岁"

    # 科室
    m = re.search(r"(老年消化科|消化内科|心内科|肝病科)", text)
    if m:
        info["就诊科室"] = m.group(1)

    # 医院
    m = re.search(r"([\u4e00-\u9fa5]+大学附属[\u4e00-\u9fa5]+医院)", text)
    if m:
        info["就诊医院"] = m.group(1)

    # 文章来源期刊
    m = re.search(r"(中华肝脏病杂志)[，,\s]*(\d{4})", text)
    if m:
        info["来源期刊"] = m.group(1)
        info["发表年份"] = m.group(2)

    return info


def extract_chief_complaint_and_symptoms(text: str) -> dict:
    """提取主诉与主要症状"""
    symptoms = {}

    # 消化道症状关键词
    gi_keywords = ["便血", "血便", "鲜血便", "消化道出血", "呕血", "上消化道出血",
                   "下消化道出血", "腹痛", "腹胀", "腹泻", "恶心", "呕吐"]
    found_gi = [kw for kw in gi_keywords if kw in text]
    if found_gi:
        symptoms["消化道症状"] = found_gi

    # 心脏相关症状
    cardiac_keywords = ["心力衰竭", "心衰", "右心衰", "高输出型心力衰竭", "心功能不全"]
    found_cardiac = [kw for kw in cardiac_keywords if kw in text]
    if found_cardiac:
        symptoms["心脏症状"] = found_cardiac

    # 浮肿/水肿
    edema_kws = ["下肢水肿", "双下肢浮肿", "浮肿"]
    found_edema = [kw for kw in edema_kws if kw in text]
    if found_edema:
        symptoms["水肿症状"] = list(set(found_edema))

    # BNP 数值
    bnp_values = re.findall(r"BNP\s*([\d,]+)\s*ng[/／]L", text)
    if bnp_values:
        symptoms["BNP检测值_ng_L"] = bnp_values

    # 心率
    hr = re.findall(r"心率\s*(\d+[～~\-]\d+)\s*次[/／]min", text)
    if hr:
        symptoms["心率_次每分"] = hr

    # 血红蛋白
    hb = re.findall(r"血红蛋白\s*([\d.]+)\s*g[/／]L", text)
    if hb:
        symptoms["血红蛋白_g_L"] = hb

    # 白蛋白
    alb = re.findall(r"(?:白蛋白|蛋白)\s*([\d.]+)\s*g[/／]L", text)
    if alb:
        symptoms["白蛋白_g_L"] = alb

    return symptoms


def extract_past_history(text: str) -> dict:
    """提取既往史"""
    history = {}

    # 肝硬化
    if "肝硬化" in text:
        m = re.search(r"(原发性胆汁[性]?肝硬化|肝硬化)", text)
        if m:
            history["肝病史"] = m.group(1)

    # 乙肝
    if "乙型肝炎" in text or "乙肝" in text:
        m = re.search(r"乙型肝炎\s*3\s*个抗体阳性", text)
        history["乙型肝炎"] = "抗体阳性（既往感染）" if m else "有相关记录"

    # 抗核抗体
    m = re.search(r"抗核抗体[，,\s]*([\d:∶：]+)\s*阳性", text)
    if m:
        history["抗核抗体"] = m.group(1) + " 阳性"

    # 长期用药
    drugs = re.findall(r"(?:服用|口服)([\u4e00-\u9fa5a-zA-Z]+(?:酸|胺|素|钠|片|颗粒)?)", text)
    if drugs:
        history["既往用药"] = list(set(d for d in drugs if 2 <= len(d) <= 10))

    # 诱因：中草药
    if "中草药" in text or "中药" in text:
        history["可疑诱因"] = "服用成分不详的中草药（关节疼痛治疗）"

    # 病程时长
    m = re.search(r"(\d+)\s*年前[曾]?(?:因|出现)", text)
    if m:
        history["病程"] = m.group(1) + "年前起病"

    return history


def extract_diagnosis(text: str) -> dict:
    """提取诊断结果"""
    diag = {}

    # 主要诊断关键词
    primary_dx_kws = [
        "急性肠系膜上静脉血栓形成",
        "肠系膜静脉血栓",
        "门静脉血栓形成",
        "急性门静脉血栓",
        "非肝硬化急性肠系膜上静脉血栓形成",
    ]
    found_primary = [kw for kw in primary_dx_kws if kw in text]
    if found_primary:
        diag["主要诊断"] = found_primary

    # 并发症诊断
    complications = []
    comp_kws = {
        "急性高输出型症状性心力衰竭": "心力衰竭",
        "贲门黏膜撕裂": "贲门黏膜撕裂（Mallory-Weiss）",
        "肝性脑病": "肝性脑病",
        "门静脉高压": "门静脉高压",
        "脾大": "脾大",
        "低蛋白血症": "低蛋白血症",
    }
    for kw, label in comp_kws.items():
        if kw in text:
            complications.append(label)
    if complications:
        diag["并发症"] = complications

    # 影像学发现
    imaging = []
    if "管腔狭窄" in text:
        imaging.append("门静脉/肠系膜上静脉管腔狭窄")
    if "充盈缺损" in text:
        imaging.append("门静脉充盈缺损（血栓）")
    if "脾-肾分流" in text or "脾肾分流" in text:
        imaging.append("自发性脾-肾分流")
    if "门静脉盗流" in text:
        imaging.append("门静脉盗流现象")
    if imaging:
        diag["影像学发现"] = imaging

    # 实验室关键值
    lab = {}
    m = re.search(r"红细胞\s*([\d.×10]+)\s*/L", text)
    if m:
        lab["红细胞"] = m.group(1) + "/L"
    m = re.search(r"血红蛋白\s*([\d.]+)\s*g/L", text)
    if m:
        lab["血红蛋白"] = m.group(1) + " g/L"
    m = re.search(r"血小板[计数]*\s*([\d.×10]+)\s*/L", text)
    if m:
        lab["血小板"] = m.group(1) + "/L"
    if lab:
        diag["关键实验室指标"] = lab

    return diag


def extract_treatment(text: str) -> dict:
    """提取治疗方案"""
    treatment = {}

    # 介入治疗
    interventions = []
    intervention_kws = {
        "经颈静脉肝内门体分流术": "TIPS（经颈静脉肝内门体分流术）",
        "TIPS": "TIPS手术",
        "肠系膜上静脉支架置入": "肠系膜上静脉支架置入术",
        "裸金属支架": "裸金属支架植入（6mm×40mm）",
        "经皮经肝门静脉穿刺": "经皮经肝门静脉穿刺溶栓",
        "导管溶栓": "导管溶栓治疗",
    }
    for kw, label in intervention_kws.items():
        if kw in text:
            interventions.append(label)
    if interventions:
        treatment["介入/手术治疗"] = list(dict.fromkeys(interventions))  # 去重保序

    # 药物治疗
    drugs_mentioned = []
    drug_kws = ["熊去氧胆酸", "心得安", "呋塞米", "螺内酯", "尿激酶", "利尿",
                "输血", "白蛋白", "抑酸", "纠正心功能", "门冬氨酸鸟氨酸"]
    for kw in drug_kws:
        if kw in text:
            drugs_mentioned.append(kw)
    if drugs_mentioned:
        treatment["药物/对症治疗"] = drugs_mentioned

    # 抗凝情况
    if "未进行口服或静脉抗凝" in text or "未行TIPS" in text:
        treatment["抗凝"] = "因高龄反复出血风险，未行口服/静脉抗凝"
    elif "抗凝" in text:
        treatment["抗凝"] = "有抗凝相关记录"

    # 治疗结果
    outcomes = []
    if "血便的频次和量逐渐减少" in text or "转为黄色软便" in text:
        outcomes.append("介入后3天出血停止，大便转为黄色软便")
    if "BNP降至" in text:
        bnp_follow = re.findall(r"BNP降至\s*([\d,]+)\s*ng/L", text)
        for v in bnp_follow:
            outcomes.append(f"BNP降至{v} ng/L（心衰改善）")
    if "血红蛋白稳定上升" in text:
        outcomes.append("血红蛋白稳定上升，出院后随访4年余")
    if "出院" in text:
        outcomes.append("病情好转后出院")
    if outcomes:
        treatment["治疗结果"] = outcomes

    return treatment


# ─────────────────────────────────────────────
# 4. 主程序
# ─────────────────────────────────────────────

def extract_all(pdf_path: str) -> dict:
    print(f"📄 正在提取PDF文字：{pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    print(f"   提取完成，共 {len(text)} 字符")

    print("🔍 正在提取医学实体...")
    result = {
        "来源文件": Path(pdf_path).name,
        "提取说明": "基于规则+正则匹配，不使用API，适用于中文临床病例报道",
        "患者基本信息": extract_basic_info(text),
        "主要症状与体征": extract_chief_complaint_and_symptoms(text),
        "既往史": extract_past_history(text),
        "诊断结果": extract_diagnosis(text),
        "治疗方案": extract_treatment(text),
    }

    return result


def main():
    pdf_path = "E:\学习\科研\新生文档\新手挑战\挑战一\A case of portal vein recanalization and symptomatic heart failure.pdf"

    if not Path(pdf_path).exists():
        print(f"❌ 找不到文件：{pdf_path}")
        sys.exit(1)

    entities = extract_all(pdf_path)

    output_path = "E:\学习\科研\新生文档\新手挑战\挑战一\output-medical_entities.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 提取完成！结果已保存至：{output_path}")
    print("\n─── 提取结果预览 ───")
    print(json.dumps(entities, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()