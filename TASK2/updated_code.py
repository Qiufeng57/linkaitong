"""
心力衰竭患者生存预测 — 完整分析脚本 (优化版)
==============================================
任务: 基于心衰临床记录数据集, 完成数据预处理、因子检测、概率预测与模型评估
作者: [你的姓名]
环境依赖: 见同目录 requirements.txt
        pip install -r requirements.txt
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 机器学习与评估
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, auc,
                             classification_report, confusion_matrix)

# 深度学习 (加分项)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局设置
# ============================================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 所有输出文件保存到脚本所在目录 (相对路径, 保证可复现性)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. 数据初处理 (Data Exploratory & Preprocessing)
# ============================================================
print("=" * 60)
print("1. 数据初处理 (Data Exploratory & Preprocessing)")
print("=" * 60)

# 1.1 加载数据 (相对路径, 脚本与数据置于同一目录)
data_path = os.path.join(OUTPUT_DIR, 'heart_failure_clinical_records_dataset.csv')
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"数据集未找到: {data_path}\n请确保脚本与 CSV 文件在同一目录下。"
    )

df = pd.read_csv(data_path)
print(f"\n[数据集形状]: {df.shape[0]} 条记录, {df.shape[1]} 个字段")
print(f"[目标变量分布]: 存活 {(df['DEATH_EVENT']==0).sum()} 例, "
      f"死亡 {(df['DEATH_EVENT']==1).sum()} 例 "
      f"(死亡率 {df['DEATH_EVENT'].mean():.1%})")

# 1.2 描述性统计
print("\n[描述性统计结果]:")
desc = df.describe().T
print(desc.to_string())

# 1.3 缺失值检查
missing_total = df.isnull().sum().sum()
if missing_total == 0:
    print("\n[缺失值检查]: 数据集无缺失值 (Missing Values = 0), 无需填充。")
else:
    print(f"\n[缺失值检查]: 发现 {missing_total} 个缺失值, 执行删除处理。")
    df.dropna(inplace=True)

# 1.4 异常值检测与处理 (IQR 方法)
print("\n[异常值检测 (IQR 方法)]:")
continuous_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                   'platelets', 'serum_creatinine', 'serum_sodium', 'time']
outlier_report = []
for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report.append({'Feature': col, 'Q1': Q1, 'Q3': Q3,
                           'IQR': IQR, 'Lower': lower, 'Upper': upper,
                           'Outliers': n_outliers})

outlier_df = pd.DataFrame(outlier_report)
print(outlier_df.to_string(index=False))

# 对异常值进行 Winsorize 截断 (截至 IQR 边界), 而非直接删除, 以保留样本量
n_before = len(df)
for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)

print(f"\n[异常值处理]: 已对连续变量执行 Winsorize 截断处理 (保留全部 {n_before} 条样本)。")
print("  说明: 临床数据中的极端值可能反映真实的危重病例, 因此采用截断 (Winsorize)")
print("  而非删除, 既抑制极端值对模型的干扰, 又不丢失宝贵的少数类样本。")


# ============================================================
# 2. 因子检测 (Risk Factor Detection / Feature Engineering)
# ============================================================
print("\n" + "=" * 60)
print("2. 因子检测 (Risk Factor Detection)")
print("=" * 60)

# 2.1 相关性热力图
plt.figure(figsize=(12, 10))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap of Heart Failure Clinical Records', fontsize=14)
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[热力图]: 已保存至 {heatmap_path}")

# 与 DEATH_EVENT 的 Pearson 相关系数排序
print("\n[各特征与 DEATH_EVENT 的 Pearson 相关系数 (绝对值排序)]:")
corr_with_target = corr_matrix['DEATH_EVENT'].drop('DEATH_EVENT').abs().sort_values(ascending=False)
print(corr_with_target.to_string())

# 2.2 统计学检验: Welch 独立样本 T 检验 (不假设方差齐性)
print("\n[独立样本 T 检验 — Welch's t-test (死亡组 vs 存活组)]:")
dead = df[df['DEATH_EVENT'] == 1]
alive = df[df['DEATH_EVENT'] == 0]
ttest_results = []
for col in continuous_cols:
    t_stat, p_val = stats.ttest_ind(dead[col], alive[col], equal_var=False)
    ttest_results.append({
        'Feature': col,
        'Mean_Dead': dead[col].mean(),
        'Mean_Alive': alive[col].mean(),
        't_statistic': t_stat,
        'p_value': p_val,
        'Significant': '***' if p_val < 0.001 else ('**' if p_val < 0.01
                        else ('*' if p_val < 0.05 else 'ns'))
    })
ttest_df = pd.DataFrame(ttest_results).sort_values('p_value')
print(ttest_df.to_string(index=False))
print("  注: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著")

# 2.3 随机森林特征重要性 (基于 Gini 不纯度)
X_all = df.drop(columns=['DEATH_EVENT'])
y_all = df['DEATH_EVENT']

rf_selector = RandomForestClassifier(n_estimators=200, random_state=SEED)
rf_selector.fit(X_all, y_all)

importance_df = pd.DataFrame({
    'Feature': X_all.columns,
    'Importance': rf_selector.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n[特征重要性排序 (Random Forest, n_estimators=200)]:")
print(importance_df.to_string(index=False))

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature',
            palette='viridis', hue='Feature', dodge=False, legend=False)
plt.title('Feature Importance (Random Forest)', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Clinical Feature')
plt.tight_layout()
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
plt.savefig(fi_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[特征重要性图]: 已保存至 {fi_path}")

# 2.4 Top 3 危险因子及医学解释
top3 = importance_df['Feature'].head(3).tolist()
print(f"\n[核心危险因子 Top 3]: {top3}")

medical_explanations = {
    'time': (
        "随访时间 (time): 最强预测因子。随访时间短的患者往往在确诊后早期即发生"
        "死亡事件, 反映病情的急性程度。较短的随访时间本身代表患者未能长期存活, "
        "与 DEATH_EVENT 高度负相关 (r ≈ -0.53)。临床启示: 对新确诊心衰患者"
        "应加强早期干预与密切监测。"
    ),
    'serum_creatinine': (
        "血清肌酐 (serum_creatinine): 肾功能核心标志物。心衰患者常伴发心肾综合征"
        " (Cardiorenal Syndrome), 心输出量下降 → 肾灌注不足 → 肌酐升高。"
        "文献表明肌酐 > 1.5 mg/dL 的心衰患者死亡风险显著增加。数据中死亡组均值"
        "显著高于存活组 (p < 0.001), 与 DEATH_EVENT 正相关 (r ≈ 0.29)。"
    ),
    'ejection_fraction': (
        "射血分数 (ejection_fraction): 左心室收缩功能的金标准指标, 正常值 50-70%。"
        "低于 40% 即为射血分数降低型心衰 (HFrEF), 泵血能力严重受损, 器官灌注不足, "
        "死亡风险随之升高。数据中与 DEATH_EVENT 呈负相关 (r ≈ -0.27), "
        "死亡组均值显著低于存活组 (p < 0.001)。"
    ),
    'age': (
        "年龄 (age): 高龄是心血管疾病的独立危险因素。随年龄增长, 心肌纤维化加重、"
        "血管顺应性下降、合并症增多, 心衰患者代偿能力减弱。数据中死亡组平均年龄"
        "显著高于存活组 (p < 0.001), 与 DEATH_EVENT 正相关 (r ≈ 0.25)。"
    ),
    'serum_sodium': (
        "血清钠 (serum_sodium): 低钠血症在心衰患者中常见, 与神经内分泌系统"
        "过度激活 (RAAS、ADH) 有关。血清钠 < 135 mEq/L 是心衰预后不良的"
        "独立预测因子。与 DEATH_EVENT 负相关 (r ≈ -0.20)。"
    ),
}

print("\n[Top 3 危险因子的医学解释]:")
for i, feat in enumerate(top3, 1):
    explanation = medical_explanations.get(feat, "暂无详细解释。")
    print(f"\n  {i}. {explanation}")


# ============================================================
# 3. 概率预测与多模型建模 (Modeling & Evaluation)
# ============================================================
print("\n" + "=" * 60)
print("3. 概率预测与建模 (Probability Prediction)")
print("=" * 60)

# 3.1 训练集/测试集划分 (8:2, 分层抽样)
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n[数据划分]: 训练集 {len(X_train)} 条, 测试集 {len(X_test)} 条")

# 3.2 数据标准化 (Z-score)
# 说明: 树模型 (Random Forest, XGBoost) 基于特征分裂, 具有尺度不变性, 理论上不需要标准化。
# 但 Logistic Regression 依赖梯度下降优化, 特征尺度差异大会导致收敛慢、正则化权重不公平;
# MLP 深度学习同理, 不同量纲的输入会导致梯度爆炸或消失。
# 因此统一使用 Z-score 标准化, 使多模型对比在公平的特征尺度下进行。
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[标准化]: 已完成 Z-score 标准化 (fit on train, transform on test)")

# 3.3 模型训练
results = {}

# --- 模型 1: Random Forest ---
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
results['Random Forest'] = (y_pred_rf, y_prob_rf)

# --- 模型 2: Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=SEED)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = (y_pred_lr, y_prob_lr)

# --- 模型 3: XGBoost ---
xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                           random_state=SEED, eval_metric='logloss',
                           use_label_encoder=False)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
results['XGBoost'] = (y_pred_xgb, y_prob_xgb)

# --- 模型 4: MLP 多层感知机 (加分项 — PyTorch 实现) ---
class HeartFailureMLP(nn.Module):
    """
    三隐藏层全连接网络, 用于心衰死亡事件二分类。
    结构: Input(12) → FC(64) → BN → ReLU → Dropout(0.3)
                    → FC(32) → BN → ReLU → Dropout(0.2)
                    → FC(16) → ReLU → FC(1) → Sigmoid
    """
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# 数据转换为 PyTorch 张量
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test_scaled)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

mlp = HeartFailureMLP(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# 训练循环
mlp.train()
for epoch in range(150):
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = mlp(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 50 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f"  [MLP] Epoch {epoch+1}/150, Avg Loss: {avg_loss:.4f}")

mlp.eval()
with torch.no_grad():
    y_prob_mlp = mlp(X_test_t).numpy().flatten()
    y_pred_mlp = (y_prob_mlp >= 0.5).astype(int)
results['MLP (Deep Learning)'] = (y_pred_mlp, y_prob_mlp)

print("\n[模型训练完成]: Random Forest, Logistic Regression, XGBoost, MLP")


# ============================================================
# 4. 模型评估 (Evaluation)
# ============================================================
print("\n" + "=" * 60)
print("4. 模型评估 (Model Evaluation)")
print("=" * 60)

eval_records = []
plt.figure(figsize=(10, 8))

for name, (y_pred, y_prob) in results.items():
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    eval_records.append({
        'Model': name, 'Accuracy': f"{acc:.4f}",
        'Precision': f"{prec:.4f}", 'Recall': f"{rec:.4f}",
        'F1 Score': f"{f1:.4f}", 'AUC': f"{roc_auc:.4f}"
    })
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

eval_table = pd.DataFrame(eval_records)
print("\n[模型评估指标对照表]:")
print(eval_table.to_string(index=False))

# 绘制 ROC 曲线
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves — Multi-Model Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, 'roc_curves_comparison.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[ROC 曲线]: 已保存至 {roc_path}")


# ============================================================
# 5. 新患者预测示例 (Clinical Inference)
# ============================================================
print("\n" + "=" * 60)
print("5. 新患者概率预测示例 (Clinical Inference)")
print("=" * 60)

# 构造两个具有鲜明临床对比的虚拟病例 (特征顺序与原 DataFrame 一致)
new_patients = pd.DataFrame([
    {   # 患者 1: 高风险 — 高龄、射血分数极低、肌酐升高、随访极早期
        'age': 75.0, 'anaemia': 1, 'creatinine_phosphokinase': 582,
        'diabetes': 1, 'ejection_fraction': 20, 'high_blood_pressure': 1,
        'platelets': 265000.0, 'serum_creatinine': 2.5,
        'serum_sodium': 130, 'sex': 1, 'smoking': 1, 'time': 4
    },
    {   # 患者 2: 低风险 — 中年、各项指标正常、长期随访平稳
        'age': 45.0, 'anaemia': 0, 'creatinine_phosphokinase': 120,
        'diabetes': 0, 'ejection_fraction': 55, 'high_blood_pressure': 0,
        'platelets': 300000.0, 'serum_creatinine': 0.8,
        'serum_sodium': 140, 'sex': 0, 'smoking': 0, 'time': 240
    }
], index=['Patient_1 (高风险)', 'Patient_2 (低风险)'])

print("\n[新患者临床体征输入]:")
print(new_patients.T.to_string())

# 标准化并预测
new_scaled = scaler.transform(new_patients)

prob_rf = rf.predict_proba(new_scaled)[:, 1]
prob_lr = lr.predict_proba(new_scaled)[:, 1]
prob_xgb = xgb_model.predict_proba(new_scaled)[:, 1]

mlp.eval()
with torch.no_grad():
    prob_mlp = mlp(torch.FloatTensor(new_scaled)).numpy().flatten()

pred_table = pd.DataFrame({
    'Random Forest':       [f"{p:.2%}" for p in prob_rf],
    'Logistic Regression': [f"{p:.2%}" for p in prob_lr],
    'XGBoost':             [f"{p:.2%}" for p in prob_xgb],
    'MLP (Deep Learning)': [f"{p:.2%}" for p in prob_mlp],
}, index=new_patients.index)

print("\n[多模型死亡概率预测对比 (P(DEATH_EVENT = 1))]:")
print(pred_table.to_string())

print("\n" + "=" * 60)
print("全流程运行完毕。输出文件:")
print(f"  - {heatmap_path}")
print(f"  - {fi_path}")
print(f"  - {roc_path}")
print("=" * 60)
