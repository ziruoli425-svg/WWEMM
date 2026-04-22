#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =========================
# 输入输出路径
# =========================
INPUT_PATH = r""
OUTPUT_PATH = r""
PARAMS_PATH = r""

# =========================
# 需要归一化的列
# 可以自己增删
# =========================
cols_to_scale = [
    'CDD', 'HDCI', 'SU25&CDD', 'PI', 'p-SPI',
    'R95p', 'Rx5day', 'CWD', 'PRCPTOT', 'R10', 'SDII',
    'TR20', 'TXx', 'SDD', 'SU25'
]

# =========================
# 读取数据
# =========================
df = pd.read_excel(INPUT_PATH)
print("原始数据形状:", df.shape)

# 检查列是否存在
missing_cols = [c for c in cols_to_scale if c not in df.columns]
if missing_cols:
    raise ValueError(f"以下列在表中不存在，请检查列名: {missing_cols}")

# =========================
# MinMax 归一化
# =========================
scaler = MinMaxScaler()

df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# =========================
# 保存归一化后的数据
# =========================
df_scaled.to_excel(OUTPUT_PATH, index=False)

# 保存每列归一化参数
params_df = pd.DataFrame({
    'feature': cols_to_scale,
    'data_min': scaler.data_min_,
    'data_max': scaler.data_max_,
    'data_range': scaler.data_range_
})
params_df.to_excel(PARAMS_PATH, index=False)

print("归一化完成。")
print("归一化后的文件已保存到:", OUTPUT_PATH)
print("归一化参数已保存到:", PARAMS_PATH)