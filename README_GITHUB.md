# WWEMM

**Weather Extremes & Winter Wheat Modeling (China)**

一个面向冬小麦物候期气候风险分析的机器学习项目。  
A machine-learning project for phenology-specific climate risk analysis of winter wheat.

## 项目简介 | Overview

- 研究中国冬小麦在不同物候阶段（P1-P6）下，对极端高温、干旱、强降水和复合热干事件的响应。  
  Analyze stage-specific (P1-P6) responses of winter wheat to extreme heat, drought, heavy rainfall, and compound hot-dry events in China.
- 使用 CatBoost + SHAP 识别关键驱动因子与非线性阈值特征。  
  Use CatBoost + SHAP to identify key drivers and nonlinear threshold patterns.

## 仓库内容 | What's Inside

- `Min-Max normalization.py`  
  对气候指标做 Min-Max 归一化。  
  Min-Max scaling for climate indicators.
- `Catboost.py`  
  模型训练、网格搜索、评估与 SHAP 可解释分析。  
  Model training, grid search, evaluation, and SHAP interpretation.
- `gpp_trend_mk.py`  
  多年 GPP 栅格逐像元趋势分析（Theil-Sen + Mann-Kendall）。  
  Pixel-wise multi-year GPP trend analysis (Theil-Sen + Mann-Kendall).
- `DDML代码.do`  
  Stata 中的 DDML 因果估计、稳健性与异质性分析。  
  DDML causal estimation in Stata, with robustness and heterogeneity checks.

## 快速开始 | Quick Start

### 1) 安装依赖 | Install dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn catboost shap pygam openpyxl rasterio pymannkendall tqdm
```

### 2) 配置路径 | Configure paths

- 在 `Min-Max normalization.py` 中设置：`INPUT_PATH`, `OUTPUT_PATH`, `PARAMS_PATH`
- 在 `Catboost.py` 中设置：`DATA_PATH`, `OUTPUT_DIR`
- 在 `gpp_trend_mk.py` 中设置：`data_dir`, `years`, `months`, `min_valid_years`
- 在 `DDML代码.do` 中设置：`data.dta` 路径与变量定义（`Y`, `D`, `X`）

Set the path variables in both scripts before running.

### 3) 运行 | Run

```bash
python "Min-Max normalization.py"
python Catboost.py
python gpp_trend_mk.py
# 在 Stata 中运行:
# do "DDML代码.do"
```

## 主要输出 | Main Outputs

- `model_metrics.xlsx`（R2/MAE/MSE/RMSE）
- `best_params.xlsx`（最优参数与 CV 结果）
- `feature_importance.*`（特征重要性）
- `SHAP-importance.jpg`, `dependence_plot*.jpg`, `SHAP.jpg`（可解释性结果）
- `GPP_{month}_TheilSen_slope.tif`, `GPP_{month}_MK_p.tif`, `GPP_{month}_MK_z.tif`（趋势与显著性栅格）
- `word.doc`（DDML 回归结果导出，来自 `outreg2`）

## 说明 | Notes

- 当前脚本以绝对路径为主，建议改为相对路径或命令行参数。  
  Current scripts mainly use absolute paths; consider replacing with relative paths or CLI args.
- 完整版文档见 `README.md`。  
  See `README.md` for full documentation.
