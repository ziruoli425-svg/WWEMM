# WWEMM: Winter Wheat Extreme-Climate Modeling

## 中文说明 (Chinese)

本项目用于分析中国冬小麦在不同物候阶段下，极端气候事件（高温、干旱、强降水及复合热干事件）对生产力（GPP）的影响。

当前仓库提供四部分核心流程：

- `Min-Max normalization.py`：对指定气候指标做 Min-Max 归一化
- `Catboost.py`：训练 CatBoost 回归模型，并输出模型评估、特征重要性与 SHAP 解释结果
- `gpp_trend_mk.py`：对多年 GPP 栅格进行逐像元 Theil-Sen 趋势估计与 Mann-Kendall 显著性检验
- `DDML代码.do`：在 Stata 中执行 DDML（双重机器学习）估计，并开展稳健性与异质性分析

### 1. 研究背景与目标

基于文稿《When Weather Extremes Matter Most: Phenology-Specific Impacts of Heat, Drought, and Hot-Dry Compounds on China's Winter Wheat Productivity》，本研究关注：

- 冬小麦不同物候阶段（P1-P6）对气候胁迫的差异化响应
- 极端高温与复合热干事件在生殖生长期的主导限制作用
- 通过可解释机器学习（SHAP）识别非线性阈值效应

### 2. 环境要求

- 推荐 Python 版本：`3.8`
- 主要依赖：`scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `catboost`, `shap`, `pygam`, `openpyxl`, `rasterio`, `pymannkendall`, `tqdm`

安装示例：

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn catboost shap pygam openpyxl rasterio pymannkendall tqdm
```

### 3. 数据准备

#### 3.1 归一化脚本输入

编辑 `Min-Max normalization.py` 中以下路径：

- `INPUT_PATH`：原始 Excel 数据
- `OUTPUT_PATH`：归一化后数据输出路径
- `PARAMS_PATH`：归一化参数输出路径

默认归一化变量：

- 干旱/复合干热：`CDD`, `HDCI`, `SU25&CDD`, `PI`, `p-SPI`
- 降水相关：`R95p`, `Rx5day`, `CWD`, `PRCPTOT`, `R10`, `SDII`
- 高温相关：`TR20`, `TXx`, `SDD`, `SU25`

#### 3.2 建模脚本输入

编辑 `Catboost.py` 中：

- `DATA_PATH`：建模数据 Excel 路径
- `OUTPUT_DIR`：结果输出目录（默认 `./outputs_catboost_gridsearch`）

脚本默认以 `GPP` 为目标变量，并默认删除 `County`, `Year`, `Month`, `GPP` 作为非特征列。

#### 3.3 趋势分析脚本输入

编辑 `gpp_trend_mk.py` 中：

- `data_dir`：GPP 栅格数据目录（输入与输出目录）
- `years`：分析年份范围（默认 `2000-2024`）
- `months`：时段标识（默认 `["310_331", "1015_1130", "Feb10_Mar10"]`）
- `min_valid_years`：像元进入趋势分析的最小有效样本数（默认 `10`）

输入文件命名规则示例：

- `MODIS_GPP_2000_310_331.tif`
- `MODIS_GPP_2012_1015_1130.tif`
- `MODIS_GPP_2024_Feb10_Mar10.tif`

#### 3.4 DDML 脚本输入

编辑 `DDML代码.do` 中：

- `use data.dta, clear`：主输入面板数据（需放在 Stata 工作目录，或改为绝对路径）
- 处理变量与控制变量：
  - 因变量：如 `GPP_MinMax` / `A`
  - 处理变量：如 `PRCPTOT_MinMax` / `water` / `TXx`
  - 控制变量：如 `CO2_MinMax`, `Tmean_MinMax`, `Rn_MinMax`, `i.year`, `i.City`
- 输出文件：`outreg2 ... using word.doc`（可改为你需要的输出文件名）

依赖说明（Stata）：

- 需要可用的 `ddml` 与 `pystacked` 环境
- 稳健性部分使用了 `winsor2` 与 `outreg2`

### 4. 运行步骤

#### 步骤 1：归一化（可选但推荐）

```bash
python "Min-Max normalization.py"
```

输出：

- 归一化后的数据文件（`OUTPUT_PATH`）
- 每个特征的 `min/max/range` 参数文件（`PARAMS_PATH`）

#### 步骤 2：模型训练与解释

```bash
python Catboost.py
```

主要流程：

- 训练/测试集划分（70/30）
- 5 折交叉验证 + 网格搜索（`learning_rate`, `depth`, `l2_leaf_reg`, `iterations`）
- 最优参数重训
- 输出评估指标：`R2`, `MAE`, `MSE`, `RMSE`
- SHAP 全局重要性与 dependence 分析
- GAM 对 SHAP 散点关系进行平滑拟合

#### 步骤 3：GPP 趋势与显著性分析

```bash
python gpp_trend_mk.py
```

主要流程：

- 构建多年份 GPP 时序数据立方体（year x row x col）
- 逐像元计算 Theil-Sen 斜率（稳健趋势）
- 使用 Yue-Wang 修正的 Mann-Kendall 检验输出 `p` 与 `z`
- 对每个时段分别生成趋势与显著性栅格

#### 步骤 4：DDML 因果估计（Stata）

在 Stata 中运行：

```stata
do "DDML代码.do"
```

主要流程：

- 构建 DDML partialling-out 框架并进行交叉拟合（`kfolds`）
- 使用多种学习器（LassoCV、随机森林、神经网络、SVM、弹性网络等）估计条件期望项
- 输出基准回归、稳健性检验（缩尾、样本分割、学习器替换）与异质性结果

### 5. 输出文件说明

默认输出目录：`outputs_catboost_gridsearch`

常见输出：

- `grid_search_cv_results.xlsx`：网格搜索详细结果
- `best_params.xlsx`：最优参数与最优 CV RMSE
- `model_metrics.xlsx`：训练集/测试集评估指标
- `prediction_comparison.jpg`：True vs Predict 对比图
- `feature_importance.xlsx` / `feature_importance.jpg`：CatBoost 特征重要性
- `SHAP-importance.jpg`：Mean(|SHAP|) 归一化重要性
- `dependence_plot2.jpg` / `dependence_plot3.jpg` / `dependence_plot.jpg`：SHAP dependence 图
- `SHAP.jpg`：SHAP 汇总组合图
- `GPP_{month}_TheilSen_slope.tif`：像元级趋势斜率图
- `GPP_{month}_MK_p.tif`：Mann-Kendall 显著性 `p` 值图
- `GPP_{month}_MK_z.tif`：Mann-Kendall `z` 统计量图
- `word.doc`：DDML 回归结果导出文档（由 `outreg2` 生成）

### 6. 注意事项

- 当前脚本路径多为绝对路径，迁移机器时请先修改路径配置。
- 脚本包含中文字体设置（如 `SimHei`, `Microsoft YaHei`）；若本机缺失字体，图形显示可能异常。
- `Catboost.py` 中 `target_column = 'y'` 仅用于图例标题，不影响训练目标；训练目标仍是 `GPP`。
- 若数据列名与脚本不一致，请同步更新列名配置。
- 趋势分析脚本已采用清晰命名 `gpp_trend_mk.py`，便于协作与复现。
- 建议将 `DDML代码.do` 重命名为英文名（如 `ddml_analysis.do`），避免跨系统编码问题。

### 7. 后续改进建议

- 将硬编码路径改为命令行参数（`argparse`）
- 提供 `requirements.txt` 与固定版本
- 增加随机种子、数据切分、参数空间等实验配置文件
- 增加多物候阶段批处理与自动汇总报告

---

## English Version

This project analyzes how extreme climate events (heat, drought, heavy rainfall, and compound hot-dry events) affect winter wheat productivity (GPP) across phenological stages in China.

The repository currently includes four core workflows:

- `Min-Max normalization.py`: applies Min-Max scaling to selected climate indicators
- `Catboost.py`: trains a CatBoost regressor and exports model evaluation, feature importance, and SHAP-based interpretation outputs
- `gpp_trend_mk.py`: performs pixel-wise Theil-Sen trend estimation and Mann-Kendall significance testing for multi-year GPP rasters
- `DDML代码.do`: runs DDML estimation in Stata, including robustness and heterogeneity analyses

### 1. Background and Objectives

Based on the manuscript *When Weather Extremes Matter Most: Phenology-Specific Impacts of Heat, Drought, and Hot-Dry Compounds on China's Winter Wheat Productivity*, this project focuses on:

- Stage-specific climate sensitivity across six phenological periods (P1-P6)
- Dominant constraints from extreme heat and compound hot-dry stress during reproductive stages
- Nonlinear threshold identification using interpretable machine learning (SHAP)

### 2. Environment Requirements

- Recommended Python version: `3.8`
- Main dependencies: `scikit-learn`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `catboost`, `shap`, `pygam`, `openpyxl`, `rasterio`, `pymannkendall`, `tqdm`

Installation example:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn catboost shap pygam openpyxl rasterio pymannkendall tqdm
```

### 3. Data Preparation

#### 3.1 Inputs for normalization script

In `Min-Max normalization.py`, set:

- `INPUT_PATH`: source Excel file
- `OUTPUT_PATH`: output path for normalized data
- `PARAMS_PATH`: output path for scaling parameters

Default variables to scale:

- Drought/compound hot-dry: `CDD`, `HDCI`, `SU25&CDD`, `PI`, `p-SPI`
- Rainfall-related: `R95p`, `Rx5day`, `CWD`, `PRCPTOT`, `R10`, `SDII`
- Heat-related: `TR20`, `TXx`, `SDD`, `SU25`

#### 3.2 Inputs for modeling script

In `Catboost.py`, set:

- `DATA_PATH`: Excel dataset path for modeling
- `OUTPUT_DIR`: output directory (default: `./outputs_catboost_gridsearch`)

The script uses `GPP` as target and drops `County`, `Year`, `Month`, and `GPP` from features.

#### 3.3 Inputs for trend-analysis script

In `gpp_trend_mk.py`, set:

- `data_dir`: input/output directory for GPP rasters
- `years`: analysis period (default `2000-2024`)
- `months`: stage windows (default `["310_331", "1015_1130", "Feb10_Mar10"]`)
- `min_valid_years`: minimum valid years per pixel for trend estimation (default `10`)

Input naming examples:

- `MODIS_GPP_2000_310_331.tif`
- `MODIS_GPP_2012_1015_1130.tif`
- `MODIS_GPP_2024_Feb10_Mar10.tif`

#### 3.4 Inputs for DDML script

In `DDML代码.do`, configure:

- `use data.dta, clear`: main panel dataset path
- outcome/treatment/control variables (e.g., `GPP_MinMax`, `PRCPTOT_MinMax`, `i.year`, `i.City`)
- `outreg2 ... using word.doc`: output report filename

Stata dependencies:

- `ddml` with `pystacked`
- `winsor2` and `outreg2` for robustness processing and result export

### 4. How to Run

#### Step 1: Normalize data (optional but recommended)

```bash
python "Min-Max normalization.py"
```

Outputs:

- normalized dataset (`OUTPUT_PATH`)
- per-feature scaling parameters (`PARAMS_PATH`)

#### Step 2: Train model and generate interpretations

```bash
python Catboost.py
```

Main pipeline:

- Train/test split (70/30)
- 5-fold cross-validation + grid search (`learning_rate`, `depth`, `l2_leaf_reg`, `iterations`)
- Refit with best parameters
- Export metrics: `R2`, `MAE`, `MSE`, `RMSE`
- SHAP global importance and dependence analysis
- GAM smoothing over SHAP scatter relationships

#### Step 3: Run GPP trend and significance analysis

```bash
python gpp_trend_mk.py
```

Main pipeline:

- Build a multi-year GPP data cube (year x row x col)
- Compute pixel-wise Theil-Sen slope
- Run Yue-Wang modified Mann-Kendall test and output `p` and `z`
- Export trend/significance rasters for each time window

#### Step 4: Run DDML estimation in Stata

Run in Stata:

```stata
do "DDML代码.do"
```

Main pipeline:

- Initialize partial DDML with cross-fitting (`kfolds`)
- Estimate nuisance functions with multiple learners (LassoCV, RF, NNet, SVM, ElasticNet)
- Export baseline, robustness, and heterogeneity results

### 5. Output Files

Default output directory: `outputs_catboost_gridsearch`

Typical outputs:

- `grid_search_cv_results.xlsx`: detailed grid-search results
- `best_params.xlsx`: best hyperparameters and best CV RMSE
- `model_metrics.xlsx`: train/test metrics
- `prediction_comparison.jpg`: True vs Predict plot
- `feature_importance.xlsx` / `feature_importance.jpg`: CatBoost feature importance
- `SHAP-importance.jpg`: normalized Mean(|SHAP|) importance
- `dependence_plot2.jpg` / `dependence_plot3.jpg` / `dependence_plot.jpg`: SHAP dependence plots
- `SHAP.jpg`: combined SHAP visualization
- `GPP_{month}_TheilSen_slope.tif`: pixel-wise trend slope map
- `GPP_{month}_MK_p.tif`: Mann-Kendall `p` value map
- `GPP_{month}_MK_z.tif`: Mann-Kendall `z` statistic map
- `word.doc`: exported DDML result table from `outreg2`

### 6. Notes

- Most paths are hard-coded absolute paths; update them before running on another machine.
- The scripts use Chinese font settings (e.g., `SimHei`, `Microsoft YaHei`); missing fonts may affect plot rendering.
- In `Catboost.py`, `target_column = 'y'` is used only for plot labels and does not change the actual target variable (`GPP`).
- If your dataset uses different column names, update the script accordingly.
- The trend script is already renamed to `gpp_trend_mk.py` for better maintainability.
- Consider renaming `DDML代码.do` to an ASCII filename (e.g., `ddml_analysis.do`) for better cross-platform compatibility.

### 7. Suggested Next Improvements

- Replace hard-coded paths with CLI arguments (`argparse`)
- Add a pinned `requirements.txt`
- Move split strategy, random seed, and search space into config files
- Add batch processing and automatic report generation across phenological stages
#   W W E M M  
 