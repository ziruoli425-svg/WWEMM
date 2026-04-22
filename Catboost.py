#!/usr/bin/env python
# coding: utf-8

# =========================
# 代码运行包版本说明（参考）
# python 3.8
# scikit-learn==1.3.2
# pandas==1.3.5
# numpy==1.21.6
# shap==0.44.1
# scipy==1.10.1
# statsmodels==0.13.2
# matplotlib==3.5.2
# catboost >= 1.2
# pygam >= 0.8.0
# =========================

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cgb

from sklearn.model_selection import GridSearchCV, KFold
from sklearn import metrics

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
pd.options.display.max_columns = None


DATA_PATH = r"D:\desktop\18-copaper\1-ziangzhou\1219\data-all\2p5.xlsx"

# 以当前代码文件所在文件夹为基准
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 在代码所在文件夹下创建输出文件夹
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_catboost_gridsearch")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("输出目录：", OUTPUT_DIR)

def log(msg):
    print(msg)
    sys.stdout.flush()

# 读取数据
log("开始读取数据...")
data = pd.read_excel(DATA_PATH)

log("数据预览：")
print(data.head())
sys.stdout.flush()

log("\n数据信息：")
print(data.info())
sys.stdout.flush()

log(f"data shape: {data.shape}")

drop_cols = ['County', 'Year', 'Month', 'GPP']
X = data.drop(columns=[c for c in drop_cols if c in data.columns]).copy()
y = data['GPP'].copy()

year_values = data['Year'].dropna().unique()
rng = np.random.default_rng(1)
n_test_years = max(1, int(np.ceil(len(year_values) * 0.3)))
test_years = rng.choice(year_values, size=n_test_years, replace=False)
train_mask = ~data['Year'].isin(test_years)
test_mask = data['Year'].isin(test_years)
X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]
log(f"X_train, X_test: {X_train.shape}, {X_test.shape}")

# 仅用训练集均值填充缺失值，避免信息泄漏
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

# 评价函数
def try_different_method(y_pred_train1, y_pred_test1, y_train, y_test):
    log('训练集:')
    r2 = metrics.r2_score(y_train, y_pred_train1)
    mae = metrics.mean_absolute_error(y_train, y_pred_train1)
    mse = metrics.mean_squared_error(y_train, y_pred_train1)
    rmse = np.sqrt(mse)
    log("R2:  " + '%.4f' % float(r2) +
        "  绝对误差: " + '%.4f' % float(mae) +
        "  均方误差：" + '%.4f' % float(mse) +
        "  均方根误差:" + '%.4f' % float(rmse))
    train_metrics = [r2, mae, mse, rmse]

    log('*' * 50)
    log('测试集:')
    r2 = metrics.r2_score(y_test, y_pred_test1)
    mae = metrics.mean_absolute_error(y_test, y_pred_test1)
    mse = metrics.mean_squared_error(y_test, y_pred_test1)
    rmse = np.sqrt(mse)
    log("R2:  " + '%.4f' % float(r2) +
        "  绝对误差: " + '%.4f' % float(mae) +
        "  均方误差：" + '%.4f' % float(mse) +
        "  均方根误差:" + '%.4f' % float(rmse))
    test_metrics = [r2, mae, mse, rmse]
    return train_metrics, test_metrics

# 训练集内部：5折交叉验证 + 网格搜索
log("\n准备 CatBoost 基础模型...")
base_model = cgb.CatBoostRegressor(
    loss_function='RMSE',
    random_seed=1,
    verbose=0,
    thread_count=-1
)

# learning rate, tree depth, regularization
param_grid = {
    'learning_rate': [0.02, 0.05],
    'depth': [6, 8],
    'l2_leaf_reg': [1, 3],
    'iterations': [500, 1000]
}

cv_5fold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1
)

n_candidates = (
    len(param_grid['learning_rate']) *
    len(param_grid['depth']) *
    len(param_grid['l2_leaf_reg']) *
    len(param_grid['iterations'])
)
n_total_fits = n_candidates * 5

log("\n开始进行训练集内部 5 折交叉验证 + 网格搜索...")
log(f"参数组合数: {n_candidates}")
log(f"总拟合次数: {n_total_fits}")
log("正在搜索最优参数，请稍等...\n")

# 为了更明显地在终端看到打印，先用 n_jobs=1
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=cv_5fold,
    n_jobs=1,
    verbose=3,
    refit=True
)

start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

best_params = grid_search.best_params_
best_cv_rmse = -grid_search.best_score_

log("\n网格搜索完成。")
log(f"耗时: {(end_time - start_time):.2f} 秒")
log(f"最优参数：{best_params}")
log(f"最优5折交叉验证RMSE：{best_cv_rmse:.6f}")

# 保存网格搜索结果
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_excel(os.path.join(OUTPUT_DIR, 'grid_search_cv_results.xlsx'), index=False)

best_params_df = pd.DataFrame([best_params])
best_params_df['best_cv_rmse'] = best_cv_rmse
best_params_df.to_excel(os.path.join(OUTPUT_DIR, 'best_params.xlsx'), index=False)

# =========================
# 使用最优参数训练最终模型
# =========================
log("\n开始用最优参数训练最终模型...")
clf = cgb.CatBoostRegressor(
    **best_params,
    loss_function='RMSE',
    random_seed=1,
    verbose=200,   # 这里保留少量训练日志
    thread_count=-1
)

clf.fit(X_train, y_train)

log("最终模型训练完成，开始预测...")
y_pred_train2 = clf.predict(X_train)
y_pred_test2 = clf.predict(X_test)

log('\n模型评分如下：\n')
train_metrics, test_metrics = try_different_method(y_pred_train2, y_pred_test2, y_train, y_test)

metrics_df = pd.DataFrame({
    'dataset': ['train', 'test'],
    'R2': [train_metrics[0], test_metrics[0]],
    'MAE': [train_metrics[1], test_metrics[1]],
    'MSE': [train_metrics[2], test_metrics[2]],
    'RMSE': [train_metrics[3], test_metrics[3]]
})
metrics_df.to_excel(os.path.join(OUTPUT_DIR, 'model_metrics.xlsx'), index=False)

# True vs Predict 图
log("\n开始绘制 True vs Predict 图...")
pred_df = pd.concat([
    pd.DataFrame({'True': y_test, 'Predict': y_pred_test2, 'Group': ['Test'] * len(y_test)}),
    pd.DataFrame({'True': y_train, 'Predict': y_pred_train2, 'Group': ['Train'] * len(y_train)})
], axis=0)

from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

x_field = "True"
y_field = "Predict"
category_field = "Group"
x_title = "True"
y_title = "Predict"

variant = 2
palette = ['#66C2A5', '#FC8D62', '#8DA0CB']
alpha_ci = 0.2
confidence = 0.95

def plot_group_regression(ax, df_group, color, x_key, y_key):
    ax.scatter(df_group[x_key], df_group[y_key], color=color, s=60, alpha=0.4,
               edgecolor='black', linewidth=1)
    if len(df_group) <= 1:
        return None

    Xr = df_group[[x_key]].values
    yr = df_group[y_key].values
    model = LinearRegression().fit(Xr, yr)
    y_hat = model.predict(Xr)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = metrics.r2_score(yr, y_hat)
    n = len(Xr)

    xs = np.linspace(Xr.min(), Xr.max(), 100)
    ys = model.predict(xs.reshape(-1, 1))

    mse = np.sum((yr - y_hat) ** 2) / (n - 2)
    se = np.sqrt(mse * (1 / n + (xs - Xr.mean()) ** 2 / np.sum((Xr - Xr.mean()) ** 2)))
    t_val = stats.t.ppf((1 + confidence) / 2, n - 2)

    ax.fill_between(xs, ys - t_val * se, ys + t_val * se, color=color, alpha=alpha_ci, linewidth=0)
    ax.plot(xs, ys, color=color, linestyle="--", linewidth=2.5, alpha=0.8)

    return dict(group=df_group[category_field].iloc[0],
                r2=r2, slope=slope, intercept=intercept, color=color)

def margin_density(ax_top, ax_right, grouped_data, x_lims, y_lims, x_key, y_key):
    x_span = np.linspace(x_lims[0], x_lims[1], 200)
    y_span = np.linspace(y_lims[0], y_lims[1], 200)
    for label, df_sub, color in grouped_data:
        kde_x = gaussian_kde(df_sub[x_key])
        kde_y = gaussian_kde(df_sub[y_key])
        ax_top.plot(x_span, kde_x(x_span), color=color, linewidth=2)
        ax_top.fill_between(x_span, kde_x(x_span), color=color, alpha=0.3)
        ax_right.plot(kde_y(y_span), y_span, color=color, linewidth=2)
        ax_right.fill_betweenx(y_span, kde_y(y_span), color=color, alpha=0.3)

margin_modes = {2: margin_density}

fig = plt.figure(figsize=(8, 6), dpi=120)
ax_main = plt.axes([0.15, 0.15, 0.60, 0.60])
ax_top = plt.axes([0.15, 0.75 + 0.005, 0.60, 0.15])
ax_right = plt.axes([0.75 + 0.005, 0.15, 0.15, 0.60])

x_pad = (pred_df[x_field].max() - pred_df[x_field].min()) * 0.08
y_pad = (pred_df[y_field].max() - pred_df[y_field].min()) * 0.08
ax_main.set_xlim(pred_df[x_field].min() - x_pad, pred_df[x_field].max() + x_pad)
ax_main.set_ylim(pred_df[y_field].min() - y_pad, pred_df[y_field].max() + y_pad)

all_groups = []
reg_records = []
for idx, grp in enumerate(pred_df[category_field].unique()):
    df_g = pred_df[pred_df[category_field] == grp]
    color = palette[idx % len(palette)]
    all_groups.append((grp, df_g, color))
    reg_info = plot_group_regression(ax_main, df_g, color, x_field, y_field)
    if reg_info:
        reg_records.append(reg_info)

margin_modes[variant](ax_top, ax_right, all_groups,
                      ax_main.get_xlim(), ax_main.get_ylim(), x_field, y_field)

text_y = 0.93
for rec in reg_records:
    ax_main.text(0.03, text_y, f"{rec['group']}: $R^2$={rec['r2']:.3f}",
                 color=rec['color'], transform=ax_main.transAxes,
                 fontsize=14, fontweight='bold')
    text_y -= 0.06

ax_main.set_xlabel(x_title, fontsize=16, fontweight='bold')
ax_main.set_ylabel(y_title, fontsize=16, fontweight='bold')

for ax in [ax_top, ax_right]:
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

ax_top.set_xlim(ax_main.get_xlim())
ax_right.set_ylim(ax_main.get_ylim())

plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_comparison.jpg'), dpi=600, bbox_inches='tight')
plt.close()


# 变量重要性
log("开始输出feature_importance_plot...")
tmp1 = pd.DataFrame({
    'fea': X_train.columns,
    'imp_raw': clf.feature_importances_
}).sort_values(by='imp_raw', ascending=False).reset_index(drop=True)

tmp1['imp_ratio'] = tmp1['imp_raw'] / tmp1['imp_raw'].sum()
tmp1['imp_pct'] = tmp1['imp_ratio'] * 100

tmp1.to_excel(os.path.join(OUTPUT_DIR, 'feature_importance.xlsx'), index=False)

IMPORTANCE_PLOT_COL = 'imp_ratio'
VALUE_FMT = "{:.4f}"

fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
sns.barplot(x=tmp1[IMPORTANCE_PLOT_COL], y=tmp1['fea'], color='orange', ax=ax)

xmax = ax.get_xlim()[1]
offset = xmax * 0.01
for p in ax.patches:
    w = p.get_width()
    y0 = p.get_y() + p.get_height() / 2
    ax.text(w + offset, y0, VALUE_FMT.format(w), va='center', ha='left', fontsize=10, color='black')

ax.set_xlabel(IMPORTANCE_PLOT_COL)
ax.set_ylabel('fea')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.jpg'), dpi=600, bbox_inches='tight')
plt.close()

# =========================
# SHAP 解释（精简版）
# =========================
log("开始计算 SHAP...")
import shap

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

explainer = shap.TreeExplainer(clf)
shap_values1 = explainer(X_test)
shap_values2 = explainer.shap_values(X_test)


# Mean(|SHAP|) 重要性图
log("输出 SHAP-imoportance.jpg ...")
mean_abs_shap = np.abs(shap_values2).mean(axis=0)
shap_imp_df = pd.DataFrame({
    'fea': X_test.columns,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

shap_imp_df['shap_ratio'] = shap_imp_df['mean_abs_shap'] / shap_imp_df['mean_abs_shap'].sum()

fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
sns.barplot(x=shap_imp_df['shap_ratio'], y=shap_imp_df['fea'], color='orange', ax=ax)

xmax = ax.get_xlim()[1]
offset = xmax * 0.01
for p in ax.patches:
    w = p.get_width()
    y0 = p.get_y() + p.get_height() / 2
    ax.text(w + offset, y0, f"{w:.4f}", va='center', ha='left', fontsize=10, color='black')

ax.set_xlabel('shap_ratio (mean(|SHAP|) normalized)')
ax.set_ylabel('fea')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP-importance.jpg'), dpi=600, bbox_inches='tight')
plt.close()

# =========================
# dependence plot
# =========================
log("开始输出 dependence plots ...")
feature_names = X.columns.tolist()

shap_importance = pd.DataFrame({
    'shap': abs(shap_values2).mean(axis=0),
    'fea': feature_names
})

exclude_fea = ['CO2', 'Tmean', 'VPD', 'Rainfall', 'Tmean_mean']
shap_importance_filtered = shap_importance[~shap_importance['fea'].isin(exclude_fea)]
list11 = shap_importance_filtered.sort_values(by='shap', ascending=False).head(8).fea.tolist()

fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=120)
axes = axes.flatten()
for i, feat in enumerate(list11):
    ax = axes[i]
    shap.dependence_plot(
        feat, shap_values2, X_test,
        feature_names=feature_names,
        interaction_index=None, show=False, ax=ax
    )
for j in range(len(list11), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dependence_plot3.jpg'), dpi=600, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 4, figsize=(15, 6), dpi=120)
axes = axes.flatten()
for i, feat in enumerate(list11):
    ax = axes[i]
    shap.dependence_plot(
        feat, shap_values2, X_test,
        feature_names=feature_names,
        interaction_index='auto', show=False, ax=ax
    )
for j in range(len(list11), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dependence_plot2.jpg'), dpi=600, bbox_inches='tight')
plt.close()

# =========================
# 组合图
# =========================
log("开始输出 SHAP ...")
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Times New Roman'

def find_knee_point(x_data, y_data, window_length=5, polyorder=2):
    if len(x_data) < window_length:
        return np.median(x_data)
    if window_length % 2 == 0:
        window_length += 1
    if polyorder >= window_length:
        polyorder = window_length - 1
        if polyorder < 1:
            polyorder = 1
    y_second_deriv = savgol_filter(y_data, window_length, polyorder, deriv=2)
    knee_index = np.argmax(np.abs(y_second_deriv))
    sorted_x = np.array(x_data)[np.argsort(x_data)]
    return sorted_x[knee_index]

aesthetic_params = {
    'suptitle_size': 14,
    'ax_label_size': 12,
    'tick_label_size': 12,
    'legend_size': 12,
    'cbar_label_size': 12,
    'summary_cbar_width': 0.015,
    'summary_cbar_height_shrink': 1.0,
    'summary_cbar_pad': 0.01,
    'dep_cbar_width': 0.005,
    'dep_cbar_height_shrink': 1.0,
    'dep_cbar_pad': 0.002,
    'dep_cbar_tick_length': 1,
    'grid_wspace': 0.45,
    'grid_hspace': 0.4
}

target_column = 'y'
fig = plt.figure(figsize=(12, 6), dpi=120)

gs_main = fig.add_gridspec(1, 1, left=0.05, right=0.45, top=1, bottom=0)
ax_main = fig.add_subplot(gs_main[0, 0])

shap.summary_plot(np.array(shap_values2), X_test, feature_names=X.columns,
                  plot_type="dot", show=False, cmap='PiYG', plot_size=None)
ax_main = plt.gca()
ax_main.set_xlabel('SHAP value (impact on model output)', fontsize=12)
ax_main.tick_params(axis='both', labelsize=12)

cbar = plt.gcf().axes[-1]
pos = cbar.get_position()
cbar.set_position([pos.x0, pos.y0, pos.width * 1.5, pos.height])
cbar.set_ylabel("Feature Values", fontsize=10, rotation=90, labelpad=-30)

ax_top = ax_main.twiny()
shap.summary_plot(np.array(shap_values2), X_test, feature_names=X.columns,
                  plot_type="bar", show=False)
fig.set_size_inches(12, 6)

bars = ax_top.patches
for bar in bars:
    bar.set_alpha(0.2)

ax_top.set_xlabel('Mean(|SHAP value|) (Feature Importance)', fontsize=12)
ax_top.xaxis.set_label_position('top')
ax_top.xaxis.tick_top()
ax_top.spines['top'].set_visible(True)
ax_top.spines['top'].set_linewidth(1.2)
ax_top.spines['top'].set_color('black')
ax_top.tick_params(axis='both', labelsize=12)

mean_abs_shaps = np.abs(shap_values1.values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': mean_abs_shaps
}).sort_values('importance', ascending=True)

top_6_features = feature_importance_df['feature'].tail(6).iloc[::-1].tolist()
top_6_features = [f for f in top_6_features if f not in exclude_fea]
if len(top_6_features) < 6:
    remaining_fea = [f for f in feature_importance_df['feature'] if f not in exclude_fea + top_6_features]
    top_6_features += remaining_fea[:6 - len(top_6_features)]

gs_dep = fig.add_gridspec(3, 2, left=0.48, right=0.98, wspace=0.45, hspace=0.35, top=1, bottom=0.0)
axes_scatter = [fig.add_subplot(gs_dep[i, j]) for i in range(3) for j in range(2)]

for i, feature in enumerate(top_6_features):
    ax = axes_scatter[i]
    feature_idx = X_test.columns.get_loc(feature)
    x_data = X_test[feature].values
    y_data = shap_values1.values[:, feature_idx]
    color_data = y_test.values

    cmap = ListedColormap(["#e5f5e0", "#c7e9c0", "#74c476", "#31a354", "#006d2c"])
    bounds = np.quantile(color_data, np.linspace(0, 1, cmap.N + 1))
    bounds = np.unique(bounds)
    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)

    scatter = ax.scatter(x_data, y_data, c=color_data, cmap=cmap, norm=norm, s=30, alpha=0.9)

    fig.canvas.draw()
    ax_pos = ax.get_position()
    cax_dep_left = ax_pos.x1 + aesthetic_params['dep_cbar_pad']
    cax_dep_bottom = ax_pos.y0 + (ax_pos.height * (1 - aesthetic_params['dep_cbar_height_shrink']) / 2)
    cax_dep_width = aesthetic_params['dep_cbar_width']
    cax_dep_height = ax_pos.height * aesthetic_params['dep_cbar_height_shrink']
    cax_dep = fig.add_axes([cax_dep_left, cax_dep_bottom, cax_dep_width, cax_dep_height])

    cbar2 = fig.colorbar(scatter, cax=cax_dep)
    cbar2.ax.set_title(target_column, fontsize=10)
    cbar2.outline.set_visible(False)
    cbar2.ax.tick_params(axis='y', length=aesthetic_params['dep_cbar_tick_length'],
                         labelsize=aesthetic_params['tick_label_size'])

    ax.set_xlabel(feature, fontsize=aesthetic_params['ax_label_size'], fontweight='bold')

    median_val = np.median(x_data)
    threshold_val = find_knee_point(x_data, y_data)

    ax.axvline(median_val, color='black', linestyle='--', linewidth=1)
    ax.axvline(threshold_val, color='red', linestyle=':', linewidth=1.2)

    line_handles = [
        Line2D([0], [0], color='black', lw=1, linestyle='--', label=f'Median: {median_val:.2f}'),
        Line2D([0], [0], color='red', lw=1, linestyle=':', label=f'Threshold: {threshold_val:.2f}')
    ]
    ax.legend(handles=line_handles, loc=1, fontsize=10, frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=10)

for j in range(len(top_6_features), len(axes_scatter)):
    fig.delaxes(axes_scatter[j])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "SHAP.jpg"), dpi=600, bbox_inches='tight')
plt.close()

# =========================
# GAM 拟合
# =========================
log("开始输出 GAM 拟合图 ...")
from pygam import LinearGAM
from pygam import s as spline_s

fig, axes = plt.subplots(2, 4, figsize=(22, 10), dpi=120)
axes = axes.flatten()

for i, feat in enumerate(list11):
    ax = axes[i]
    feat_idx = X_test.columns.get_loc(feat)

    X_var = X_test[feat].values
    y_shap = shap_values2[:, feat_idx]

    gam = LinearGAM(spline_s(0)).fit(X_var.reshape(-1, 1), y_shap)

    XX = np.linspace(X_var.min(), X_var.max(), 200).reshape(-1, 1)
    y_pred = gam.predict(XX)
    conf_int = gam.confidence_intervals(XX, width=0.95)

    sc = ax.scatter(X_var, y_shap, label='SHAP values', s=8, cmap='GnBu', c=X_var)
    ax.plot(XX, y_pred, color='red', label='GAM fit')
    ax.fill_between(XX.ravel(), conf_int[:, 0], conf_int[:, 1], color='red', alpha=0.2, label='95% CI')
    plt.colorbar(sc, ax=ax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(feat, fontsize=16, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('SHAP value for ' + feat, fontsize=16, fontweight='bold', fontfamily='Times New Roman')

    ax.tick_params(axis='both', labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

for j in range(len(list11), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dependence_plot.jpg'), dpi=600, bbox_inches='tight')
plt.close()

log("\n全部输出完成，文件已保存到： " + OUTPUT_DIR)