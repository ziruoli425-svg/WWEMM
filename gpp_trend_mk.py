import os
import numpy as np
import rasterio
from scipy.stats import theilslopes
import pymannkendall as mk
from tqdm import tqdm

# ================= 参数配置 =================
data_dir = r"E:\data\qingban\GPP"  # 数据输入输出路径
years = range(2000, 2025)
months = ["310_331", "1015_1130", "Feb10_Mar10"]  # 研究时段

years_array = np.array(list(years), dtype=np.float32)
min_valid_years = 10  # 统计学阈值：进入趋势分析的最小有效年份数

# ================= 数据处理 =================
for month in months:
    print(f"\n[Info] 开始处理时段: {month}")

    # 获取空间元数据
    sample_file = os.path.join(data_dir, f"MODIS_GPP_{years[0]}_{month}.tif")
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"缺失参考文件: {sample_file}")

    with rasterio.open(sample_file) as src:
        height, width = src.read(1).shape
        out_meta = src.meta.copy()

    # ================= 1. 构建时间序列数据立方体 =================
    ts_array = np.zeros((len(years), height, width), dtype=np.float32)
    print(f"[Info] 正在读取影像并构建时序矩阵...")

    for i, year in enumerate(years):
        file_path = os.path.join(data_dir, f"MODIS_GPP_{year}_{month}.tif")
        if not os.path.exists(file_path):
            ts_array[i, :, :] = np.nan
            continue

        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)

            # 清洗无效值 (保留 GPP=0 的真实生态意义，仅剔除负值或特定异常值)
            data[data < 0] = np.nan
            # data[data >= 32000] = np.nan  # 如有特定的水体/云遮挡 fill_value，取消此注释

            ts_array[i, :, :] = data

    # ================= 2. 像元级趋势与显著性检验 =================
    slope_map = np.full((height, width), np.nan, dtype=np.float32)
    mk_p_map = np.full((height, width), np.nan, dtype=np.float32)
    mk_z_map = np.full((height, width), np.nan, dtype=np.float32)

    print(f"[Info] 逐像元计算 Theil-Sen Median 趋势与 Mann-Kendall 显著性...")
    for row in tqdm(range(height), desc="Processing Spatial Grids"):
        for col in range(width):
            y = ts_array[:, row, col]

            valid_mask = ~np.isnan(y)
            valid_count = np.sum(valid_mask)

            # 过滤样本量不足的像元
            if valid_count < min_valid_years:
                continue

            try:
                y_valid = y[valid_mask]
                x_valid = years_array[valid_mask]

                # Theil-Sen 趋势计算
                slope, _, _, _ = theilslopes(y_valid, x_valid)
                slope_map[row, col] = slope

                # Mann-Kendall 显著性检验 (基于 Yue-Wang 2004 消除一阶自相关)
                mk_result = mk.yue_wang_modification_test(y_valid)
                mk_p_map[row, col] = mk_result.p
                mk_z_map[row, col] = mk_result.z

            except Exception:
                # 忽略数值计算中极少数可能出现的矩阵奇异或收敛失败
                continue

    # ================= 3. 结果输出 =================
    out_meta.update({"dtype": "float32"})

    slope_file = os.path.join(data_dir, f"GPP_{month}_TheilSen_slope.tif")
    with rasterio.open(slope_file, "w", **out_meta) as dst:
        dst.write(slope_map, 1)

    mk_file = os.path.join(data_dir, f"GPP_{month}_MK_p.tif")
    with rasterio.open(mk_file, "w", **out_meta) as dst:
        dst.write(mk_p_map, 1)

    z_file = os.path.join(data_dir, f"GPP_{month}_MK_z.tif")
    with rasterio.open(z_file, "w", **out_meta) as dst:
        dst.write(mk_z_map, 1)

print("\n[Info] 所有时段数据处理及空间统计输出完毕。")