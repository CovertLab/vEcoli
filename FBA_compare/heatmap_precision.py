import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 读取数据
num = 5
df = pd.read_csv("merged_flux_data_complete.csv")

# 检查数据类型并打印一些信息
print("Data info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("Data types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())

# 只保留有type值的flux
df_filtered = df[df["type"].notna()].copy()
print(f"\nFiltered data shape: {df_filtered.shape}")

# 检查是否有"inf"字符串
for col in df_filtered.columns:
    if "UB95" in col or "LB95" in col or "best_fit" in col:
        inf_count = (
            (df_filtered[col] == "inf").sum()
            if df_filtered[col].dtype == "object"
            else 0
        )
        print(f"Column {col}: {inf_count} 'inf' values")


# 计算precision (这里使用置信区间宽度)
def calculate_precision(best_fit, lb95, ub95):
    """计算precision，处理无穷大值"""
    # 检查是否为空值
    if pd.isna(best_fit) or pd.isna(lb95) or pd.isna(ub95):
        return np.nan

    # 转换为数值类型并处理字符串"inf"
    try:
        best_fit = float(best_fit)
        lb95 = float(lb95)
        ub95 = float(ub95)
    except (ValueError, TypeError):
        return np.nan

    # 检查是否为无穷大
    if np.isinf(ub95) or np.isinf(lb95) or np.isinf(best_fit):
        return np.nan

    return (ub95 - lb95) / 3.92


# 为每个数据集计算precision
for i in range(1, num + 1):
    best_fit_col = f"data{i}_best_fit"
    lb_col = f"data{i}_LB95"
    ub_col = f"data{i}_UB95"
    precision_col = f"data{i}_precision"

    if all(col in df_filtered.columns for col in [best_fit_col, lb_col, ub_col]):
        print(f"\nProcessing {best_fit_col}, {lb_col}, {ub_col}")
        df_filtered[precision_col] = df_filtered.apply(
            lambda row: calculate_precision(
                row[best_fit_col], row[lb_col], row[ub_col]
            ),
            axis=1,
        )
        print(
            f"Created {precision_col} with {df_filtered[precision_col].notna().sum()} valid values"
        )


# 处理flux名称的函数
def process_flux_name(flux_name):
    """
    处理flux名称：
    - 含有net的，用<->连接反应物与产物，并删除(net)
    - 不含net的，用->连接反应物与产物
    """
    if pd.isna(flux_name):
        return flux_name

    flux_str = str(flux_name)

    # 检查是否包含"net"
    if "net" in flux_str.lower():
        flux_str = flux_str.replace("=", "-")
        flux_str = flux_str.replace("(net)", "")

    return flux_str


# 处理flux名称
df_filtered["flux_processed"] = df_filtered["flux"].apply(process_flux_name)

# 显示一些处理后的flux名称示例
print("\nFlux name processing examples:")
print("=" * 50)
example_count = 0
for i, (original, processed) in enumerate(
    zip(df_filtered["flux"], df_filtered["flux_processed"])
):
    if original != processed and example_count < 5:
        print(f"Original:  {original}")
        print(f"Processed: {processed}")
        print()
        example_count += 1

# 准备热图数据
precision_cols = [f"data{i}_precision" for i in range(1, num + 1)]
existing_precision_cols = [col for col in precision_cols if col in df_filtered.columns]

print(f"\nPrecision columns found: {existing_precision_cols}")

if not existing_precision_cols:
    print("No precision columns found! Exiting...")
    exit()

# 创建热图数据矩阵，使用处理后的flux名称
heatmap_data = df_filtered[
    ["flux_processed", "type_name"] + existing_precision_cols
].copy()
heatmap_data = heatmap_data.rename(columns={"flux_processed": "flux"})

# 检查是否有valid precision值
total_valid = sum(heatmap_data[col].notna().sum() for col in existing_precision_cols)
print(f"Total valid precision values: {total_valid}")

if total_valid == 0:
    print("No valid precision values found! Exiting...")
    exit()

# 按type_name排序
type_order = df_filtered.groupby("type_name")["type"].first().sort_values().index
df_filtered["type_name"] = pd.Categorical(
    df_filtered["type_name"], categories=type_order, ordered=True
)
heatmap_data = heatmap_data.sort_values("type_name")

# 设置索引为flux名称
heatmap_data = heatmap_data.set_index("flux")

# 只保留precision列
precision_matrix = heatmap_data[existing_precision_cols]

# 重命名列
precision_matrix.columns = [
    f"Data{i}" for i in range(1, len(existing_precision_cols) + 1)
]

# 创建图形
plt.figure(figsize=(8, 20))

# 创建自定义颜色映射
colors = ["#229453", "#F4CE69", "#0F1423"]
custom_cmap = LinearSegmentedColormap.from_list("precision", colors, N=256)
# 设置缺失值为白色
custom_cmap.set_bad(color="white")

# 绘制热图
ax = sns.heatmap(
    precision_matrix,
    annot=True,
    fmt=".2f",
    cmap=custom_cmap,
    cbar_kws={"label": "Precision (CI Width)"},
    annot_kws={"size": 8},
    mask=False,
)  # 不遮罩NaN值，让colormap处理

# 设置标题和标签
plt.title(
    "Flux Precision Heatmap\n(White=Missing Data)",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
plt.xlabel("Dataset", fontsize=12)
plt.ylabel("Flux Name", fontsize=12)


# 在右侧添加type_name标签
type_positions = (
    heatmap_data.reset_index().groupby("type_name").size().cumsum()
    - heatmap_data.reset_index().groupby("type_name").size() / 2
)
type_names = heatmap_data.groupby("type_name").size().index

# 旋转x轴标签
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# 调整布局
plt.tight_layout()

plt.savefig("flux_precision_heatmap.png", dpi=300)
# 显示图形
plt.show()

# 打印一些统计信息
print("Precision Statistics:")
print("=" * 50)
for i, col in enumerate(existing_precision_cols):
    data_name = f"Data{i + 1}"
    valid_values = precision_matrix[data_name].dropna()
    if len(valid_values) > 0:
        print(f"{data_name}:")
        print(f"  Mean precision: {valid_values.mean():.3f}")
        print(f"  Median precision: {valid_values.median():.3f}")
        print(f"  Min precision: {valid_values.min():.3f}")
        print(f"  Max precision: {valid_values.max():.3f}")
        print(f"  Valid measurements: {len(valid_values)}")
        print()
