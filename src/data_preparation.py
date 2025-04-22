# src/data_preparation.py
import os
import sys
import argparse
from config import logging, OUTPUT_PATH
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import gc

# ================== 配置 ==================
PARAM_RANGES = {
    "normal": {
        "foot_length": (22.0, 27.0),
        "foot_width": (8.5, 11.0),
        "arch_height": (2.8, 3.5),
        "depth_diff": (-0.5, 0.5),
        "pressure_offset": (-1.0, 1.0)
    },
    "o_type": {
        "foot_length": (23.0, 28.0),
        "foot_width": (9.0, 12.0),
        "arch_height": (2.5, 3.2),
        "depth_diff": (0.8, 3.0),
        "pressure_offset": (-3.0, -0.5)
    },
    "x_type": {
        "foot_length": (21.0, 26.0),
        "foot_width": (8.0, 10.5),
        "arch_height": (3.0, 3.8),
        "depth_diff": (-2.5, -0.5),
        "pressure_offset": (0.5, 3.0)
    }
}

MAX_RETRIES = 50
BMI_RANGE = (16.0, 35.0)
DEFAULT_MIN_HEIGHT = 140  # 单位cm
DEFAULT_MAX_HEIGHT = 200


# ================== 辅助函数 ==================
def calculate_depth(ground_types, humidity):
    depth_base = {
        "concrete": 0.5,
        "wet_soil": 4.0,
        "dry_soil": 2.5,
        "sand": 3.0,
        "grass": 1.8
    }
    depth = np.array([depth_base[gt] for gt in ground_types])
    humidity_effect = 1 + (humidity - 50) / 200
    noise = np.random.normal(0, 0.3, len(ground_types))
    return (depth * humidity_effect + noise).round(1)


def calculate_pressure(foot_length, foot_width):
    area = foot_length * foot_width * 0.8
    base_pressure = 75 + (area - 180) * 0.5
    noise = np.random.normal(0, 5, len(foot_length))
    return np.clip(base_pressure + noise, 70, 100).round(1)


def calculate_height(foot_length, leg_type, min_height, max_height):
    base_height = foot_length * 6.5
    type_correction = {
        "normal": np.random.normal(0, 1.5),
        "o_type": np.random.normal(-2, 1.5),
        "x_type": np.random.normal(1, 1.5)
    }
    heights = base_height + type_correction[leg_type]
    return np.clip(heights, min_height, max_height).round(1)


def calculate_weight(height, foot_width, arch_height):
    bmi = np.random.normal(21.5, 1.5, len(height))
    base_weight = bmi * (height / 100) ** 2
    adjustment = foot_width * 0.3 + arch_height * 0.8
    noise = np.random.normal(0, 1.5, len(height))
    return np.clip(base_weight + adjustment + noise, 45, 90).round(1)


# ================== 分片生成与校验 ==================
def validate_chunk(chunk_df):
    """校验分片数据有效性"""
    bmi = chunk_df["weight"] / (chunk_df["height"] / 100) ** 2
    invalid = (bmi < BMI_RANGE[0]) | (bmi > BMI_RANGE[1])
    return invalid.sum() == 0


def generate_valid_chunk(chunk_size, ratios, min_height, max_height):
    """生成通过校验的有效分片"""
    for attempt in range(1, MAX_RETRIES + 1):
        chunk = generate_chunk(chunk_size, ratios, min_height, max_height)
        if validate_chunk(chunk):
            return chunk
        logging.warning(f"重试 {attempt}/{MAX_RETRIES}: 检测到异常BMI数据")
        del chunk
        gc.collect()
    raise ValueError(f"无法生成有效分片，已达最大重试次数 {MAX_RETRIES}")


def generate_chunk(chunk_size, ratios, min_height, max_height):
    """生成原始分片数据"""
    dfs = []
    remaining = chunk_size

    allocated = {}
    for i, (leg_type, ratio) in enumerate(ratios.items()):
        num = remaining if i == len(ratios) - 1 else int(chunk_size * ratio)
        allocated[leg_type] = num
        remaining -= num

    for leg_type, num in allocated.items():
        if num <= 0: continue
        df = generate_samples(num, leg_type, min_height, max_height)
        df["leg_type"] = leg_type[0].upper()
        dfs.append(df)

    chunk_df = shuffle(pd.concat(dfs, ignore_index=True), random_state=np.random.randint(0, 1000))
    chunk_df["ground_type"] = chunk_df["ground_type"].astype("category")
    chunk_df["leg_type"] = chunk_df["leg_type"].map({"N": 0, "O": 1, "X": 2})
    return chunk_df


def generate_samples(count, leg_type, min_height, max_height):
    params = PARAM_RANGES[leg_type]
    data = {
        "foot_length": np.random.uniform(*params["foot_length"], count).round(1),
        "foot_width": np.random.uniform(*params["foot_width"], count).round(1),
        "arch_height": np.random.uniform(*params["arch_height"], count).round(1),
        "depth_diff": np.random.uniform(*params["depth_diff"], count).round(1),
        "pressure_offset_x": np.random.uniform(*params["pressure_offset"], count).round(1),
        "ground_type": np.random.choice(
            ["concrete", "wet_soil", "dry_soil", "sand", "grass"],
            size=count,
            p=[0.3, 0.2, 0.2, 0.2, 0.1]
        ),
        "humidity": np.random.randint(15, 95, count)
    }
    data["depth"] = calculate_depth(data["ground_type"], data["humidity"])
    data["pressure_avg"] = calculate_pressure(data["foot_length"], data["foot_width"])
    data["height"] = calculate_height(
        data["foot_length"], leg_type,
        min_height, max_height
    ).round(1)
    data["weight"] = calculate_weight(data["height"], data["foot_width"], data["arch_height"]).round(1)
    return pd.DataFrame(data)


# ================== 主程序 ==================
def main(total_count=1000, chunk_size=100000,
         min_height=DEFAULT_MIN_HEIGHT, max_height=DEFAULT_MAX_HEIGHT):
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        ratios = {"normal": 0.6, "o_type": 0.3, "x_type": 0.1}

        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)

        num_chunks, remainder = divmod(total_count, chunk_size)
        chunks = [chunk_size] * num_chunks + ([remainder] if remainder else [])

        header = True
        for i, current_size in enumerate(chunks):
            logging.info(f"生成分片 {i + 1}/{len(chunks)} ({current_size}条)")

            # 生成并校验分片
            chunk_df = generate_valid_chunk(
                current_size, ratios,
                min_height, max_height
            )

            # 二次完整性检查
            if len(chunk_df) != current_size:
                chunk_df = chunk_df.sample(n=current_size, replace=True, random_state=42)
                logging.warning("强制修正分片大小")

            chunk_df.to_csv(
                OUTPUT_PATH,
                mode="w" if i == 0 else "a",
                header=header,
                index=False
            )
            header = False

            del chunk_df
            gc.collect()

        # 读取最终数据生成统计报告
        final_df = pd.read_csv(OUTPUT_PATH)
        logging.info("\n" + "=" * 50)
        logging.info("📊 最终数据统计报告:")

        # 基础特征统计
        stats_columns = [
            ('foot_length', '脚长(cm)', 1),
            ('foot_width', '脚宽(cm)', 1),
            ('arch_height', '足弓高(cm)', 1),
            ('depth_diff', '深度差(cm)', 1),
            ('pressure_offset_x', '压力偏移', 1),
            ('depth', '足迹深度(cm)', 1),
            ('pressure_avg', '平均压力', 0),
            ('height', '身高(cm)', 0),
            ('weight', '体重(kg)', 0),
            ('humidity', '环境湿度(%)', 0)
        ]

        for col, desc, decimals in stats_columns:
            stats = final_df[col].agg(['mean', 'min', 'max'])
            format_str = f"均值={stats['mean']:.{decimals}f} 范围({stats['min']:.{decimals}f}-{stats['max']:.{decimals}f})"
            logging.info(f"{desc:>10}: {format_str}")

        # 分类分布统计
        leg_type_dist = final_df['leg_type'].value_counts().sort_index()
        leg_type_dist.index = ['正常腿型', 'O型腿', 'X型腿']
        logging.info("\n👥 腿型分布:\n" + leg_type_dist.to_string())

        # BMI统计
        bmi = final_df["weight"] / (final_df["height"] / 100) ** 2
        valid_ratio = bmi.between(*BMI_RANGE).mean()
        logging.info(f"\n🏋️ BMI分析:")
        logging.info(f"  理论范围: {BMI_RANGE[0]:.1f}-{BMI_RANGE[1]:.1f}")
        logging.info(f"  实际范围: {bmi.min():.1f}-{bmi.max():.1f}")
        logging.info(f"  合法比例: {valid_ratio * 100:.1f}%")

        # 身高校验
        height_violation = ((final_df["height"] < min_height) |
                            (final_df["height"] > max_height)).sum()
        logging.info(f"\n📏 身高限制检查:")
        logging.info(f"  设定范围: {min_height}-{max_height}cm")
        logging.info(f"  实际范围: {final_df['height'].min()}-{final_df['height'].max()}cm")
        logging.info(f"  违规数量: {height_violation}条")

        # 文件信息
        logging.info("\n💾 存储信息:")
        logging.info(f"文件路径: {OUTPUT_PATH}")
        logging.info(f"总数据量: {len(final_df):,} 条")
        logging.info(f"文件大小: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.2f} MB")
        logging.info("=" * 50 + "\n")

    except Exception as e:
        logging.error(f"‼️ 数据生成失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="脚印数据生成器")
    parser.add_argument("--count", type=int, default=1000,
                        help="总样本数量（默认：1000）")
    parser.add_argument("--chunk_size", type=int, default=500000,
                        help="分片大小（默认：500000）")
    parser.add_argument("--min_height", type=int, default=DEFAULT_MIN_HEIGHT,
                        help=f"最低身高限制(cm)（默认：{DEFAULT_MIN_HEIGHT}）")
    parser.add_argument("--max_height", type=int, default=DEFAULT_MAX_HEIGHT,
                        help=f"最高身高限制(cm)（默认：{DEFAULT_MAX_HEIGHT}）")
    args = parser.parse_args()

    main(
        total_count=args.count,
        chunk_size=args.chunk_size,
        min_height=args.min_height,
        max_height=args.max_height
    )
