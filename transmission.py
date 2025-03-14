import numpy as np
import time
import cupy as cp
import matplotlib.pyplot as plt
import sys
import pandas as pd

def test():
    # 初始化结果列表
    direct_times = []
    compressed_times = []
    n_tokens_list = list(range(1000, 15001, 500))  # 从2000到16000，每隔400

    # 参数设置
    n_heads = 8
    head_dim = 128
    repeat_times = 50

    # 测试不同n_tokens下的传输时间
    results = []
    for n_tokens in n_tokens_list:
        data_size = n_tokens * n_heads * head_dim * 2  # FP16的字节数

        # 生成随机数据
        data = np.random.randn(n_tokens, n_heads, head_dim).astype(np.float16)
        print(f"Data size for {n_tokens} tokens: {sys.getsizeof(data)} bytes")

        # 计算压缩参数
        b = 8  # 压缩为8-bit整数
        max_val = np.max(data)
        min_val = np.min(data)
        lambda_ = (max_val - min_val) / (2**b - 1)
        z = round((-2**b) / (max_val - min_val))

        # 压缩数据
        def compress(data, lambda_, z):
            return np.round(data / lambda_ + z).astype(np.int8)

        # 解压缩数据
        def decompress(data_q, lambda_, z):
            return lambda_ * (data_q - z).astype(np.float16)

        # 直接传输时间
        direct_time_total = 0
        for _ in range(repeat_times):
            start_time = time.perf_counter()
            gpu_data_direct = cp.asarray(data)  # CPU到GPU
            cp.asnumpy(gpu_data_direct)  # GPU到CPU
            direct_time_total += time.perf_counter() - start_time
        direct_time = direct_time_total / repeat_times

        # 压缩后传输时间
        data_q_packed = compress(data, lambda_, z)  # 压缩
        compressed_time_total = 0
        for _ in range(repeat_times):
            start_time = time.perf_counter()
            gpu_data_q_packed = cp.asarray(data_q_packed)  # CPU到GPU
            cp.asnumpy(gpu_data_q_packed)  # GPU到CPU
            data_decompressed = decompress(gpu_data_q_packed, lambda_, z)  # 解压缩
            compressed_time_total += time.perf_counter() - start_time
        compressed_time = compressed_time_total / repeat_times

        # 保存结果
        results.append([n_tokens, direct_time, compressed_time])
        print(f"Packed data size for {n_tokens} tokens: {sys.getsizeof(data_q_packed)} bytes")

    # 保存结果到CSV文件
    df = pd.DataFrame(results, columns=['n_tokens', 'direct_time', 'compressed_time'])
    df.to_csv('transfer_time_results_gqa.csv', index=False)

test()

# 读取CSV文件
df = pd.read_csv('transfer_time_results_gqa.csv')

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 设置图的大小，长宽比为2.8
width = 10
height = width / 2.8
plt.figure(figsize=(width, height))

# 将传输时间转换为毫秒
df['direct_time_ms'] = df['direct_time'] * 1000
df['compressed_time_ms'] = df['compressed_time'] * 1000

# 绘制折线图，设置线段粗细为2
plt.plot(df['n_tokens'], df['direct_time_ms'], label='Direct Transfer', marker = 'o', linewidth=4.5, color = "#00008B")
plt.plot(df['n_tokens'], df['compressed_time_ms'], label='Compressed Transfer', marker = 'x', linewidth=4.5, color = "crimson")

# 设置坐标轴标签和图例的字体大小
plt.xlabel('Number of Tokens', fontsize=16)
plt.ylabel('Transfer Time with Compress (ms)', fontsize=16)
plt.title('Transfer Time Comparison', fontsize=18)

# 设置图例的字体大小
plt.legend(fontsize=16)

# 设置网格线
plt.grid(True)

# 设置横纵坐标刻度的大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=20)

# 保存图表为图片文件
plt.savefig('transfer_time_comparison_gqa.pdf', bbox_inches='tight')

# 显示图表
plt.show()