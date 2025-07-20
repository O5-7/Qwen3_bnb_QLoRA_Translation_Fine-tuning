import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # 高斯平滑

# 生成示例数据（假设已经记录了训练过程的loss）
# np.random.seed(42)
# raw_loss = np.random.randn(200) * 0.3 + np.linspace(2, 0.1, 200)

raw_loss = np.array([float(d[:-1]) for d in open("./loss_log.txt",mode='r').readlines()])

steps = np.arange(len(raw_loss))


# 平滑处理函数
def smooth_curve(loss, window_size=100, sigma=200):
    """
    带边界处理的平滑函数
    :param loss: 原始loss序列
    :param window_size: 滑动窗口大小（奇数）
    :param sigma: 高斯滤波标准差
    :return: (平滑曲线, 方差范围)
    """
    # 高斯平滑
    smoothed = gaussian_filter1d(loss, sigma=sigma)

    # 计算滑动窗口方差
    pad = window_size // 2
    padded = np.pad(loss, pad, mode="edge")  # 边界填充
    variances = []
    for i in range(len(loss)):
        window = padded[i : i + window_size]
        variances.append(np.std(window))

    return smoothed, np.array(variances)


# 获取平滑曲线和方差
smoothed_loss, std_dev = smooth_curve(raw_loss)

# 创建画布
plt.figure(figsize=(12, 6), dpi=100)
plt.grid(True, linestyle="--", alpha=0.6)
plt.title("Training Loss Curve with Confidence Band", fontsize=14, pad=20)

# 绘制原始loss（半透明显示）
plt.scatter(steps, raw_loss, color="royalblue", alpha=0.15, label="Raw Loss")

# 绘制平滑曲线
line = plt.plot(steps, smoothed_loss, linewidth=2.5, color="darkorange", label="Smoothed Loss")

# 绘制方差范围带
plt.fill_between(steps, smoothed_loss - std_dev, smoothed_loss + std_dev, color="gold", alpha=0.25, label="±1 Std Dev")

# 图例和坐标轴设置
plt.legend(loc="upper right", framealpha=0.95)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Loss Value", fontsize=12)
plt.ylim(bottom=0)  # 确保y轴从0开始

# 显示重要统计信息
stats_text = f"""
Final Loss: {smoothed_loss[-1]:.4f}
Min Loss: {smoothed_loss.min():.5} (±{std_dev[smoothed_loss.argmin()]:.4f})
"""
plt.annotate(stats_text, xy=(0.72, 0.7), xycoords="axes fraction", fontfamily="monospace", bbox=dict(boxstyle="round", alpha=0.2, color="gray"))

# 优化布局
plt.tight_layout()
plt.show()
