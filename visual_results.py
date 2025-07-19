import re
import matplotlib.pyplot as plt


def extract_metrics(log_file_path):
    # 定义正则表达式来匹配loss、mae和mse的数值
    pattern = re.compile(r'Loss: ([\d\.]+), MSE: ([\d\.]+) MAE: ([\d\.]+)')
    losses = []
    maes = []
    mses = []

    # 读取日志文件并提取数值
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                print(match)
                losses.append(float(match.group(1)))
                mses.append(float(match.group(2)))
                maes.append(float(match.group(3)))

    return losses, mses, maes


def plot_metrics(losses, maes, mses, start, end):
    losses_1 = losses[start:end]
    maes_1 = maes[start:end]
    mses_1 = mses[start:end]
    epochs = range(1, len(losses_1) + 1)

    plt.figure(figsize=(12, 6))

    # 绘制loss曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses_1, 'r', label='Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制mae曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, maes_1, 'g', label='MAE')
    plt.title('MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # 绘制mse曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, mses_1, 'b', label='MSE')
    plt.title('MSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 使用示例
log_file_path = '/data/che_xiao/crowd_count/counter/checkpoints/train.log'
losses, mses, maes = extract_metrics(log_file_path)
plot_metrics(losses, maes, mses, 1016, 1516)# 12-512

