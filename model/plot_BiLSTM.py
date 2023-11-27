# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(7, 5), dpi=180)

size = 4
x = np.arange(size)

# a = [0.9353, 0.9804, 0.9209, 0.9128]
b = [0.9149, 0.9748, 0.9198, 0.8816]
c = [0.9449, 0.9843, 0.9275, 0.9263]
d = [0.9241, 0.9734, 0.9734, 0.8953]

plt.ylim(0.8, 1.0)
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 2

# plt.bar(x, a, width=width, label='LSTM')
plt.bar(x + width, b, width=width, label='None Bi-LSTM')
plt.bar(x + width + width, c, width=width, label='One Bi-LSTM(proposed)')
plt.bar(x + width + width + width, d, width=width, label='Two Bi-LSTM')

# 在左侧显示图例
plt.legend(loc="upper right")

# 设置标题
# plt.title("Comparison of Precision and False Alarm Rate\n When Using Different Feature Schemes")
# 为两条坐标轴设置名称
plt.ylabel("Bi-LSTM Module")

aaa = ['ACC', 'AUC', 'Recall', 'F1-score']
bbb = range(4)
plt.xticks(bbb, aaa)

plt.savefig("Abation/Bi-LSTMl Module.jpg")

plt.show()