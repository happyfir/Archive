# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(7, 5), dpi=180)

size = 4
x = np.arange(size)


a = [0.9248, 0.9778, 0.8873, 0.9010]
b = [0.9206, 0.9806, 0.8547, 0.8995]
c = [0.9449, 0.9843, 0.9275, 0.9263]
d = [0.9165, 0.9771, 0.8633, 0.8922]

plt.ylim(0.8, 1.0)
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a, width=width, label='kernel = 2')
plt.bar(x + width, b, width=width, label='kernel = 2,3')
plt.bar(x + width + width, c, width=width, label='kernel = 2,3,4(proposed)')
plt.bar(x + width + width + width, d, width=width, label='kernel = 2,3,4,5')

# 在左侧显示图例
plt.legend(loc="upper right")

# 设置标题
# plt.title("Comparison of Precision and False Alarm Rate\n When Using Different Feature Schemes")
# 为两条坐标轴设置名称
plt.ylabel("Convolutional Module")

aaa = ['ACC', 'AUC', 'Recall', 'F1-score']
bbb = range(4)
plt.xticks(bbb, aaa)

plt.savefig("Abation/Convolutional Module.jpg")

plt.show()