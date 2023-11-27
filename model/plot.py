# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(7, 5), dpi=180)

size = 4
x = np.arange(size)

a = [0.8577, 0.9295, 0.7971, 0.8139]
b = [0.8666, 0.8289, 0.8199, 0.9344]
c = [0.9352, 0.8954, 0.9153, 0.9833]
d = [0.9449, 0.9843, 0.9275, 0.9863]

plt.ylim(0.75, 1.0)
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a, width=width, label='SVM')
plt.bar(x + width, b, width=width, label='AutoEncoder')
plt.bar(x + width + width, c, width=width, label='DynamicAnalysis')
plt.bar(x + width + width + width, d, width=width, label='proposed model')

# 在左侧显示图例
plt.legend(loc="upper left",prop = {'size':8})

# 设置标题
# plt.title("Comparison of Precision and False Alarm Rate\n When Using Different Feature Schemes")
# 为两条坐标轴设置名称
plt.ylabel("Compare")

aaa = ['ACC', 'AUC', 'Recall', 'F1-score']
bbb = range(4)
plt.xticks(bbb, aaa)

plt.savefig("Abation/compare.jpg")

plt.show()