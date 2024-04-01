# -*- coding:utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(7, 5), dpi=180)

size = 4
x = np.arange(size)
#
a = [0.9449, 0.9275, 0.9263, 0.9843]
b = [0.9529, 0.9338, 0.9363, 0.9889]


plt.ylim(0.75, 1.0)
total_width, n = 0.4, 2
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a, width=width, label='未使用Attention')
plt.bar(x + width, b, width=width, label='使用Attention')

# 在左侧显示图例
plt.legend(loc="upper left",prop = {'size':8})

# 设置标题
# plt.title("Comparison of Precision and False Alarm Rate\n When Using Different Feature Schemes")
# 为两条坐标轴设置名称
plt.ylabel("模型检测结果对比")

aaa = ['ACC', 'Recall', 'F1-score', 'AUC']
bbb = range(4)
plt.xticks(bbb, aaa)

plt.savefig("Abation/result1.jpg")

plt.show()