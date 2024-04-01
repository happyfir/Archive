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

a = [0.9214, 0.9724, 0.8586, 0.9001]
b = [0.9292, 0.9750, 0.9084, 0.9050]
c = [0.9449, 0.9843, 0.9275, 0.9263]
d = [0.9208, 0.9729, 0.8772, 0.8965]

plt.ylim(0.8, 1.0)
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a, width=width, label='无自编码器')
plt.bar(x + width, b, width=width, label='自编码器-2 密集层')
plt.bar(x + width + width, c, width=width, label='自编码器-3 密集层(Mal-CLAM)')
plt.bar(x + width + width + width, d, width=width, label='自编码器-4 密集层')

# 在左侧显示图例
plt.legend(loc="upper right",prop = {'size':9})

# 设置标题
# plt.title("Comparison of Precision and False Alarm Rate\n When Using Different Feature Schemes")
# 为两条坐标轴设置名称
plt.ylabel("自编码器模块消融实验")

aaa = ['ACC', 'AUC', 'Recall', 'F1-score']
bbb = range(4)
plt.xticks(bbb, aaa)

plt.savefig("Abation/AutoEncoder Module.jpg")

plt.show()