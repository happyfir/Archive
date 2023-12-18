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
before_f1 = [0.8132, 0.8199, 0.9153, 0.9263]
after_f1 = [0.7922, 0.8011, 0.8718, 0.9111]

before_acc = [0.8577, 0.8666, 0.9352, 0.9449]
after_acc = [0.8281, 0.8499, 0.8973, 0.9344]

before_Recall = [0.7826, 0.8289, 0.8954, 0.9275]
after_Recall = [0.7233, 0.7956, 0.8181, 0.9245]

before_AUC = [0.9284, 0.9334, 0.9833, 0.9843]
after_AUC = [0.9222, 0.9205, 0.9627, 0.9681]

plt.ylim(0.60, 1.0)
total_width, n = 0.8, 4
width = total_width / n
x = x - (total_width - width) / 2

plt.plot(x,before_AUC, marker = 'o', markersize = 3)
plt.plot(x,after_AUC , marker = 'o', markersize = 3)

# 在左侧显示图例
plt.legend(['original','intervention'])
for a,b in zip(x,before_AUC):
    plt.text(a, b , b ,ha = 'center',va = 'bottom', fontsize = 10)
for a,b in zip(x,after_AUC):
    plt.text(a, b , b ,ha = 'center',va = 'top', fontsize = 10)

# 设置标题
# plt.title("Comparison of Precision and False Alarm Rate\n When Using Different Feature Schemes")
# 为两条坐标轴设置名称
plt.ylabel("Compare AUC")

aaa = ['SVM', 'AutoEncoder', 'DynamicAnalysis', 'proposed model']
bbb = range(4)
plt.xticks(bbb, aaa)

plt.savefig("Abation/robust_AUC.jpg")

plt.show()