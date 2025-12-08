# LSCARegNet


## 施工中，

有任何关于配准的问题，请私信我（B站账号：暑假作业多了）或者知乎（知乎账号：妹儿真红）或者配准萌新群滴我（QQ一群：869211738；二群：929134506）。
后期会上传所有代码，包括数据集预处理代码和处理好的数据集。

数据集预处理流程代码见
```
https://github.com/MischiefGhostOgre/LungRegistrationFromZero
```
文字教程见

```
https://www.bilibili.com/read/cv44002523/?jump_opus=1
```

参考引用
```
[1] 郭逸凡,蒋婷,彭宽宽等.LSCA-RegNet:基于线性局部窗口交叉注意力机制的可变形图像配准网络[J].四川师范大学学报（自然科学版）,2026,?(?):?-?.
```


----------------------
给自己画的图形化摘要




![image](https://github.com/MischiefGhostOgre/LSCARegNet/blob/main/images/abstract.png)



----------------------

![image](https://github.com/MischiefGhostOgre/LSCARegNet/blob/main/images/Fig3.svg)
* 图3 在三个数据集的验证集上结果
* （A） OASIS数据集 （B） IXI数据集 （C） DirLab数据集
* Fig.3 Results on validation sets of three datasets

--------------------------


![image](https://github.com/MischiefGhostOgre/LSCARegNet/blob/main/images/Fig4.svg)
* 图4 不同模型在脑部数据集上的可视化结果
* Fig.4 Visualization results of different models on brain datasets

---------------------


![image](https://github.com/MischiefGhostOgre/LSCARegNet/blob/main/images/Fig5.svg)
* 图5 不同模型在肺部数据集上的可视化结果
* Fig.5 Visualization results of different models on lung datasets

----------------------

* 表3 在LPBA据集上性能
* Tab.3 Performance results on LPBA dataset


Methods	| Avg. DSC | 	%Jet(J_ϕ) ≤0	| Cost Time
---- | ---- | ----- |----
SyN	| 0.705(0.018) | 	<0.01 |	36.8
VoxelMorph	| 	0.635(0.048)	| 	<0.01 | 	0.15
TransMorph	| 	0.671(0.023)	| 	0.37  | 	0.35
TransMatch	| 	0.643(0.026)	| 	0.28  | 	0.42
ModeT     	| 	0.696(0.024)	| 	0.03  | 	0.63
UTSRMorph	  | 	0.666(0.021)	| 	0.14  | 	0.22
Ours      	| 	0.714(0.020)	| 	<0.01 | 	0.27

----------------------

* 表5 在DirLab数据集上性能 (TRE单位：毫米)
* Tab.5 Performance results on DirLab dataset（TRE unit: mm）


Cases  | Case 7 | Case 8 | .
-------- | ----- | ------- | ---
Methods  | Avg. TRE | Avg. TRE | Cost times
Initial  | 11.03(7.40)	|	15.00(9.01) | -
SyN      | 2.00(1.20)	  |	1.76(1.38) | 15.7
VoxelMorph | 3.96(3.71)	|	7.82(6.93) | 0.10
TransMorph | 2.27(1.44)	|	3.37(2.36) | 0.23
TransMatch | 1.79(1.12)	|	3.38(2.65) | 0.24
ModeT      | 1.64(1.00)	| 2.83(2.14) | 0.44
UTSRMorph  | 4.11(2.57)	|	7.50(6.39) | 0.13
Ours       | 1.60(1.25)	|	1.65(1.34) | 0.20
