#!/usr/bin/env python
# coding: utf-8

# In[99]:


# coding:utf-8

# 【 项目：深圳市二手房房价分析及预测 】

# --------------------------------【 数据理解 】--------------------------------------

# (一) 导入数据 并进行初步的数据观察

# 1.改变工作目录到 数据集所在文件夹

import os
import pandas as pd

pd.set_option('display.max_columns', None)

data_path = r'D:\learning\project_data\Data_of_House_Price_ShenZhen'
save_path = r'D:\learning\project_data'

# 改变当前工作目录到指定的路径
os.chdir(data_path)
loadfile = os.listdir()  # 读取目录下所有文件
df = pd.DataFrame()

for i in loadfile:
    #   locals() 没有返回局部名字空间，返回拷贝。修改的是拷贝，实际变量值无影响。
    locals()[i.split('.')[0]] = pd.read_excel(i)
    df = pd.concat([df, locals()[i.split('.')[0]]], ignore_index=True)
df = df.rename(columns={'Unnamed: 0': 'houseid'})
print('深圳各区二手房房价信息合并完成')
print('=' * 100)
print(df.head())
print(df.count())

# In[105]:

# 2.查看数据情况

# # 1) 数据总体情况
print('数据集中的样本量共有%d个' % df.shape[0])  # 行数
print(df.duplicated())  # 是否有重复
print(df.head())
print(df.describe(include='all').T)  # 查看描述基本信息
del df['houseid']
del df['floor_num']
# df.drop_duplicates() 删除重复行
df['hall'] = df['hall'].astype('int64')  # 改变数据类型
df.info()  # 查看数据类型是否有误

# In[107]:


# # 2) 查看分类变量的取值
import sqlite3

con = sqlite3.connect(':memory:')
df.to_sql('df', con)
print(list(df))
print('查看分类变量的取值')
for i in list(df[['district', 'roomnum', 'hall', 'C_floor', 'school', 'subway']]):
    s = pd.read_sql_query('SELECT DISTINCT %s from df' % i, con)
    print(s)
    print('-' * 12)
print('=' * 100)

# In[110]:


# # 3) 将城区名由拼音改成中文，方便之后作图时查看
dist_dict = {'longgang': '龙岗', 'longhua': '龙华', 'baoan': '宝安', 'nanshan': '南山', 'futian': '福田', 'luohu': '罗湖',
             'pingshan': '坪山', 'guangming': '光明', 'yantian': '盐田'}
df['district'] = df['district'].apply(lambda x: dist_dict[x])
# # 3) 将楼层名变为英文，方便之后作图时查看
C_floor = {' 低层': 'low', ' 中层': 'middle', ' 高层': 'high'}
df['C_floor'] = df['C_floor'].apply(lambda x: C_floor[x])
print(df.head())
print('=' * 100)

# In[5]:


# # 4) 生成一个excel表格用于分析列名含义
table_columns = pd.DataFrame(columns=['表名', '表说明', '列名', '列名含义', '备注'])
table_columns['列名'] = list(df)
table_columns.to_excel(save_path + r'\列名含义分析表.xls')
print('列名含义分析表 已生成')
print('=' * 100)  # 划分割线，方便查看

# In[112]:


# --------------------------------【 目标拆解分析 】--------------------------------------

# (二) 因变量分析

# # 1. 单位面积房价分析

# # # 设置负号显示及字体
from matplotlib import pylab

pylab.rcParams['font.sans-serif'] = ['SimHei']
pylab.rcParams['axes.unicode_minus'] = False

# # # 1) 作因变量直方图
import matplotlib.pyplot as plt

df.per_price.hist(grid=False, bins=20, color='lightblue')
plt.title('单位面积房价频数图')
plt.xlabel('单位面积房价（万元/平方米）')
plt.ylabel('频数')
plt.show()
'''
通过直方图可发现数据是偏态分布的（右偏 → 那么之后可能会取对数）
'''

# In[113]:


# # # 2) 查看 均值、中位数、标准差、四分位数
print(df.per_price.agg(['mean', 'median', 'std']))
print(df.per_price.quantile([0.25, 0.5, 0.75]))

# In[114]:


print('max:%s万元/㎡' % max(df.per_price))
print('min:%s万元/㎡' % min(df.per_price))

# In[125]:


# (三) 自变量

# # 1. 整体来看
# # # 1) 分类变量 看 各个取值 的 数量统计 情况
for i in range(len(list(df))):
    if i != list(df).index('AREA') and i != list(df).index('per_price'):
        print(df.columns.values[i], ':')
        print(df[df.columns.values[i]].value_counts())
    else:
        continue

'''
dist列中 每个区的样本量均在300以上，该数据只爬取了每区前10页，所有样本量差不多;
roomnum列中 最多的类型是3房，6~9房的占比很少;
halls列中 最普遍的是2个;
floor列中 middle的数量多一些，high和low基本持平;
subway列中 有地铁多一点;
school列中 学区房更多一些
'''

# In[126]:


# # # 2) 连续变量 看 最小、最大值、平均数、中位数、标准差
print('AREA:')
print(df.AREA.agg(['min', 'mean', 'median', 'max', 'std']).T)

# In[127]:


# # 2. 分开各个量 → district列

# # # 各个区的房屋信息数量情况比较，饼图
colors = ['#71ae46', '#96b744', '#c4cc38', '#ebe12a', '#eab026', '#e3852b', '#d85d2a', '#ce2626', '#ac2026', '#71ae46']
df.district.value_counts().sort_values().plot(kind='pie', autopct="%1.1f%%", colors=colors)
plt.title('各个区的房屋数量情况比较')
plt.show()
print(df.district.value_counts().sort_values(ascending=False))

# In[128]:


# # # 各个区房价均值比较
colors = ['#71ae46', '#71ae46', '#96b744', '#c4cc38', '#ebe12a', '#eab026', '#e3852b', '#d85d2a', '#ce2626', '#ac2026']
df.per_price.groupby(df.district).mean().sort_values().plot(kind='barh', color=colors)
plt.xlabel('各个区房价均值(万元/平方米)')
plt.ylabel('区名')
plt.title('各个区的房价均值情况表')
A = df.per_price.groupby(df.district).mean().sort_values()
for i in range(len(list(df.district.value_counts()))):
    plt.text(A[i], i, round(A[i], 4), fontsize=12, verticalalignment="center", horizontalalignment="right")
plt.show()

# In[129]:


# # # 不同城区的单位面积房价（箱线图）
import seaborn as sns

df_temp1 = df[['district', 'per_price']].sort_values(by=['district'])
df_temp1['district'] = df_temp1['district'].astype('category')
df_temp1['district'] = df_temp1['district'].cat.set_categories(['坪山', '龙岗', '光明', '盐田', '龙华', '罗湖', '宝安', '福田', '南山'])
plt.figure(figsize=(12, 6))
sns.boxplot(x='district', y='per_price', data=df_temp1, linewidth=0.5,
            palette=sns.cubehelix_palette(16, start=2, rot=2, dark=0, light=.95))
plt.ylabel('单位面积房价（万元/㎡）')
plt.xlabel('城区')
plt.title('不同城区的单位面积房价的分组箱线图')
plt.show()
'''
由箱线图可知，随着x值的不同，中心水平是有变化的，故可初步判断二者是有关系的
'''

# In[131]:


# # 3. 分开各个量 → roomnum列

# # # 不同卧室数的单位面积房价
df.per_price.groupby(df.roomnum).mean().plot(kind='bar', color=colors)
plt.xticks(rotation=360)
plt.title('不同卧室数的单位面积房价-直方图')
plt.show()

df_temp2 = df[['roomnum', 'per_price']]
# df_temp2.boxplot(by='roomnum',patch_artist=True)   # patch_artist上下四分位框内是否填充,True为填充
plt.figure(figsize=(12, 6))
sns.boxplot(x=df_temp2['roomnum'], y=df_temp2['per_price'],
            palette=sns.cubehelix_palette(16, start=2, rot=0, dark=0, light=.95))
plt.title('不同卧室数的单位面积房价-箱线图')
plt.show()

'''
直方图中roomnum=9的房价较大，考虑到数量较少，个别情况影响比较大
'''

# In[133]:


# # 4. 分开各个量 → hall列

# # # 不同厅数的单位面积房价
df.per_price.groupby(df.hall).mean().plot(kind='bar', color=colors)
plt.xticks(rotation=360)
plt.title('不同厅数的单位面积房价-直方图')
plt.show()

df_temp3 = df[['hall', 'per_price']]
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_temp3['hall'], y=df_temp3['per_price'],
            palette=sns.cubehelix_palette(10, start=2, rot=0, dark=0, light=.95))
plt.title('不同厅数的单位面积房价-箱线图')
plt.show()

# In[134]:


# # 5. 分开各个量 → C_floor列
df_temp4 = df[['C_floor', 'per_price']]
sns.boxplot(x=df_temp4['C_floor'], y=df_temp4['per_price'],
            palette=sns.cubehelix_palette(6, start=2, rot=0, dark=0, light=.95))
plt.title('不同楼层的单位面积房价-箱线图')
plt.show()
'''
不同卧室数的单位面积房价差异不大;
不同厅数的单位面积房价有一定影响;
不同楼层的单位面积房价差异不明显
'''

# In[135]:


# 6.分开各个量 → subway

# # # 是否临近地铁对单位面积房价的影响
df_temp5 = df[['subway', 'per_price']]
sns.boxplot(x=df_temp5['subway'], y=df_temp5['per_price'],
            palette=sns.cubehelix_palette(4, start=2, rot=0, dark=0, light=.95))
plt.title('是否临近地铁对单位面积房价的影响-箱线图')
plt.show()

# In[136]:


# # 7.分开各个量 → school

# # # 是否是学区房对房价的影响
df_temp6 = df[['school', 'per_price']]
sns.boxplot(x=df_temp6['school'], y=df_temp6['per_price'],
            palette=sns.cubehelix_palette(4, start=2, rot=0, dark=0, light=.95))
plt.title('是否是学区房对房价的影响-箱线图')
plt.show()


# In[137]:


# # stack2dim是自定义的标准化的堆叠柱形图库包，宽度表示数量
def stack2dim(raw, i, j, rotation=0, location='upper left'):
    '''
    此函数是为了画两个维度标准化的堆积柱状图
    要求是目标变量j是二分类的
    raw为pandas的DataFrame数据框
    i、j为两个分类变量的变量名称，要求带引号，比如"school"
    rotation：水平标签旋转角度，默认水平方向，如标签过长，可设置一定角度，比如设置rotation = 40
    location：分类标签的位置，如果被主体图形挡住，可更改为'upper left'

    '''
    import math
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    data_raw = pd.crosstab(raw[i], raw[j])
    data = data_raw.div(data_raw.sum(1), axis=0)  # 交叉表转换成比率，为得到标准化堆积柱状图

    # 计算x坐标，及bar宽度
    createVar = locals()
    x = [0]  # 每个bar的中心x轴坐标
    width = []  # bar的宽度
    k = 0
    for n in range(len(data)):
        # 根据频数计算每一列bar的宽度
        createVar['width' + str(n)] = data_raw.sum(axis=1)[n] / sum(data_raw.sum(axis=1))
        width.append(createVar['width' + str(n)])
        if n == 0:
            continue
        else:
            k += createVar['width' + str(n - 1)] / 2 + createVar['width' + str(n)] / 2 + 0.05
            x.append(k)

            # 以下是通过频率交叉表矩阵生成一列对应堆积图每一块位置数据的数组，再把数组转化为矩阵
    y_mat = []
    n = 0
    for p in range(data.shape[0]):
        for q in range(data.shape[1]):
            n += 1
            y_mat.append(data.iloc[p, q])
            if n == data.shape[0] * 2:
                break
            elif n % 2 == 1:
                y_mat.extend([0] * (len(data) - 1))
            elif n % 2 == 0:
                y_mat.extend([0] * len(data))

    y_mat = np.array(y_mat).reshape(len(data) * 2, len(data))
    y_mat = pd.DataFrame(y_mat)  # bar图中的y变量矩阵，每一行是一个y变量

    # 通过x，y_mat中的每一行y，依次绘制每一块堆积图中的每一块图
    createVar = locals()
    for row in range(len(y_mat)):
        createVar['a' + str(row)] = y_mat.iloc[row, :]
        if row % 2 == 0:
            if math.floor(row / 2) == 0:
                label = data.columns.name + ': ' + str(data.columns[row])
                plt.bar(x, createVar['a' + str(row)],
                        width=width[math.floor(row / 2)], label='0', color='#5F9EA0')
            else:
                plt.bar(x, createVar['a' + str(row)],
                        width=width[math.floor(row / 2)], color='#5F9EA0')
        elif row % 2 == 1:
            if math.floor(row / 2) == 0:
                label = data.columns.name + ': ' + str(data.columns[row])
                plt.bar(x, createVar['a' + str(row)], bottom=createVar['a' + str(row - 1)],
                        width=width[math.floor(row / 2)], label='1', color='#8FBC8F')
            else:
                plt.bar(x, createVar['a' + str(row)], bottom=createVar['a' + str(row - 1)],
                        width=width[math.floor(row / 2)], color='#8FBC8F')

    plt.title(j + ' vs ' + i)
    group_labels = [data.index.name + ': ' + str(name) for name in data.index]
    plt.xticks(x, group_labels, rotation=rotation)
    plt.ylabel(j)
    plt.legend(shadow=True, loc=location)
    plt.show()


# In[138]:


# # # subway+school 地铁 + 学区
print(pd.crosstab(df.subway, df.school))
sub_shc = pd.crosstab(df.subway, df.school)
sub_shc = sub_shc.div(sub_shc.sum(1), axis=0)
print('-' * 20)
print(sub_shc)
print('-' * 20)
# # # subway+school 的 两维度分组堆积图
# from stack2dim import *
stack2dim(df, i='subway', j='school')

'''
是否临近地铁和是否时学区房对房价有一定影响；
地铁房中的学区房比重 比 非地铁房中的学区比重更大
'''

# In[139]:


# # 8.面积 AREA x,y均为连续变量，画散点图
df_temp7 = df[['AREA', 'per_price']]
plt.scatter(df_temp7.AREA, df_temp7.per_price, marker='.',
            color='#00686b')
plt.title('面积AREA 和 单位面积房价per_price 的 散点图')
plt.ylabel('单位面积房价')
plt.xlabel('面积（㎡）')
plt.show()

# In[140]:


# # # 求AREA和per_price的相关关系矩阵
import numpy as np

df_temp_A = np.array(df_temp7['per_price'])
df_temp_B = np.array(df_temp7['AREA'])
df_temp_AB = np.array([df_temp_A, df_temp_B])
print(np.corrcoef(df_temp_AB))
'''
从散点图可看出，点由左向右逐渐稀疏，考虑对y取对数后再作散点图
'''

# In[141]:


# # # 面积AREA 和 单位面积房价per_price(取对数后) 的 散点图
A = df_temp7.copy()
A['per_price_ln'] = A['per_price'].apply(lambda x: np.log(x))
plt.scatter(A.AREA, A.per_price_ln, marker='.', color='#00686b')
plt.title('面积AREA 和 单位面积房价per_price(取对数后) 的 散点图')
plt.ylabel("单位面积房价（取对数后）")
plt.xlabel("面积（平方米）")
plt.show()

# In[142]:


df_temp_C = np.array(A['per_price_ln'])
df_temp_D = np.array(A['AREA'])
df_temp_CD = np.array([df_temp_C, df_temp_D])
print(np.corrcoef(df_temp_CD))  # corrcoef相关系数
'''
依然呈现明显的右边集中特点，考虑对x变量也取对数
'''

# In[143]:


# # # 面积AREA(取对数后) 和 单位面积房价per_price(取对数后) 的 散点图
B = df_temp7.copy()
B['per_price_ln'] = np.log(B['per_price'])
B['AREA_ln'] = np.log(B['AREA'])
plt.scatter(B.per_price_ln, B.AREA_ln, marker='.', color='#00686b')
plt.title('面积AREA(取对数后) 和 单位面积房价per_price(取对数后) 的 散点图')
plt.ylabel("单位面积房价（取对数后）")
plt.xlabel("面积（平方米）（取对数后）")
plt.show()

# In[144]:


df_temp_E = np.array(B['per_price_ln'])
df_temp_F = np.array(B['AREA_ln'])
df_temp_EF = np.array([df_temp_E, df_temp_F])
print(np.corrcoef(df_temp_EF))
'''
两者都取对数后，分布较为正态了，故之后建模时，应该二者都取对数
'''


# In[145]:


# get_sample代码
# coding:utf-8
def get_sample(df, sampling="simple_random", k=1, stratified_col=None):
    """
    对输入的 dataframe 进行抽样的函数

    参数:
        - df: 输入的数据框 pandas.dataframe 对象

        - sampling:抽样方法 str
            可选值有 ["simple_random", "stratified", "systematic"]
            按顺序分别为: 简单随机抽样、分层抽样、系统抽样

        - k: 抽样个数或抽样比例 int or float
            (int, 则必须大于0; float, 则必须在区间(0,1)中)
            如果 0 < k < 1 , 则 k 表示抽样对于总体的比例
            如果 k >= 1 , 则 k 表示抽样的个数；当为分层抽样时，代表每层的样本量

        - stratified_col: 需要分层的列名的列表 list
            只有在分层抽样时才生效

    返回值:
        pandas.dataframe 对象, 抽样结果
    """
    import random
    import pandas as pd
    from functools import reduce
    import numpy as np
    import math

    len_df = len(df)
    if k <= 0:
        raise AssertionError("k不能为负数")
    elif k >= 1:
        assert isinstance(k, int), "选择抽样个数时, k必须为正整数"
        sample_by_n = True
        if sampling is "stratified":
            alln = k * df.groupby(by=stratified_col)[stratified_col[0]].count().count()  # 有问题的
            # alln=k*df[stratified_col].value_counts().count()
            if alln >= len_df:
                raise AssertionError("请确认k乘以层数不能超过总样本量")
    else:
        sample_by_n = False
        if sampling in ("simple_random", "systematic"):
            k = math.ceil(len_df * k)

    # print(k)

    if sampling is "simple_random":
        print("使用简单随机抽样")
        idx = random.sample(range(len_df), k)
        res_df = df.iloc[idx, :].copy()
        return res_df

    elif sampling is "systematic":
        print("使用系统抽样")
        step = len_df // k + 1  # step=len_df//k-1
        start = 0  # start=0
        idx = range(len_df)[start::step]  # idx=range(len_df+1)[start::step]
        res_df = df.iloc[idx, :].copy()
        # print("k=%d,step=%d,idx=%d"%(k,step,len(idx)))
        return res_df

    elif sampling is "stratified":
        assert stratified_col is not None, "请传入包含需要分层的列名的列表"
        assert all(np.in1d(stratified_col, df.columns)), "请检查输入的列名"

        grouped = df.groupby(by=stratified_col)[stratified_col[0]].count()
        if sample_by_n == True:
            group_k = grouped.map(lambda x: k)
        else:
            group_k = grouped.map(lambda x: math.ceil(x * k))

        res_df = df.head(0)
        for df_idx in group_k.index:
            df1 = df
            if len(stratified_col) == 1:
                df1 = df1[df1[stratified_col[0]] == df_idx]
            else:
                for i in range(len(df_idx)):
                    df1 = df1[df1[stratified_col[i]] == df_idx[i]]
            idx = random.sample(range(len(df1)), group_k[df_idx])
            group_df = df1.iloc[idx, :].copy()
            res_df = res_df.append(group_df)
        return res_df

    else:
        raise AssertionError("sampling is illegal")


# In[146]:


# --------------------------------【 建立房价预测模型 】--------------------------------------

# (一) 检验变量 → 每个解释变量是否和被解释变量独立（完全不独立）
# # 1.进行抽样
'''
原始样本量太大，若要使用基于P值的模型构建，需进行抽样，考虑本数据分十个不同的区，选择分层抽样
'''
data = get_sample(df, sampling='stratified', k=250, stratified_col=['district'])
# # 2.检验各变量的解释力度
'''
前面分析得到的结论是：
不同卧室数的单位面积房价差异不大;
不同厅数的单位面积房价有一定影响;
不同楼层的单位面积房价差异不明显；
是否临近地铁和是否学区房对房价有一定影响；
'''
# statsmodels模型未知的情况下来检验模型的线性显著性
import statsmodels.api as sm
from statsmodels.formula.api import ols

l = ['district', 'roomnum', 'hall', 'school', 'subway']
for i in l:
    print('%s 的P值为：%.4f' % (i, sm.stats.anova_lm(ols('per_price~C(%s)' % i, data=data).fit())._values[0][4]))
'''
以上P值均为零
原假设是随机变量之间相互独立，
当前检验为 p 值为 0，说明 p 值很小，即落在了拒绝域，从而说明你检验的两个变量完全不独立，就是有很强的相关关系
'''

# In[147]:


# # # 对于hall列，由之前的柱状图可知，厅数为3的要明显和其他数量的有较大的区别，对此将其转换为{“厅数为3”:1}和“{厅数不为3”:0}的二分类变量
data['hall_judge'] = data['hall'].apply(lambda x: '厅数为3' if x == 3 else '厅数不为3')
print(data.head())

# In[148]:


# # # 对于对分类变量，生成哑变量(通常取值 0 或 1，来反映某个变量的不同属性)
data1 = pd.get_dummies(data[['district', 'C_floor']])  # get_dummies将分类变量转换为哑变量/指示变量
print(data1.head())
data1.drop(['district_坪山', 'C_floor_high'], axis=1, inplace=True)  # 这两个是参照组

# In[149]:


# # # 生成的哑变量与其他所需变量合并成新的数据表
data2 = pd.concat([data1, data[['school', 'subway', 'hall_judge', 'roomnum', 'AREA', 'per_price']]], axis=1)
print(data2.head())

# In[150]:


# (二) 回归模型
# # 1.线性回归模型
lm1 = ols(
    'per_price ~ district_罗湖+district_南山+district_光明+district_龙华+district_盐田+district_龙岗+district_福田+district_宝安+school+subway+C_floor_middle+C_floor_low+AREA',
    data=data2).fit()
lm1_summary = lm1.summary()
print(lm1_summary)  # 回归结果展示

data1['predict1'] = lm1.predict(data2)
data1['resid1'] = lm1.resid
data1.plot('predict1', 'resid1', kind='scatter', color='#00686b')
plt.show()
'''
作散点图 → 模型诊断图：从左往右图形发散，即：存在异方差现象，对因变量取对数
'''

# In[151]:


# # 2.对数线性模型
data2['per_price_ln'] = np.log(data2['per_price'])
data2['AREA_ln'] = np.log(data2['AREA'])

# # # 1) 只对y取对数
lm2 = ols(
    'per_price_ln ~district_罗湖+district_南山+district_光明+district_龙华+district_盐田+district_龙岗+district_福田+district_宝安+school+subway+C_floor_middle+C_floor_low+AREA',
    data=data2).fit()
lm2_summary = lm2.summary()
print(lm2_summary)

data2['predict2'] = lm2.predict(data2)
data2['resid2'] = lm2.resid
data2.plot('predict2', 'resid2', kind='scatter', color='#00686b')
plt.show()

# In[152]:


lm3 = ols(
    'per_price_ln ~ district_罗湖+district_南山+district_光明+district_龙华+district_盐田+district_龙岗+district_福田+district_宝安+school+subway+C_floor_middle+C_floor_low+AREA_ln',
    data=data2).fit()
lm3_summary = lm3.summary()
print(lm3_summary)
data2['predict3'] = lm3.predict(data2)
data2['resid'] = lm3.resid
data2.plot('predict3', 'resid', kind='scatter', color='#00686b')
plt.show()

# In[153]:


# # 3.有交互项的对数线性模型，城区和学区之间的交互作用
schools = ['龙岗', '光明', '盐田', '龙华', '罗湖', '宝安', '福田', '南山']
print('坪山非学历区\t', round(data[(data['district'] == '坪山') & (data['school'] == 0)]['per_price'].mean(), 2), '万元/㎡\t',
      '坪山学历区\t', round(data[(data['district'] == '坪山') & (data['school'] == 1)]['per_price'].mean(), 2), '万元/㎡\t')
print('-' * 20)
for i in schools:
    print(
        i + '非学区房\t',
        round(data2[(data2['district_' + i] == 1) & (data2['school'] == 0)]['per_price'].mean(), 2),
        '万元/平方米\t',
        i + '学区房\t',
        round(data2[(data2['district_' + i] == 1) & (data2['school'] == 1)]['per_price'].mean(), 2),
        '万元/平方米')

# In[154]:


# # # 图形描述
d = pd.DataFrame()
dist = ['坪山', '龙岗', '光明', '盐田', '龙华', '罗湖', '宝安', '福田', '南山']
Noschool = []
school = []
for i in dist:
    Noschool.append(
        data[(data['district'] == i) & (data['school'] == 0)]['per_price'].mean())
    school.append(
        data[(data['district'] == i) & (data['school'] == 1)]['per_price'].mean())
d['district'] = pd.Series(dist)
d['Noschool'] = pd.Series(Noschool)
d['school'] = pd.Series(school)
print(d)
d1 = d['Noschool'].T.values
d2 = d['school'].T.values
plt.figure(figsize=(10, 6))
x1 = range(0, len(d))
x2 = [i + 0.3 for i in x1]
plt.bar(x1, d1, color='#00686b', width=0.3, alpha=0.6, label='非学区房')
plt.bar(x2, d2, color='#c82d31', width=0.3, alpha=0.6, label='学区房')
plt.xlabel('城区')
plt.ylabel('单位面积价格')
plt.title('分城区、是否学区的房屋价格')
plt.legend(loc='upper left')
plt.xticks(range(0, 10), dist)
plt.show()
'''
除了 南山 之外，其他区的 学区房价格 均比 非学区房 高
'''

# In[155]:


# # # 探索 南山 和 盐田 学区房价格比较低的原因，是否是样本量的问题
num_noschool_ns = data[(data['district'] == '南山') & (data['school'] == 0)].shape[0]
num_school_ns = data[(data['district'] == '南山') & (data['school'] == 1)].shape[0]
print('南山非学区房\t', num_noschool_ns, '\t',
      '南山学区房\t', num_school_ns, '\t',
      '南山学区房占南山所有二手房的{}'.format(int(num_school_ns) / int(int(num_school_ns) + int(num_noschool_ns))))

# In[158]:


# # # 分城区的学区房分组箱线图
school = ['坪山', '龙岗', '光明', '盐田', '龙华', '罗湖', '宝安', '福田', '南山']
for i in school:
    # sns.boxplot(x=df_temp6['school'], y=df_temp5['per_price'],
    #             palette=sns.cubehelix_palette(
    #                 4, start=2, rot=0, dark=0, light=.95))
    sns.boxplot(x=data[data.district == i]['school'],
                y=data[data.district == i]['per_price'],
                palette=sns.cubehelix_palette(
                    4, start=2, rot=0, dark=0, light=.95))
    # data[data.district==i][['school','per_price']].
    #     boxplot(by='school',patch_artist=True,color='#b6b51f')
    plt.xlabel(i + '学区房')
    plt.show()

# In[159]:


# (三) 假想情形，做预测，x_new是新的自变量
'''
预测要找一个条件为：
1.南山区
2.有3个房间
3.面积大概再80㎡左右
4.有地铁
5.学区房
的房子的大概花费
'''
x_new1 = data2.head(1).copy()
print(x_new1)

x_new1['dist_南山'] = 1
x_new1['roomnum'] = 3
x_new1['AREA_ln'] = np.log(80)
x_new1['subway'] = 1
x_new1['school'] = 1
print('-' * 30)
print(x_new1)

# 预测值
import math

print("单位面积房价：", round(math.exp(lm3.predict(x_new1)), 2), "万元/平方米")  # round 保留几位小数
print("总价：", round(math.exp(lm3.predict(x_new1)) * 80, 2), "万元")  # exp 返回x的指数
'''
输出预测的 单位面积房价 和 总价
'''

