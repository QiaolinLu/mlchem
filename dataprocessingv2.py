from function import fake_data
import pandas as pd
from sklearn.utils import shuffle
seed = 7
file_path = r'D:\Programming\algorithm\AI\mlchem\data.xlsx'
df = pd.read_excel(file_path)
df = df.set_index("Unnamed: 0")
df.index_name = "function_groups"
# 切分训练集、验证集、测试集
df.index.name = "function_groups/parameters"
df = df.drop(["o-NO2 ", "o-COMe"])
df_vtrain = df.sample(frac=0.8, random_state = 42)
df_test=df[~df.index.isin(df_vtrain.index)]
mu = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
sigma = df_vtrain.std().values
df_vtrain = shuffle(fake_data(df_vtrain, 0.05, 5))#生成的数据偏差在原数据的0.1sigma内，以原数据为基础生成10个数据


X_Mulliken_vtrain = df_vtrain.loc[:, ['x1', 'x2', 'x3', 'x6', 'x7', 'x10', 'x11']].values
X_NBO_vtrain = df_vtrain.loc[:, ['x1', 'x2', 'x3', 'x6', 'x7', 'x12', 'x13']].values
X_vtrain = df_vtrain.loc[:, ['x1', 'x2', 'x3', 'x6', 'x7', 'x10', 'x11', 'x12', 'x13']].values
y_BindingEnergy_vtrain = df_vtrain.loc[:, ['y1']].values.flatten()
y_ActivationEnergy_vtrain = df_vtrain.loc[:, ['y2']].values.flatten()
y_Q_vtrain = df_vtrain.loc[:, ['y3']].values.flatten()

# test dataset

X_Mulliken_test = df_test.loc[:, ['x1', 'x2', 'x3', 'x6', 'x7', 'x10', 'x11']].values
X_NBO_test = df_test.loc[:, ['x1', 'x2', 'x3', 'x6', 'x7', 'x12', 'x13']].values
X_test = df_test.loc[:, ['x1', 'x2', 'x3', 'x6', 'x7', 'x10', 'x11', 'x12', 'x13']].values
y_BindingEnergy_test = df_test.loc[:, ['y1']].values.flatten()
y_ActivationEnergy_test = df_test.loc[:, ['y2']].values.flatten()
y_Q_test = df_test.loc[:, ['y3']].values.flatten()
"""
y1: 结合能2-pcm y2: 活化能2-ad1-pcm y3: 反应热2-ad1-pcm 
x1: C-F键长 x2: 平衡结构的LUMO x3: 平衡结构的HOMO x7: b-SOMO能量 
x10: C1-Mulliken x11: F-Mulliken x12: C1-NBO x13: F-NBO
"""
class datapreprocessing():
    def df_vtrain(self):
        return df_vtrain
    def X_vtrain(self):
        return X_vtrain
    def X_Mulliken_vtrain(self):
        return X_Mulliken_vtrain
    def X_Mulliken_test(self):
        return X_Mulliken_test
    def y_BindingEnergy_vtrain(self):
        return y_BindingEnergy_vtrain
    def y_BindingEnergy_test(self):
        return y_BindingEnergy_test
    def y_ActivationEnergy_vtrain(self):
        return y_ActivationEnergy_vtrain
    def y_ActivationEnergy_test(self):
        return y_ActivationEnergy_test
    def y_Q_vtrain(self):
        return y_Q_vtrain
    def y_Q_test(self):
        return y_Q_test


if __name__ == "__main__":
    print("df_vtrain: \n", df_vtrain)