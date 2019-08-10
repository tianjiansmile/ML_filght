# _*_coding:utf-8 _*_
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('D:/特征提取/payday/payday2_repayment_data.csv',encoding = 'utf-8')
    groupby = df.groupby(df['product_name'])
    for name, group in df.groupby('product_name'):
        group.to_csv(name+'.csv')

