import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import pickle
import json
df=pd.read_csv(r"C:\Users\Samyak Shah\OneDrive\Desktop\real estate price prediction project\bengaluru_house_prices.csv")
df1=df.drop(['availability','society','balcony','area_type'],axis='columns')

df1['size_numeric'] = df1['size'].str.extract('(\d+)').astype(float)

df2=df1.drop(['size'],axis='columns')
mean_bath=df2['bath'].mean()
df2['bath']=df2['bath'].fillna(mean_bath)
mean_size=df2['size_numeric'].mean()
df2['size_numeric']=df2['size_numeric'].fillna(mean_size)
df3=df2.dropna(subset=['location'])
def convert_sqft_num(x):
    x=str(x).strip()
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0].strip())+float(tokens[1].strip()))/2
    try:
        return float(x)
    except:
        return None
df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_num)

df5=df4.copy()
df5['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
df5=df5.dropna(subset=['total_sqft'])

df5['price_per_sqft']=(df5['price']*100000)/df5['total_sqft']

df6=df5.copy()
# print(len(df6['location'].unique()))
df6['location']=df6['location'].apply(lambda x:x.strip())
location_stats=df6['location'].value_counts(ascending=False)
less_than_10=location_stats[location_stats<=10]
df7=df6.copy()
df7['location']=df7['location'].apply( lambda x: 'other' if x in less_than_10 else x)
# print(len(df7['location'].unique()))
df8=df7.copy()
# print(df8.shape)
# print(len(df8[(df8['total_sqft']/df8['size_numeric'])<300]))
df9=df8[~(df8['total_sqft']/df8['size_numeric']<300)]
# print(df9.shape)
# print(df9['price_per_sqft'].describe())

def remove_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf['price_per_sqft'])
        st=np.std(subdf['price_per_sqft'])
        reduced_df=subdf[(subdf['price_per_sqft']>(m-st))&(subdf['price_per_sqft']<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df10=remove_outliers(df9)
# print(df10.shape)
df11=df10.copy()
bath_outlier=(df11[df11['bath']>(df11['size_numeric']+2)])
df12=df11[df11['bath']<(df11['size_numeric']+2)]

def plot_scatter(df,location):
    bhk2=df[(df['location']==location) & (df['size_numeric']==2)]
    bhk3=df[(df['location']==location) & (df['size_numeric']==3)]
    plt.scatter(bhk2['total_sqft'],bhk2['price'],color='blue',marker='+')
    plt.scatter(bhk3['total_sqft'],bhk3['price'],color='red',marker='*')
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()

def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location,loc_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in loc_df.groupby('size_numeric'):
            bhk_stats[bhk]={
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk,bhk_df in loc_df.groupby('size_numeric'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    
    return df.drop(exclude_indices,axis='index')
# print(df12.shape)
df13=remove_bhk_outliers(df12)
# print(df13.shape)
# plot_scatter(df13,"Hebbal")
# plt.show()

dummies=pd.get_dummies(df13['location']).astype(int)
df14=pd.concat([df13,dummies],axis='columns')
df15=df14.drop(['other'],axis='columns')
y=df15['price']
X=df15.drop(['location','price_per_sqft','price'],axis='columns')
# print(X.head(3))    
# print(y.head())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
print(len(X_train))
print(len(X_test))
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# print(cross_val_score(LinearRegression(), X, y, cv=cv))

model=LinearRegression()
model.fit(X_train,y_train)
# print(model.score(X_test,y_test))


def predict_price(location,sqft,bath,size_numeric):    
    loc_index = np.where(X.columns==location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = size_numeric
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]
print(predict_price('Hebbal',1000, 2, 2))
print(predict_price('Indira Nagar',1000, 2, 2))
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(model,f)
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))