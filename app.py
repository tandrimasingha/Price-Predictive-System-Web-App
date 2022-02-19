import streamlit as st
import sklearn
from sklearn import datasets
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



r = st.sidebar.radio("Navigation Menu",["Home","Car Price","Gold Price","Bitcoin Price","Mobile Price","Avocado Price"])

if r=="Home":
    
    st.write("""
    # Price Predictive System
    #    
    """)
    st.image("Price.png")
    st.subheader("This App Predict the price of folowing things ->")
    st.text("1. Car Price Prediction ")
    st.text("2. Salary Prediction")
    # st.text("3. Gold Price Predictions")
    st.text("3. Crypto Currency Bitcoin Price Prediction")
    st.text("4. Mobile Price Prediction")
    st.text("5. Avocado Price Prediction")
    
    
#car price prediction
car_dataset = pd.read_csv('car data.csv')
# encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)
from sklearn.linear_model import LinearRegression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

if r=='Car Price':
    st.header("Know the Price of Cars")
    year=st.number_input("Year in which can was bought",min_value=2000,max_value=2020,step=1)
    Present_Price=st.number_input("Present_Price of car")
    Kms_Driven=st.number_input("Kms Driven",min_value=1000,max_value=10000000000,step=1)
    fuel_Type=st.number_input("Type of Fuel Used (0:Petrol, 1:Disel, 2:CNG)", min_value=0, max_value=2,step=1)
    Seller_Type=st.number_input("Type of Seller (0:Dealer, 1:Individual)",min_value=0, max_value=1,step=1)
    transmission=st.number_input("Transmission (0:Manual, 1:Automatic)",min_value=0, max_value=1,step=1)
    Owner=st.number_input("owner",min_value=0,max_value=20,step=1)
    training_data_prediction = lin_reg_model.predict([[year,Present_Price,Kms_Driven,fuel_Type,Seller_Type,transmission,Owner]])
    
    if(st.button("Predict")):
        st.success(f"Your Predicted Salary Is {abs(training_data_prediction)}")
        

# Salary Prediction
# from sklearn.tree import DecisionTreeRegressor

# df=pd.read_csv("Salary_Data.csv")

# x=df["YearsExperience"]
# y=df["Salary"]
# x=np.array(df["YearsExperience"]).reshape(-1,1)
# y=np.array(df["Salary"]).reshape(-1,1)
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# model_lir=LinearRegression()
# model_lir.fit(x_train,y_train)


# if(r=='Salary Prediction'):
#     st.header("Know Your Predicted Salary with years of experience")
#     exp=st.number_input("Enter Your Experience In Years(Range 0 to 20)",min_value=0,max_value=20,step=1)
#     exp=np.array(exp).reshape(1,-1)
#     preds=model_lir.predict(exp)[0][0]

    # if(st.button("Predict")):
    #     st.success(f"Your Predicted Salary Is {round(preds)}")
        
# Gold Price Prediction

from sklearn.ensemble import RandomForestRegressor
gold_data = pd.read_csv('gld_price_data.csv')
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)

if r=='Gold Price':
    st.header("Know the Price of Gold")
    SPX=st.number_input("Capilatization index of companies that is the stock value")
    USO=st.number_input("United States Oil Price")
    SLV=st.number_input("Silver Price Value")
    EUR_USD=st.number_input("Enter the currency pair")
    test_data_prediction = regressor.predict([[SPX,USO,SLV,EUR_USD]])
    if(st.button("Predict")):
        st.success(f"Your Predicted Salary Is {abs(test_data_prediction)}")
        

# Biocoin Price Prediction
bitcoin=pd.read_csv('coin_Bitcoin.csv')
bitcoin.drop(["Name"],axis=1, inplace=True)
bitcoin.drop(["SNo"],axis=1, inplace=True)
bitcoin.drop(["Symbol"],axis=1, inplace=True)

# import datetime as dt
# bitcoin["Date"]=pd.to_datetime(bitcoin["Date"])
# bitcoin['Date_year'] = bitcoin["Date"].dt.year
# bitcoin['Date_month'] = bitcoin["Date"].dt.month
# bitcoin['Date_day'] = bitcoin["Date"].dt.day
# bitcoin['Date_hour'] = bitcoin["Date"].dt.hour
# bitcoin['Date_minute'] = bitcoin["Date"].dt.minute
# bitcoin['Date_seconde'] = bitcoin["Date"].dt.second
bitcoin.drop(["Date"], axis=1, inplace=True)


X=bitcoin.drop(["Marketcap"], axis=1)
Y=bitcoin["Marketcap"]

from sklearn.linear_model import Lasso
Ls=Lasso()
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)
Ls.fit(xtrain,ytrain)

if r=='Bitcoin Price':
    st.header("Know the Price of Bitcoin")
    High=st.number_input("Highest Price of bitcoin")
    Low=st.number_input("Lowest Price of")
    Open=st.number_input("Opening Price of bitcoin")
    Close=st.number_input("Closing Price of Bitcoin")
    volume=st.number_input("Volume of the bitcoin")
    
    ypred=Ls.predict([[High,Low,Open,Close,volume]])
    if(st.button("Predict")):
        st.success(f"Your Predicted Salary Is {abs(ypred)}")
        
#Mobile Price Prediction
dataset=pd.read_csv('train.csv')
X=dataset.drop('price_range',axis=1)
y=dataset['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)

if r=='Mobile Price':
    st.header("Know the Price of Mobile Phones")
    battery_power=st.number_input("Battery Power of Mobile")
    blue=st.number_input("Has bluetooth or not",min_value=0, max_value=1,step=1)
    clock_speed=st.number_input("speed at which microprocessor executes instructions")
    dual_sim=st.number_input("Has dual sim support or not",min_value=0, max_value=1,step=1)
    fc=st.number_input("Front Camera mega pixels")
    four_g=st.number_input("Has 4G or not",min_value=0, max_value=1,step=1)
    int_memory=st.number_input("Internal Memory in Gigabytes")
    m_dep=st.number_input("Mobile Depth in cm")
    mobile_wt=st.number_input("Weight of mobile phone")
    n_cores=st.number_input("Number of cores of processor")
    pc=st.number_input("Primary Camera mega pixels")
    px_height=st.number_input("Pixel Resolution Height")
    px_width=st.number_input("Pixel Resolution Width")
    ram=st.number_input("Random Access Memory in Megabytes")
    sc_h=st.number_input("Screen Height of mobile in cm")
    sc_w=st.number_input("creen Width of mobile in cm")
    talk_time=st.number_input("longest time that a single battery charge will last when you are")
    three_g=st.number_input("Has 3G or not",min_value=0, max_value=1,step=1)
    touch_screen=st.number_input("Has touch screen or not",min_value=0, max_value=1,step=1)
    wifi=st.number_input("Has wifi or not",min_value=0, max_value=1,step=1)
    
    ypred=knn.predict([[battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi]])
    if(st.button("Predict")):
        st.success(f"Your Predicted Mobile Price Range Is {abs(ypred)}")
        
        

train=pd.read_csv('avocado.csv')
train.drop(['Unnamed: 0','region'],axis=1,inplace=True)

x = train.drop(['Date','AveragePrice'],axis=1)
y = train['AveragePrice']

le = preprocessing.LabelEncoder()
for i in x.columns:
    if x[i].dtype == 'object':
        x[i] = le.fit_transform(x[i].astype(str))
        
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state = 26)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 250,n_jobs=-1)
rfr.fit(x_train,y_train)

if r=='Avocado Price':
    st.header("Know the Price of Avocado")
    Total_Volume=st.number_input(" Total number of avocados sold")
    blue4046=st.number_input("Total number of small avocados sold (PLU 4046)")
    b4225=st.number_input("Total number of medium avocados sold (PLU 4225)")
    d4770=st.number_input("Total number of large avocados sold (PLU 4770)")
    Total_Bags=st.number_input("Total number of bags")
    Small_Bags=st.number_input("Total number of small bags")
    Large_Bags=st.number_input("Total number of large bags")
    XLarge_Bags=st.number_input("Total number of extra large bags")
    type=st.number_input("whether the price/amount is for conventional or organic")
    year=st.number_input("Year of sale")
    
    
    ypred=rfr.predict([[Total_Volume,blue4046,b4225,d4770,Total_Bags,Small_Bags,Large_Bags,XLarge_Bags,type,year]])
    if(st.button("Predict")):
        st.success(f"Your Predicted Avocado Price Is {abs(ypred)}")
    
    
        
        
    
    

    
    
    
    

        
        

   
    

    
    
    