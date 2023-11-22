!pip install git+https://github.com/thinh-vu/vnstock.git@beta
# Cài đặt thư viện
import streamlit as st
import pandas as pd
import vnstock
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
# Title
st.title("Prediction Stock VietNam")
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

# Hiển thị dữ liệu lấy về
if ticker == '':
  # Ví dụ lấy giá Vinhome từ ngày 06/11/2022 - 06/11/2023
  data_stock = vnstock.stock_historical_data('VHM','2022-11-06','2023-11-06','1D')
  ticker = 'VHM'
else:
  data_stock = vnstock.stock_historical_data(ticker, str(start_date) , str(end_date),'1D')
data_stock = data_stock.set_index('time')
# data_stock.drop('ticker', axis=1, inplace=True)
fig = px.line(data_stock, x = data_stock.index, y = data_stock['close'], title = ticker)
#----- Hiển thị
st.plotly_chart(fig)


data_stock.drop('ticker', axis=1, inplace=True)
# Gồm có giá mở cửa, đóng cửa, thấp nhất, cao nhất, khối lượng giao dịch
data_stock.head(5)
#-------- Hiển thị
st.write(f"Dữ liệu chứng khoán {ticker}")
st.dataframe(data_stock)

# Vẽ biểu đồ thể hiện sự tăng giảm của cổ phiếu
bieu_do = go.Figure(data=go.Ohlc(x=data_stock.index, open=data_stock['open'],
                              high=data_stock['high'], low = data_stock['low'],
                              close = data_stock['close']))
#--------- Hiển thị
st.write("Biểu đồ thể hiện sự tăng giảm của cổ phiếu")
st.plotly_chart(bieu_do, use_container_width=True)

# Vẽ biểu đồ theo giá đóng cửa, mở cửa
plt.style.use('fivethirtyeight')
plt.figure(figsize=(16,4))
plt.title(f"Giá cổ phiếu của Cty có mã giao dịch là {ticker}")
plt.plot(data_stock["close"],linewidth = 2)
plt.plot(data_stock["open"],linewidth = 2)
plt.xlabel("Ngày",fontsize=18)
plt.ylabel("Giá cố phiếu ($) ",fontsize=18)
plt.legend(['Giá đóng cửa', 'Giá mở cửa'])

#--------- Hiển thị
st.write("Biểu diễn sự chênh lệch giữa giá đóng cửa và mở cửa")
st.pyplot(plt)


#Biểu đồ cột
plt.figure(figsize=(16,6))
plt.bar(data_stock.index, data_stock['high'], color = 'red')
plt.title('Biểu đồ giá đóng cửa')
plt.xlabel('Ngày')
plt.ylabel('Giá cao nhất')
plt.xticks(rotation=50)
plt.yticks(rotation=100)
#--------- Hiển thị
st.write("Biểu đồ giá đóng cửa")
st.pyplot(plt)
# --------------------------------------------------------------------------------
# Xử lý dữ liệu trước khi train model
import math
from  sklearn.preprocessing import MinMaxScaler
import numpy as np
#Xử lý dữ liệu trước khi train
#Tạo bảng dữ liệu chỉ lấy cột giá chốt giao dịch ở cuối ngày
data = data_stock.filter(['close'])
#Tạo chuỗi chỉ chứa giá trị cổ phiếu
dataset = data.values
#Tính số hàng để huấn luyện mô hình
# 80% Train 20% Test
training_data_len = math.ceil( len(dataset) *.8)
#Chuyển toàn bộ dữ liệu về khoản [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
#Create the scaled training data set
train_data = scaled_data[0:training_data_len  , : ]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0]) # 60 ngày để tạo ra 1 input
    y_train.append(train_data[i,0]) # lấy ngày tiếp theo để tạo ra output
#Chuyển dữ liệu train và dữ liệu test về dạng numpy để có thể làm việc với mô hình LSTM
x_train, y_train = np.array(x_train), np.array(y_train)

#Dữ liệu dùng để kiếm tra kết quả của mô hình
test_data = scaled_data[training_data_len - 60: , : ]
#Tạo dữ liệu dùng để test mô hình
x_test = []
y_test =  dataset[training_data_len : , : ]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
#----------------------------------------------------------------------------------
# Model Linear Regressive
from sklearn.linear_model import LinearRegression
model_linear = LinearRegression(fit_intercept=True)
model_linear.fit(x_train, y_train)
yfit=model_linear.predict(x_test)
predictions1 = scaler.inverse_transform([yfit])
#---------------------------------------------------------------------------------
# Model KNN
from sklearn.neighbors import KNeighborsRegressor
knn_regressor=KNeighborsRegressor(n_neighbors = 5)
knn_model=knn_regressor.fit(x_train,y_train)
y_knn_pred=knn_model.predict(x_test)
predictions2 = scaler.inverse_transform([y_knn_pred])
#---------------------------------------------------------------------------------
# Model Suport Vector Machine (SVM)
from sklearn.svm import SVR
svm_regressor = SVR(kernel='linear')
svm_model=svm_regressor.fit(x_train,y_train)
y_svm_pred=svm_model.predict(x_test)
predictions3 = scaler.inverse_transform([y_svm_pred])
#--------------------------------------------------------------------------------
#Model Long Short Term Memmory
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Tạo bảng dữ liệu chỉ lấy cột giá chốt giao dịch ở cuối ngày
data = data_stock.filter(['close'])
#Tạo chuỗi chỉ chứa giá trị cổ phiếu
dataset = data.values
#Tính số hàng để huấn luyện mô hình
training_data_len = math.ceil( len(dataset) *.8)
#Chuyển toàn bộ dữ liệu về khoản [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
#Create the scaled training data set
train_data = scaled_data[0:training_data_len  , : ]
#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
#Chuyển dữ liệu train và dữ liệu test về dạng numpy để có thể làm việc với mô hình LSTM
x_train, y_train = np.array(x_train), np.array(y_train)
#Định dạng lại giá trị đầu vào cho mô hình LSTM (Ma trận 3 chiều)
x_train_lstm = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#Cấu hình Mô hình LSTM (2 tầng LSTM)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
#Biên dịch mô hình
model.compile(optimizer='adam', loss='mean_squared_error')
#Tiến hành training dữ liệu (Số epochs là số lần training dữ liệu)
model.fit(x_train_lstm, y_train, batch_size=1, epochs=3)
#Dữ liệu dùng để kiếm tra kết quả của mô hình
test_data = scaled_data[training_data_len - 60: , : ]
  #Tạo dữ liệu dùng để test mô hình
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])
#Chuyển đổi dữ liệu test về kiểu numpy để tiến hành test với mô hình LSTM
x_test = np.array(x_test)
#Định dạng lại giá trị đầu vào cho mô hình LSTM (Ma trận 3 chiều)
x_test_lstm = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
#Lấy ra giá chứng khoán sau khi đã dự đoán
predictions = model.predict(x_test_lstm)
#Chuyễn lại giá chứng khoán về dạng số thập phân
predictions = scaler.inverse_transform(predictions)
predictions4 = []
for i in predictions:
  predictions4.append(i[0])
#--------------------------------------------------------------------------------
# Model Suport Vector Machine (SVM)
# Dự đoán bằng model
print('Giá trị dự đoán: ',predictions4[5],'\nGiá trị thực tế: ',y_test[5])
print('Độ chênh lệch: ', abs(predictions4[5] - y_test[5]))
#--------------------------------------------------------------------------------
# Cập nhật kết quả để vẽ biểu đồ
kq = data_stock.tail(50)
kq['Lstm'] = predictions4
kq['Linear Regresstive'] = predictions1[0]
kq['KNN'] = predictions2[0]
kq['SVM'] = predictions3[0]
#################################################################################

#Biểu đồ so sánh dự đoán của các mô hình với giá thực tế
plt.figure(figsize=(15,6))
plt.title(f"Biểu đồ so sánh dự đoán của các mô hình với giá thực tế của cổ phiếu có mã {ticker}")
plt.plot(predictions1[0],color="Orange",linewidth=2)
plt.plot(predictions2[0],color="Blue",linewidth=2)
plt.plot(predictions3[0],color="green",linewidth=2)
plt.plot(predictions4,color="Purple",linewidth=2)
plt.plot(y_test,color="red",linewidth=2)
plt.xlabel("Ngày",fontsize=18)
plt.ylabel("Giá cổ phiếu ( ngàn VND )",fontsize=18)
plt.legend(["Linear Regression","KNN","SVM","LSTM","Giá thật"], loc='lower right')

#--------- Hiển thị
st.write("Biểu đồ dự đoán")
st.pyplot(plt)

# # Vẽ kết hợp trước và lúc dự đoán
# data_noi = data_stock[:201]
# kq = data_noi.append(kq)
# # Biểu đồ so sánh dự đoán của các mô hình với giá thực tế
# plt.figure(figsize=(14, 5))
# plt.title(f"Biểu đồ so sánh dự đoán của các mô hình với giá thực tế của cổ phiếu có mã {ticker}")
# plt.plot(kq['Linear Regresstive'],color="Orange",linewidth=2)
# plt.plot(kq['KNN'],color="Blue",linewidth=2)
# plt.plot(kq['SVM'],color="green",linewidth=2)
# plt.plot(kq['Lstm'],color="Purple",linewidth=2)
# plt.plot(kq['close'],color="red",linewidth=2)
# plt.xlabel("Ngày",fontsize=18)
# plt.ylabel("Giá cổ phiếu ( ngàn VND )",fontsize=18)
# plt.legend(["Linear Regressive","KNN","SVM","LSTM", "Giá thật"], loc='lower left')

# #--------- Hiển thị
# st.write("Biểu đồ dự đoán so với thời gian trước")
# st.pyplot(plt)
