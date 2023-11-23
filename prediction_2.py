# Cài đặt thư viện
import streamlit as st
import pandas as pd
import vnstock
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
# Title
st.title("Prediction Stock VietNam")
ticker = st.sidebar.text_input('Mã chứng khoán 1', key = 1)
start_date = st.sidebar.date_input('Ngày bắt đầu', key = 2)
end_date = st.sidebar.date_input('Ngày kết thúc', key = 3)
ticker1 = st.sidebar.text_input('Mã chứng khoán 2',key = 4)

# Hiển thị dữ liệu lấy về của mã chứng khoán 1
try:
  if ticker == '':
    # Ví dụ lấy giá Vinhome từ ngày 06/11/2022 - 06/11/2023
    data_stock = vnstock.stock_historical_data('VHM','2022-11-06','2023-11-06','1D')
    ticker = 'VHM'
  else:
    if start_date >= end_date:
      data_stock = vnstock.stock_historical_data(ticker,'2022-11-06','2023-11-06','1D')
    else:
      data_stock = vnstock.stock_historical_data(ticker, str(start_date) , str(end_date),'1D')
except:
  data_stock = vnstock.stock_historical_data('VHM','2022-11-06','2023-11-06','1D')

# Hiển thị dữ liệu lấy về của mã chứng khoán 2
try:
  if ticker1 == '':
    # Ví dụ lấy giá FPT từ ngày 06/11/2022 - 06/11/2023
    data_stock_1 = vnstock.stock_historical_data('FTS','2022-11-06','2023-11-06','1D')
    ticker1 = 'FTS'
  else:
    if start_date >= end_date:
      data_stock_1 = vnstock.stock_historical_data(ticker1,'2022-11-06','2023-11-06','1D')
    else:
      data_stock_1 = vnstock.stock_historical_data(ticker1, str(start_date) , str(end_date),'1D')
except:
  data_stock_1 = vnstock.stock_historical_data('FTS','2022-11-06','2023-11-06','1D')


# Gồm có giá mở cửa, đóng cửa, thấp nhất, cao nhất, khối lượng giao dịch
#-------- Hiển thị
st.write(f"Dữ liệu chứng khoán {ticker}")
st.dataframe(data_stock.style.background_gradient(cmap ='RdYlBu').set_properties(**{'font-size': '20px'}))

# Thông tin công ty
st.title(f"Thông tin về lịch sử công ty có mã chứng khoán: {ticker}")
thong_tin = vnstock.company_profile(ticker)[['historyDev']].values[0][0].split(';')
for i in thong_tin:
  st.write(i)

data_stock = data_stock.set_index('time')
data_stock_1 = data_stock_1.set_index('time')
# data_stock.drop('ticker', axis=1, inplace=True)
fig = px.line(data_stock, x = data_stock.index, y = data_stock['close'], title = ticker)
#----- Hiển thị
st.title(f"Biểu đồ giá đóng cửa chứng khoán mã {ticker}")
st.title(f"Từ {start_date} đến {end_date}")
st.plotly_chart(fig)

data_stock.drop('ticker', axis=1, inplace=True)


# Vẽ biểu đồ thể hiện sự tăng giảm của cổ phiếu
bieu_do = go.Figure(data=go.Ohlc(x=data_stock.index, open=data_stock['open'],
                              high=data_stock['high'], low = data_stock['low'],
                              close = data_stock['close']))
#--------- Hiển thị
st.title(f"Biểu đồ đầy đủ của chứng khoán mã {ticker}")
st.plotly_chart(bieu_do, use_container_width=True)

# Vẽ biểu đồ so sánh giá đóng cửa hai mã chứng khoán
plt.style.use('fivethirtyeight')
plt.figure(figsize=(16,4))
plt.title("Biểu đồ giá đóng cửa")
plt.plot(data_stock["close"],linewidth = 2)
plt.plot(data_stock_1["close"],linewidth = 2)
plt.xlabel("Ngày",fontsize=18)
plt.ylabel("Giá cố phiếu ($) ",fontsize=18)
plt.legend([ticker, ticker1])

#--------- Hiển thị
st.write(f"Biểu đồ so sánh giá đóng cửa giữa mã {ticker} và {ticker1}")
st.pyplot(plt)


#Biểu đồ cột
data_stock_close = vnstock.stock_historical_data(ticker,'2022-11-06','2023-11-06','1D')[['close', 'time']]
data_stock_close['year'] = data_stock_close['time'].apply(lambda x: str(x.month) +'/'+ str(x.year))
data_theo_thang = data_stock_close.groupby('year').mean().dtypes.astype('int').round()
#Biểu đồ thể hiện giá đóng cửa trung bình của tháng
plt.figure(figsize=(15,6))
plt.bar(data_theo_thang.index, data_theo_thang['close'] , color = 'red',width = 0.5)
plt.title('Biểu đồ giá đóng cửa')
plt.xlabel('Ngày')
plt.ylabel('Giá cao nhất')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
# Chuyển đổi từ list sang array
xs =  np.array(list(data_theo_thang.index))
ys =  np.array(list(data_theo_thang['close']))
for x,y in zip(xs,ys):

    label = "{:}".format(y)

    plt.annotate(label,
                 (x,y),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

plt.show()
#--------- Hiển thị
st.title("Biểu đồ giá đóng cửa")
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
#################################################################################


# Cập nhật kết quả để vẽ biểu đồ
kq = data_stock[-len(predictions1[0]):]
kq['Lstm'] = predictions4
kq['Linear Regresstive'] = predictions1[0]
kq['KNN'] = predictions2[0]
kq['SVM'] = predictions3[0]
# Vẽ kết hợp trước và lúc dự đoán
data_noi = data_stock[:len(data_stock) - len(predictions1[0])]
kq = data_noi.append(kq)
# kq = kq.set_index('time')
# Biểu đồ so sánh dự đoán của các mô hình với giá thực tế
plt.figure(figsize=(14, 5))
plt.title(f"Biểu đồ so sánh dự đoán của các mô hình với giá thực tế {ticker}")
plt.plot(kq['Linear Regresstive'],color="Orange",linewidth=2)
plt.plot(kq['KNN'],color="Blue",linewidth=2)
plt.plot(kq['SVM'],color="green",linewidth=2)
plt.plot(kq['Lstm'],color="Purple",linewidth=2)
plt.plot(kq['close'],color="red",linewidth=2)
plt.xlabel("Ngày",fontsize=18)
plt.ylabel("Giá cổ phiếu ( ngàn VND )",fontsize=18)
plt.legend(["Linear Regressive","KNN","SVM","LSTM", "Giá thật"], loc='lower left')

st.title("Biểu đồ dự đoán của các mô hình AI")
st.pyplot(plt)
# plt
# plt.savefig('foo.png')
# plt.savefig('foo.pdf')


y_test_2 = []
for i in y_test:
  y_test_2.extend(i)
y_test_2 = np.array(y_test_2)

# Tính độ lệch rmse chuẩn của các mô hình

rmse1 = np.sqrt(np.mean(((predictions1[0]- y_test_2)**2)))
rmse2 = np.sqrt(np.mean(((predictions2[0]- y_test_2)**2)))
rmse3 = np.sqrt(np.mean(((predictions3[0]- y_test_2)**2)))
rmse4 = np.sqrt(np.mean(((predictions4- y_test_2)**2)))

mae1 = np.mean(np.abs((predictions1[0]- y_test_2)))
mae2 = np.mean(np.abs((predictions2[0]- y_test_2)))
mae3 = np.mean(np.abs((predictions3[0]- y_test_2)))
mae4 = np.mean(np.abs((predictions4[0]- y_test_2)))

mape1 = np.mean(np.abs((y_test_2 - predictions1[0]) / y_test_2))
mape2 = np.mean(np.abs((y_test_2 - predictions2[0]) / y_test_2))
mape3 = np.mean(np.abs((y_test_2 - predictions3[0]) / y_test_2))
mape4 = np.mean(np.abs((y_test_2 - predictions4[0]) / y_test_2))


gia_trung_binh_test = np.mean(y_test_2)

bang_danh_gia = pd.DataFrame([['Linear Regression',gia_trung_binh_test, rmse1, mae1, mape1],
              ['KNN',gia_trung_binh_test, rmse2, mae2, mape2],
              ['SVM',gia_trung_binh_test, rmse3, mae3, mape3],
              ['LSTM',gia_trung_binh_test, rmse4, mae4, mape4]],
             columns = ['Model', 'Giá trung bình', 'RMSE', 'MAE', 'MAPE'])
st.title("Đánh giá các mô hình AI")
st.dataframe(bang_danh_gia.style.background_gradient(cmap ='RdYlBu').set_properties(**{'font-size': '20px'}))
