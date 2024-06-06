library(tensorflow)
library(keras)
library(quantmod)
library(zoo)
getSymbols(c("^IXIC", "^VIX", "DX-Y.NYB"),from="2006-01-03", to="2021-09-29")
getSymbols(c("EFFR","UNRATE","UMCSENT"),src='FRED',from="2006-01-03", to="2021-09-29")
EFFR = na.approx(EFFR)
DX = `DX-Y.NYB`$`DX-Y.NYB.Close`
colnames(DX) = "DX"
myts = cbind(IXIC$IXIC.Close, VIX$VIX.Close, EFFR, UNRATE, UMCSENT, DX)
myts = myts[-which(is.na(myts$IXIC.Close)),]
myts$UNRATE[1] = 4.9
myts$UMCSENT[1] = 91.5
myts$UNRATE = na.locf(myts$UNRATE)
myts$UMCSENT = na.locf(myts$UMCSENT)
myts$DX = na.approx(myts$DX)

getSymbols("^IXIC", from="2005-11-25", to="2021-09-29")
myts$MACD = MACD(IXIC$IXIC.Close)$macd[-(1:25)]
myts$ATR = ATR(IXIC)$atr[-(1:25)]
myts$RSI = RSI(IXIC$IXIC.Close)[-(1:25)]
getSymbols(c("^IXIC"),from="2006-01-03", to="2021-09-29")

#normalization
normalize = function(x) (x - min(x))/ diff(range(x))
myts = data.frame(lapply(myts, normalize))

lag = 11

#length of train dataset is 3960*0.8 = 3168
#train.size = dimension of x.train matrix = length of train dataset - lag
train.size = 3168 - lag

#length of test dataset is 3960*0.2 = 792
#test.size = dimension of x.test matrix = length of test dataset - lag
test.size = 792 - lag
train = myts[1:(train.size+lag), ]
test = myts[(train.size+lag+1):dim(myts)[1], ]
batch.size = 11

#create x.train, y.train, x.test, and y.test arrays
create_xtrain = function(train1){
  x.train = matrix(NA,train.size,lag)
  for (i in 1:train.size){
    x.train[i,] = train1[i:(lag-1+i)]
  }
  return(x.train)
}
create_xtest = function(test1){
  x.test = matrix(NA,test.size,lag)
  for (i in 1:test.size){
    x.test[i,] = test1[i:(lag-1+i)]
  }
  return(x.test)
}
y.train = train$IXIC.Close[-(1:lag)]
y.test = test$IXIC.Close[(lag+1):(test.size+lag)]

train_data = lapply(train, create_xtrain)
test_data = lapply(test, create_xtest)


x.train = array(data = c(train_data$IXIC.Close, train_data$VIX.Close, train_data$EFFR, train_data$UNRATE,
                         train_data$UMCSENT, train_data$DX, train_data$MACD, train_data$ATR, train_data$RSI), 
                dim = c(train.size, lag, 9))
y.train = array(data = y.train, dim = c(train.size, 1))
x.test = array(data = c(test_data$IXIC.Close, test_data$VIX.Close, test_data$EFFR, test_data$UNRATE,
                        test_data$UMCSENT, test_data$DX, test_data$MACD, test_data$ATR, test_data$RSI), 
               dim = c(test.size, lag, 9))
y.test = array(data = y.test, dim = c(test.size, 1))

#Model
model = keras_model_sequential()
model %>%
  layer_lstm(units = 150,
             input_shape = c(lag, 9),
             batch_size = batch.size,
             return_sequences = FALSE, 
             stateful = TRUE) %>%
  layer_dense(units = 1)
model %>%
  compile(loss = 'mae', optimizer = 'adagrad')
model

model %>% fit(
  x.train,
  y.train,
  epochs = 100,
  batch_size = batch.size)

pred_out = model %>% predict(x.test, batch_size = batch.size) 
predictions = pred_out*(max(IXIC$IXIC.Close) - min(IXIC$IXIC.Close)) + min(IXIC$IXIC.Close)
actual = y.test *(max(IXIC$IXIC.Close) - min(IXIC$IXIC.Close)) + min(IXIC$IXIC.Close)

#Plot
plot(predictions,type="l",col="red")
lines(actual,col="green")
plot(predictions, actual)


#RMSE
sqrt(mean((predictions - actual)^2))
#MAPE
mean(abs((actual-predictions)/actual))*100
#R
cor(predictions, actual)


pred_out = model %>% predict(x.train, batch_size = batch.size) 
predictions = pred_out*(max(IXIC$IXIC.Close) 
                        - min(IXIC$IXIC.Close)) + min(IXIC$IXIC.Close)
actual = y.train *(max(IXIC$IXIC.Close) 
                   - min(IXIC$IXIC.Close)) + min(IXIC$IXIC.Close)

#Plot
plot(predictions,type="l",col="red")
lines(actual,col="green")
plot(predictions, actual)

sqrt(mean((predictions - actual)^2))
#MAPE
mean(abs((actual-predictions)/actual))*100
#R
cor(predictions, actual)

