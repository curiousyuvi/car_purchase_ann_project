# Import Required packages
install.packages("neuralnet")
set.seed(500)
library(neuralnet)

# data-set taken from Kaggle
# https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction?select=car_purchasing.csv

# ready data
colnames(car_purchasing)=c("customer_name","email","country","gender","age","annual_salary","credit_card_debt","net_worth","car_purchase_amount")
data <- car_purchasing[,5:9]

# Normalize the data
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

# Split the data into training and testing set
index <- sample(1:nrow(data), round(0.75 * nrow(data)))
train_ <- scaled[index,]
test_ <- scaled[-index,]

# Build Neural Network
nn <- neuralnet(car_purchase_amount ~ net_worth + credit_card_debt + annual_salary + age, data = train_, hidden = c(5, 3), linear.output = TRUE)

# Predict on test data
pr.nn <- compute(nn, test_)

# Compute mean squared error
pr.nn_ <- pr.nn$net.result * (max(data$car_purchase_amount) - min(data$car_purchase_amount)) + min(data$car_purchase_amount)
test.r <- (test_$car_purchase_amount) * (max(data$car_purchase_amount) - min(data$car_purchase_amount)) + min(data$car_purchase_amount)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)

# Plot the neural network
plot(nn)


# Plot regression line
plot(test.r, pr.nn_, col = "red", main = 'Real vs Predicted')
abline(0, 1, lwd = 2)
