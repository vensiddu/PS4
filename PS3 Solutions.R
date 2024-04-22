# Date:			      Mar 4, 2024
# Last Updated:   
# Author:		    	Gaurav Doshi
# Description:		Solutions for Problem Set 3: ECON 4803/8803 Sp 2024
#	

Packages <- c("ggplot2", "dplyr", "stargazer", "glmnet", "leaps", "e1071", "collapse")
lapply(Packages, library, character.only = TRUE)

# 1. Load the data
airbnb_data <- read.csv("~/Dropbox/gd documents/Academics/Lectures/GA Tech/ECON 4803 ML for Econ/Fall 23/Problem Sets/Data/airbnb_data.csv")
airbnb_data$X <- NULL

datam <- airbnb_data[complete.cases(airbnb_data[, c("price", "accommodates", "beds", "number_of_reviews", "review_scores_rating")]),]
datam$host_experience <- difftime(as.Date("2023-06-01"), as.Date(datam$host_since), units = "days")
datam$host_experience <- as.double(datam$host_experience)/365
datam <- datam[complete.cases(datam[, c("host_experience")]),]

datam$entire_apt <- ifelse(datam$room_type == "Entire home/apt", 1, 0)

datam$host_is_superhost <- ifelse( (datam$host_response_rate >=90 & datam$number_of_reviews >=10 & datam$review_scores_rating >= 4.8 ), 1, 0)

datam <- datam[complete.cases(datam[, c("host_is_superhost")]),]
datam <- datam[order(datam$id),]
datam <- datam[complete.cases(datam[, c("host_is_superhost")]),]
datam <- datam[order(datam$id),]

# Preliminary data cleaning:
datam$host_identity_verified <- ifelse(datam$host_identity_verified == "t", 1, 0)
datam <- datam[complete.cases(datam[, c("review_scores_rating", "review_scores_accuracy", "review_scores_value")]),]
#datam$host_is_superhost <- as.factor(datam$host_is_superhost)

# analysis
set.seed(0)
train <- sample(1:nrow(datam), .9*nrow(datam))
test <- (-train)  
y_test <- datam$host_is_superhost[test]
data_train <- datam[train, ]

# b. LPM
lpm <- lm(host_is_superhost ~ review_scores_rating, data =  datam[train, ])
data_train$lpm_prediction <- predict(lpm,  datam[train, ])

# logit 
logit <- glm(host_is_superhost ~ review_scores_rating, data =  datam[train, ], family = binomial(link = "logit"))
data_train$logit_pred <- predict(logit, datam[train, ], type = "response")

# probit
probit <- glm(host_is_superhost ~ review_scores_rating, data =  datam[train, ], family = binomial(link = "probit"))
data_train$probit_pred <- predict(probit, datam[train, ], type = "response")

# regression coefficients:
coeff_table <- matrix(data = NA, nrow = 2, ncol=4)
coeff_table[,1] <- names(lpm$coefficients)
coeff_table[,2] <- round(lpm$coefficients, 3)
coeff_table[,3] <- round(probit$coefficients, 3)
coeff_table[,4] <- round(logit$coefficients, 3)
colnames(coeff_table) <- c("","LPM","Probit","Logit")
stargazer(coeff_table, summary = F, rownames = F)

# SVM with gaussian kernel
data_train <- datam[train, c("host_is_superhost", "review_scores_rating", "host_experience")]
data_train$host_is_superhost <- as.factor(ifelse(data_train$host_is_superhost == 0, -1, 1))

tune.out <- tune(svm, host_is_superhost ~  ., data = data_train, kernel="radial", scale = TRUE, gamma = 0.01, 
              ranges = list(cost = c(1, 10, 100, 10^3, 10^4)))

svm.pred <- predict(tune.out$best.model, newdata = datam[test, c("review_scores_rating", "host_experience")])
svm.pred <- ifelse(svm.pred == -1 , 0 , 1)


# l1 regularized logit
x <- model.matrix(host_is_superhost ~  (review_scores_rating + host_experience + review_scores_accuracy + beds + review_scores_value)^2 + 
                    I(review_scores_rating^2) + I(host_experience^2) + I(review_scores_accuracy^2) + I(beds^2) + I(review_scores_value^2),  data = datam)
y <- datam$host_is_superhost
lasso.logit <- cv.glmnet(x[train,] , y[train], alpha = 1, family = binomial(link = "logit"))
lambda_hat <- lasso.logit$lambda.min
data_train$logit_lasso_pred <- predict(lasso.logit, s = lambda_hat, x[train,], type = "response")


# predict the binary response for test data
test_lpm <- mean((predict(lpm,  datam[test, ]) >= .5) != y_test)

test_logit <- mean((predict(logit, datam[test, ], type = "response") >= .5) != y_test)

test_probit <- mean((predict(probit, datam[test, ], type = "response") >= .5) != y_test)

test_svm <- mean(svm.pred != y_test)

test_lasso_logit <- mean((predict(lasso.logit, s = lambda_hat, x[test, ], type = "response") >= .5) != y_test)

mse_table <- matrix(data = NA, nrow = 5, ncol=2)
mse_table[,1] <- c("LPM", "Logit", "Probit", "SVM", "Regularized Logit")
mse_table[,2] <- round(rbind(test_lpm, test_logit, test_probit, test_svm, test_lasso_logit), 3)
stargazer(mse_table, summary = F, rownames = F)


