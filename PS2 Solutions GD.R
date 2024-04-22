# Date:			      Feb 15, 2024
# Last Updated:   
# Author:		    	Gaurav Doshi
# Description:		Solutions for Problem Set 2: ECON 4803/8803 Spring 2024
#	

Packages <- c("ggplot2", "dplyr", "stargazer", "glmnet", "leaps", "data.table", "tictoc")
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

datam$noise1 <- datam$host_experience + rnorm(nrow(datam), .01)
datam$noise2 <- datam$host_is_superhost + rnorm(nrow(datam), .01)
datam$noise3 <- datam$number_of_reviews + rnorm(nrow(datam), .01)

# 2. Analysis
regressions <- function(sample_size, noise_flag){
  
  # set training and testing data
  train <- sample(1:nrow(datam), sample_size*nrow(datam))
  test <- (-train)  
  y_test <- datam$price[test]
  data_train <- datam[train, ]
  result <- rep(NA, 14)
    
  if(noise_flag == 0){
  # create the model matrix without noise
  model.mat <- model.matrix(price ~ (accommodates + beds + host_experience +  number_of_reviews + review_scores_rating)^2 + host_is_superhost + entire_apt +
                                I(accommodates^2) + I(beds^2) + I(host_experience^2) + I(number_of_reviews^2) + I(review_scores_rating^2), 
                              data = datam)
    } else{
  # create the model matrix with noise
  model.mat <- model.matrix(price ~ (accommodates + beds + host_experience + number_of_reviews + review_scores_rating)^2 + host_is_superhost + entire_apt + 
                                  I(accommodates^2) + I(beds^2) + I(host_experience^2) + I(number_of_reviews^2) + I(review_scores_rating^2) + 
                                  noise1 + noise2 + noise3, data = datam)
    }
    
    # c. Linear regression
  if(noise_flag == 0){ 
    ols <- lm(price ~ accommodates + beds + host_experience + host_is_superhost + entire_apt + number_of_reviews + review_scores_rating, 
              data = datam[train, ]) }
  else {
    ols <- lm(price ~ accommodates + beds + host_experience + host_is_superhost + entire_apt + number_of_reviews + review_scores_rating +
                noise1 + noise2 + noise3, data = datam[train, ])
  }
    ols_df <- as.data.frame(model.mat)
    ols_df$price <- datam$price
    result[1] <- summary(ols)$r.squared
    ols.pred <- predict(ols, ols_df[test , ])
    result[2] <- mean((ols.pred - y_test)^2)
    
    # d. Polynomials and interactions in linear regression
    ols_full <- lm(price ~ ., data = ols_df[train,])
    result[3] <- summary(ols_full)$r.squared
    
    ols_full.pred <- predict(ols_full, ols_df[test , ])
    result[4] <- mean((ols_full.pred - y_test)^2)
    
    # e. Backward selection
    if(noise_flag == 0){ 
      # regsubsets automatically adds intercept
      regfit.bwd <- regsubsets(price ~ . -1, data = ols_df[train,],  nvmax=23, method = "backward") }
    else {
      # regsubsets automatically adds intercept
      regfit.bwd <- regsubsets(price ~ . -1, data = ols_df[train,],  nvmax=26, method = "backward")
    }
    
    regfit.bwd.summ <- summary(regfit.bwd)
    
    # 1. using BIC
    ncoef <- which.min(regfit.bwd.summ$bic)
    coefbwd <- coef(regfit.bwd, id = ncoef)
    test.mat <- model.matrix(price ~ ., ols_df[test, ])
    pred <- test.mat[, names(coefbwd)] %*% coefbwd
    result[5] <- mean((pred - y_test)^2)  
    
    # 2. use R2
    ncoef <- which.max(regfit.bwd.summ$rsq)
    coefbwd <- coef(regfit.bwd, id = ncoef)
    pred <- test.mat[, names(coefbwd)] %*% coefbwd
    result[6] <- mean((pred - y_test)^2)  
    
    # f. Ridge & Lasso
    lambda <- c(0, 10, 20)
    price <- datam$price
    ridge_reg <- glmnet(model.mat[train,], price[train], alpha = 0, lambda = lambda, thresh = 1e-12, standardize = FALSE)
    lasso_reg <- glmnet(model.mat[train,], price[train], alpha = 1, lambda = lambda, thresh = 1e-12, standardize = FALSE)
    
    for (i in 1:length(lambda)) {
      ridge.pred <- predict(ridge_reg, s = lambda[i], newx = model.mat[test,])
      result[7+(i-1)] <- mean((ridge.pred - y_test)^2)
      
      lasso.pred <- predict(lasso_reg, s = lambda[i], newx = model.mat[test,])
      result[10+(i-1)] <- mean((lasso.pred - y_test)^2)
    }
    
    # g. 10 fold CV for Ridge and Lasso
    # create a grid of lambdas
    # grid <- 10^seq(5, -2, length = 200)
    ridge.reg <- glmnet(model.mat[train,], price[train], alpha = 0, thresh = 1e-12)
    lasso.reg <- glmnet(model.mat[train,], price[train], alpha = 1, hresh = 1e-12)
      
    # Ridge
    ridge.cv <- cv.glmnet(model.mat[train, ], price[train], alpha = 0, nfolds = 10)
    lambda_hat <- ridge.cv$lambda.min
    ridge.cv.pred <- predict(ridge.reg, s = lambda_hat, newx = model.mat[test,])
    result[13] <- mean((ridge.cv.pred - y_test)^2)
    
    # Lasso
    lasso.cv <- cv.glmnet(model.mat[train, ], price[train], alpha = 1, nfolds = 10)
    lambda_hat <- ridge.cv$lambda.min
    lasso.cv.pred <- predict(lasso.reg, s = lambda_hat, newx = model.mat[test,])
    result[14] <- mean((lasso.cv.pred - y_test)^2)

    # round off MSE values in the results vector 
    result[c(-1,-3)] <- round(result[c(-1,-3)], 0)
    
    # return the vector 'result' that contains the required outputs
    return(result)
}

set.seed(0)
sample_size <- c(.5, .1, .02)
noise_flag <- c(0, 1)

# matrix to store the results - 14 columns and 6 rows 
results_matrix <- matrix(data = 0, nrow = 14, ncol = 6)

col <- 1
for (i in 1:length(sample_size)) {
  for (j in 1:length(noise_flag)) {
    results_matrix[, col] <- regressions(sample_size[i], noise_flag[j])
    col <- col + 1
  }
}

results.df <- as.data.frame(results_matrix)
colnames(results.df) <- c("Tr 0.5, No noise", "Tr 0.5, noise", "Tr 0.1, No noise", "Tr 0.1, noise", "Tr 0.02, No noise", "Tr 0.02, noise")
rownames(results.df) <- c("OLS R2", "OLS MSE", "Full OLS R2", "Full OLS MSE", "BS BIC, MSE", "BS R2, MSE", "Ridge l=0 MSE", "Ridge l=5 MSE", "Ridge l=10 MSE",
                          "Lasso l=0 MSE", "Lasso l=5 MSE", "Lasso l=10 MSE", "Ridge CV MSE", "Lasso CV MSE")

stargazer(results.df, summary = F, rownames = T)

