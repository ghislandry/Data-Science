# This script implements a recommender system for movies using 
# collaborative filtering learning algorithm
#

library(multcomp)


set.seed(123)

CostFunction <- function(params, Y, R, num.users, num.features, num.movies, lambda){
  
  # param: params: A list containing data frames X and theta; X the first element and theta is the second
  # Note that X and theta are indexed as X and theta respectively.
  # param: Y: the movies rating matrix, one row per movie
  # param: R: A binary matrix with R_i,j = 1, if user i rated movie j; 0 otherwise
  # param: num.users: the number of users
  # param: num.features: the number of features
  # param: num.movies: the numeber of movies (we need num.features and num.movies to retrieve X and theta)
  # param: lambda: the regularization parameter lambda
  # return: In order a list containing the cost, gradient of theta, and gradient of X. Use
  # safer to use gradient.X and gradient.theta to access the gradient of theta and X respectively
  
  # Let us retrieve the feature vector X 
  X = params$X
  theta = params$theta # X.theta.mat[ , (num.features +1): dim(X.theta.mat)[2]]
  
  # evaluation of the cost function
  # note that X.grad, resp theta.grad is a matrices such that the i'th row of X.grad correnponds to the gradient for features 
  # vector x_i  for the i'th movie and the j'th row of theta.grad corresponds to the gradient of one parameter
  # vector theta^j for the j'th user. 
  J = 0
  X.grad = matrix(rep(0, length(X)), nrow = dim(X)[1], ncol = dim(X)[2])
  theta.grad = matrix(rep(0, length(theta)), nrow = dim(theta)[1], ncol = dim(theta)[2])
  
  # Cost function
  
  # for (i in seq(num_movies)) {
  #   
  #   for(j in seq(num_users)){
  #     
  #     if(R[i,j] != 0){
  #       theta.j = matrix(as.numeric(theta[j, ]), ncol = dim(theta)[2])
  #       X.i = matrix(as.numeric(X[i, ]), ncol = dim(X)[2])
  #       # Note that theta is already a row vector (transpose of the column vector theta)
  #       # X.i needs to be transposed as it is read as a row vector
  #       J <- J + (1/2)*(theta.j%*%t(X.i) - Y[i, j])^2
  #     } 
  #   }
  # }
  
  # Vectorize implementation
  
  J = (1/2) * sum( R*((tcrossprod(data.matrix(X), data.matrix(theta)) - Y)^2))
  
  
  # X gradient Gradient (iterative implementation): regularized 
  # for (i in seq(dim(X)[1])){
  #   
  #   tmp = rep(0, dim(X)[2])
  #   
  #   for (k in seq(num_features)) {
  #     tmp[k] = 0
  #     for (j in seq(dim(theta)[2])) {
  #       
  #       if(R[i, j] == 1){
  #         theta.j = matrix(as.numeric(theta[j, ]), ncol = dim(theta)[2])
  #         X.i = matrix(as.numeric(X[i, ]), ncol = dim(X)[2])
  #         tmp[k] <- tmp[k] + (theta.j%*%t(X.i) - Y[i, j])*theta.j[k] #+ (lambda*X.i[k])
  #       }
  #     }
  #   }
  #   X.grad[i, ] = tmp + lambda*matrix(as.numeric(X[i, ]), ncol = dim(X)[2])
  # }
  # Vectorize implementation of X gradient
  
  for (i in seq(dim(X)[1])){
    idx = which(R[i, ] != 0 )
    Y_temp = Y[i,idx]
    theta_temp = theta[idx, ]
    
    # X[i, ] * transpose(theta_tmp) - Y_tmp
    tmp = data.matrix(X[i, ]) %*% t(data.matrix(theta_temp)) - Y_temp
    
    X.grad[i, ] = data.matrix(tmp)%*%data.matrix(theta_temp) + lambda*data.matrix(X[i, ])
    
  }
  
  # Implementation of theta gradient
  
  # for(j in seq(dim(theta)[1])){
  #   tmp = rep(0, dim(theta)[2])
  #   
  #   for (k in seq(num.features)) {
  #     tmp[k] = 0
  #     for (i in seq(dim(X)[1])) {
  #       if(R[i, j] == 1){
  #         theta.j = data.matrix(theta[j, ])
  #         X.i = data.matrix(X[i, ])
  #         tmp[k] <- tmp[k] + (theta.j%*%t(X.i) - Y[i, j])*X.i[k]
  #       }
  #     }
  #   }
  #   theta.grad[j, ] = tmp + lambda*data.matrix(theta[j, ])
  # }
  
  # Vectorized implementation!
  
  for(j in seq(dim(theta)[1])){
    idx = which(R[ ,j] == 1 )
    Y_temp = Y[idx, j]
    X_temp = X[idx, ]
    tmp = data.matrix(X_temp) %*% t(data.matrix(theta[j, ])) - Y_temp
    theta.grad[j, ] = data.matrix(t(data.matrix(tmp))%*%data.matrix(X_temp) + lambda*theta[j, ])
  }
  
  J = J + (lambda/2)*( sum(X^2) + sum(theta^2))
  
  list(cost = J, gradient.theta = theta.grad, gradient.X = X.grad)
}

LoadMovieList <- function(filename){
  con = file("movie_ids.txt", open='r')
  line <- readLines(con = con)
  close(con)
  mlist = vector("list", length(line))
  for (i in seq(length(line))) {
    spltstr <- strsplit(line[i], " ")[[1]]
    mlist[[i]] <- paste(spltstr[2:length(spltstr)], collapse = " ")
  }
  mlist
}

NormalizeRatings <- function(Y, R){
  m = dim(Y)[1]
  n = dim(Y)[2]
  
  Ymean = matrix(rep(0, m), ncol = 1, nrow = m)
  Ynorm = matrix(rep(0, length(Y)), nrow = dim(Y)[1], ncol = dim(Y)[2])
  
  for (i in seq(m)) {
    idx = which(R[i, ] != 0)
    Ymean[i, ] = apply(Y[i, ], 1, function(x){mean(x[x!= 0])})
    Ynorm[i, idx] = as.vector(as.numeric(Y[i, idx] - Ymean[i, 1]))
  }
  list(ynorm = Ynorm, ymean = Ymean)
}


GradientDescent <- function(params, Y, R, num.features, lambda, alpha, num.iterations){
  
  # We use batch gradient descent as the size of our traning data set is relatively small. 
  
  # Use gradient descent to minimise the cost function with respect to parameters theta
  # and features x^i.  
  # param: params: A list containing data frames X and theta; X the first element and theta is the second
  # param: Y: the movies rating matrix, one row per movie
  # param: R: A binary matrix with R_i,j = 1, if user i rated movie j; 0 otherwise
  # param: alpha: the learning rate
  # param: num.iterations: the number of gradient descent iterations
  # param: lambda: the regularization parameter lambda
  # return: In order a list containing the cost, theta, and X (both theta and X are matrices)
  
  num.movies = dim(Y)[1]
  num.users = dim(Y)[2]
  
  param = params
  
  J_history = rep(0, num.iterations)
  
  for (it in seq(num.iterations)) {
    # params, Y, R, num.users, num.features, num.movies, lambda
    result = CostFunction(param, Y, R, num.users, num.features, num.movies, lambda)
    
    J_history[it] = result$cost
    
    param$X = param$X - alpha*result$gradient.X
    
    param$theta = param$theta - alpha*result$gradient.theta
    
    print(paste("Iteration", it, "Cost:", result$cost, collapse = " "))
  }
  
  list(cost.history = J_history, X = param$X, theta = param$theta)
}

Train <- function(Y, R, lambda = 1, alpha = 0.01, num.features = 3, num.iterations = 20){
  
  num.movies = dim(Y)[1]
  num.users = dim(Y)[2]
  
  set.seed(123)
  X = matrix(scale(rnorm(num.movies*num.features, mean = 0, sd = 1)), nrow = num.movies, ncol = num.features)
  set.seed(123)
  theta = matrix(scale(rnorm(num.users*num.features, mean = 0, sd = 1)), nrow = num.users, ncol = num.features)
  
  params = list(X = as.data.frame(X), theta = as.data.frame(theta))
  
  result = GradientDescent(params, Y, R, num.features, lambda, alpha, num.iterations)
  
  result
}

# movies ratings by users
Y = read.csv("movies_users_rating_data.csv", header = F, stringsAsFactors = F)
# binary value indicator, R[i,j] = 1 if user j has rated movie i, 0 otherwise
R = read.csv("binary_value_indicator_data.csv", header = F, stringsAsFactors = F)

movies <- LoadMovieList("movie_ids.txt")

my.ratings = matrix(rep(0, 1682), ncol = 1, nrow = 1682)

my.ratings[1, 1] = 4

my.ratings[98, 1] = 2

my.ratings[7, 1] = 3
my.ratings[12, 1] = 5
my.ratings[54, 1] = 4
my.ratings[64, ] = 5
my.ratings[66, ] = 3
my.ratings[69, ] = 5
my.ratings[183, ] = 4
my.ratings[226, ] = 5
my.ratings[355, ] = 5


for (i in seq(length(my.ratings))) {
  if(my.ratings[i, 1] > 0){
    print(paste("Rated", my.ratings[i, 1], "for", movies[[i]], collapse = " "))
  }
}


Y = cbind(Y, my.ratings) 

R = cbind(R, ifelse(my.ratings != 0, 1, 0))

Z = NormalizeRatings(Y, R)
Ynorm = Z$ynorm
Ymean = Z$ymean
num.features = 10

lambda = 10

alpha = 0.001

res <- Train(Ynorm, R, lambda, alpha, num.features, 100)

Xt = data.matrix(res$X)
thetat = data.matrix(res$theta)


pred = Xt %*%t(thetat) 

my.predictions <- pred[, 944] + Ymean

sorted <- sort.int(my.predictions, decreasing = T, index.return = T)

# Output the top 10 ratings

print("Top recommandations for you.")
for(i in seq(10)){
  
  strout = paste("Predicting Rating", round(sorted$x[i], 2), "for movie", movies[[sorted$ix[i]]], sep = " ", collapse = " ")
  print(strout)
}

print("Original ratings provided")

for (i in seq(length(my.ratings))) {
  if (my.ratings[i] > 0 ){
    strout = paste("Rated", my.ratings[i], "for", movies[[i]], sep = " ", collapse = " ")
    print(strout)
  }
}