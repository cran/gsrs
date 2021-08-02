#' Train the group-specific model and test model performance
#' @author Yifei Zhang, Xuan Bi
#' @references Xuan Bi, Annie Qu, Junhui Wang & Xiaotong Shen
#' A Group-Specific Recommender System, Journal of the American Statistical Association, 112:519, 1344-1353
#' DOI: 10.1080/01621459.2016.1219261. Please contact the author should you encounter any problems
#' A fast version written in Matlab is available at https://sites.google.com/site/xuanbigts/software.
#' @description This gssvd() function uses ratings dataset to train a group-specific recommender system, tests the performance, and output the key matrix for prediction.
#' To make the training process run in parallel, doParallel package is recommended to use.
#' For more details regarding how the simulated dataset created, please refer to http://dx.doi.org/10.1080/01621459.2016.1219261.
#' @import MASS
#' @import foreach
#' @import doParallel
#' @importFrom stats coef quantile rnorm
#' @param train Train set, a matrix with three columns (userID, movieID, ratings)
#' @param test Test set, a matrix with three columns (userID, movieID, ratings)
#' @param B Number of user groups, 10 by default, don't need to specify if user_group prarmeter is not NULL
#' @param C Number of item groups, 10 by default, don't need to specify if item_group prarmeter is not NULL
#' @param K Number of latent factors
#' @param tol_1 The stopping criterion for outer loop in the proposed algorithm, 1e-3 by default
#' @param tol_2 The stopping criterion for sub-loops, 1e-5 by default
#' @param lambda Value of penalty term in ridge regression for ALS, 2 by default
#' @param max_iter Maximum number of iterations in the training process, 100 by default
#' @param verbose Boolean, if print out the detailed intermediate computations in the training process, 0 by default
#' @param user_group Optional parameter, should be a n-dim vector, n is total number of users, each element in the vector represents the group ID for that user (We will use missing pattern if not specified)
#' @param item_group Optional parameter, should be a m-dim vector, m is total number of items, each element in the vector represents the group ID for that item (We will use missing pattern if not specified)
#' @return Return the list of result, including matrix \code{P}, \code{Q}, \code{S}, \code{T} and RMSE of test set (\code{RMSE_Test})
#' @examples ## Training model on the simulated data file
#' library(doParallel)
#' registerDoParallel(cores=2)
#' # CRAN limits the number of cores available to packages to 2,
#' # you can use cores = detectCores()-1 in the real work setting.
#' getDoParWorkers()
#' example_data_path = system.file("extdata", "sim_data.txt", package="gsrs")
#' ratings = read.table(example_data_path, sep =":", header = FALSE)[1:100,]
#' # Initialization Parameters
#' K=3
#' B=10
#' C=10
#' lambda = 2
#' max_iter = 1 # usually more than 10;
#' tol_1=1e-1
#' tol_2=1e-1
#' # Train Test Split
#' N=dim(ratings)[1]
#' test_rate = 0.3
#' train.row=which(rank(ratings[, 1]) <= floor((1 - test_rate) * N))
#' test.row=which(rank(ratings[, 1]) > floor((1 - test_rate) * N))
#' train.data=ratings[train.row,1:3]
#' test.data=ratings[test.row,1:3]
#' # Call gssvd function
#' a = gssvd(train=train.data, test=test.data, B=B, C=C, K=K,
#' lambda=lambda, max_iter=max_iter, verbose=1)
#' stopImplicitCluster()
#' # Output the result
#' a$RMSE_Test
#' head(a$P)
#' head(a$Q)
#' head(a$S)
#' head(a$T)
#' @export
gssvd = function(train,test,B=10,C=10,K,tol_1=1e-3,tol_2=1e-5,lambda=2, max_iter=100,verbose=0,user_group=NULL,item_group=NULL){

  #Re-orgnize dataset
  train.data = train
  test.data = test
  full.data = rbind(train.data, test.data)

  #Parameters

  n=max(full.data[,1])
  m=max(full.data[,2])
  N=dim(full.data)[1]

  max_iter = max_iter
  K=K
  lambda2 = lambda
  B=B
  C=C
  stopping_outer = tol_1
  stopping_inner = tol_2

  #global variables(only use to remove notes in package checking)

  i=0
  b=0
  u=0

  #Group info (Based on missing pattern if no specific group info provided)
  #For user group
  if(!is.null(user_group)){
    group.user=user_group
  } else {
    num.rat=as.vector(table(factor(train.data[,1], levels=1:(n))))
    group.user=grouping(num.rat,quantile(num.rat,c(0:B)/B))

  }

  #For item group
  if(!is.null(item_group)){
    group.item=item_group
  } else {
    num.movie_rat=as.vector(table(factor(train.data[,2], levels=1:(m))))
    group.item=grouping(num.movie_rat,quantile(num.movie_rat,c(0:C)/C))
  }

  #Factor Matrix Initialization
  Uals=matrix(0,n,K)
  Uals[,]=matrix(rnorm(n*K,0,0.3),n,K)  # P

  Vals=matrix(rnorm(m*K,0,0.3),m,K)     # Q

  Oals=matrix(0,B,K)
  Oals[,]=matrix(rnorm(B*K,0,0.3),B,K)  # S

  Gals=matrix(rnorm(C*K,0,0.3),C,K)     # T

  #Start training
  diff=1
  iter=1

  while(diff>stopping_outer&iter<=max_iter){ # Stop criterion in step 4

    print(c("Iterition:",noquote(iter)))
    UandO0=(Uals+Oals[group.user[],])    #P+S
    VandG0=(Vals+Gals[group.item[],])    #Q+T
    UandO=UandO0

    #Subloop 1

    if(verbose){print("Sub-loop 1 begins")}
    diff.item=1

    while(diff.item>stopping_inner){

      t1=train.data[,3]-as.matrix(UandO%*%t(Gals[group.item[],]))[(train.data[,2]-1)*n+train.data[,1]] # t1=y-(P+S)*(T)

      Vals_new=foreach(i=1:m,.combine='cbind') %dopar%{
        pref(K,t1[(train.data[,2]==i)],UandO[train.data[train.data[,2]==i,1],],lambda2) #coefs of ridge regression
      } #Notes: pref(K,t1[(train.data[,2]==i)],UandO[train.data[train.data[,2]==i,1],],lambda2) | y -> t1[(train.data[,2]==i)] | x -> UandO[train.data[train.data[,2]==i,1],]
      Vals_new=t(Vals_new) # Ridge Regression update q
      rm(t1)

      q1=(train.data[,3]-as.matrix(UandO%*%t(Vals_new[]))[(train.data[,2]-1)*n+train.data[,1]])        # q1=y-(P+S)*(Q)

      Gals_new=foreach(c=1:C,.combine='cbind') %dopar%{
        pref(K,q1[group.item[train.data[,2]]==c],UandO[train.data[group.item[train.data[,2]]==c,1],],lambda2)
      }
      Gals_new=t(Gals_new) # Ridge Regression update t

      rm(q1)

      diff.item=sum(as.matrix(Vals_new-Vals)^2)/m/K+sum(as.matrix(Gals_new-Gals)^2)/C/K

      if(verbose){print(diff.item)}

      Vals=Vals_new # update Q
      Gals=Gals_new # update T
      rm(Vals_new);rm(Gals_new)
    }
    rm(UandO)

    VandG=(Vals+Gals[group.item[],])

    #Subloop 2

    if(verbose){print("Sub-loop 2 begins")}
    diff.user=1

    while(diff.user>stopping_inner){
      s1=(train.data[,3]-as.matrix(Oals[group.user[],]%*%t(VandG[]))[(train.data[,2]-1)*n+train.data[,1]])# s1=y-(Q+T)*(S)

      Uals_new=foreach(u=1:n,.combine='cbind') %dopar%{
        pref(K,s1[(train.data[,1]==u)],VandG[train.data[train.data[,1]==u,2],],lambda2)
      }
      Uals_new=(t(Uals_new))
      rm(s1)

      p1=(train.data[,3]-as.matrix(Uals_new%*%t(VandG[]))[(train.data[,2]-1)*n+train.data[,1]])           # p1=y-(Q+T)*(S)
      Oals_new=foreach(b=1:B,.combine='cbind') %dopar%{
        pref(K,p1[group.user[train.data[,1]]==b],VandG[train.data[group.user[train.data[,1]]==b,2],],lambda2)
      }
      Oals_new=(t(Oals_new))
      rm(p1)

      diff.user=sum(as.matrix(Uals_new-Uals)^2)/n/K+sum(as.matrix(Oals_new-Oals)^2)/B/K

      if(verbose){print(diff.user)}

      Uals=Uals_new
      Oals=Oals_new
      rm(Uals_new);rm(Oals_new)
    }
    rm(VandG)

    #Compute if converge in Step 4
    diff=sum((Uals+Oals[group.user[],]-UandO0)^2)/n/K+sum((Vals+Gals[group.item[],]-VandG0)^2)/m/K

    if(verbose) {print(c("Diff.all:",diff))}

    iter=iter+1;
    if(iter>max_iter) {
      print('Maximum number of iterations achieved!\n')
      break
    }
  }

  if(verbose) {print(c("Training process completed, now difference = ",diff))}
  print("Training process completed!\n")

  n=max(train.data[,1])
  ysgi=(Uals+Oals[group.user[],])%*%(t(Vals+Gals[group.item[],]))
  ysgi.test=(as.matrix(ysgi)[(test.data[,2]-1)*n+test.data[,1]])
  RMSE.test=sqrt(mean(as.matrix(test.data[,3]-ysgi.test)^2))
  rm(ysgi);rm(ysgi.test)

  p = Uals
  q = Vals
  s = Oals[group.user[],] # S matrix for every users
  t = Gals[group.item[],] # T matrix forevery items
  pred_result = RMSE.test

  #Output value
  result_list <- list(P=p,Q=q,S=s,T=t,RMSE_Test=pred_result)

  return(result_list)
  }

#Two helper functions

#assign group id based on info
grouping<-function(info,cutoff){
  group.id=rep(0,length(info))
  num.group=length(cutoff)-1
  for(i in 1:(num.group-1))
    group.id[(info>=cutoff[i])&(info<cutoff[i+1])]=i
  group.id[(info>=cutoff[num.group])&(info<=cutoff[num.group+1])]=num.group

  return(group.id)
}

#calculate item preference for "foreach" function
pref=function(K,y,x,lambdaALS){
  num=length(y)
  if(num==0)
    prefer=rep(0,K)
  if(num==1)
    prefer=solve(x%*%t(x)+diag(lambdaALS,K))%*%(x*y)
  if(num>=2)
    prefer=as.numeric(coef(lm.ridge(y~x-1,lambda=lambdaALS))) # Why minus 1 ?
  return(prefer)
}
