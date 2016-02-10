install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")

require(xgboost)
set.seed(1)
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test  <- agaricus.test

param <- list("objective" = "binary:logistic",
              "eval_metric" = "logloss",
              "eta" = 1, "max.depth" = 2)
bst.cv = xgb.cv(param=param, data = as.matrix(train$data), label = train$label, nfold = 10, nrounds = 20)
plot(log(bst.cv$test.logloss.mean),type = "l")
bst <- xgboost(data = as.matrix(train$data), label = train$label, max.depth = 2, eta = 1, nround = 5,
               nthread = 2, objective = "binary:logistic")

preds=predict(bst,test$data)
print(-mean(log(preds)*test$label+log(1-preds)*(1-test$label)))
trees = xgb.model.dt.tree(dimnames(train$data)[[2]],model = bst)

# Get the feature real names
names <- dimnames(train$data)[[2]]
importance_matrix <- xgb.importance(names, model = bst)
xgb.plot.importance(importance_matrix[1:10])
xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2)