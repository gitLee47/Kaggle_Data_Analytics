#Kaggle
install.packages("Metrics")

library(data.table)
library(Matrix)
library(xgboost)
library(Metrics)

ID = 'id'
TARGET = 'loss'
SEED = 0
SHIFT = 200

TRAIN_FILE = "C:/Study/kaggle_Pro/train.csv"
TEST_FILE = "C:/Study/kaggle_Pro/test.csv"
SUBMISSION_FILE = "C:/Study/kaggle_Pro/sample_submission.csv"


train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

y_train = log(train[,TARGET, with = FALSE] + SHIFT)[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]

ntrain = nrow(train)
train_test = rbind(train, test)

features = names(train)

for (f in features) {
  if (class(train_test[[f]])=="character") {
    #cat("VARIABLE : ",f,"\n")
    levels <- sort(unique(train_test[[f]]))
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}

# in order to speed up fit within Kaggle scripts have removed 30
# least important factors as identified from local run
features_to_drop <- c("cat67","cat21","cat60","cat65", "cat32", "cat30",
                      "cat24", "cat74", "cat85", "cat17", "cat14", "cat18",
                      "cat59", "cat22", "cat63", "cat56", "cat58", "cat55",
                      "cat33", "cat34", "cat46", "cat47", "cat48", "cat68",
                      "cat35", "cat20", "cat69", "cat70", "cat15", "cat62")

x_train = train_test[1:ntrain,-features_to_drop, with = FALSE]
x_test = train_test[(ntrain+1):nrow(train_test),-features_to_drop, with = FALSE]

## for 1113 local run comment out above and uncoment below
#x_train = train_test[1:ntrain,]
#x_test = train_test[(ntrain+1):nrow(train_test),]


dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest = xgb.DMatrix(as.matrix(x_test))


xgb_params = list(
  seed = 0,
  colsample_bytree = 0.5,
  subsample = 0.8,
  eta = 0.01, # replace this with 0.01 for local run to achieve 1113.93
  objective = 'reg:linear',
  max_depth = 12,
  alpha = 1,
  gamma = 2,
  min_child_weight = 1,
  base_score = 7.76
)


xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}


## Commented out in order to run within Kaggle sript time restrictions
## Uncoment and run locally to get best_nrounds of 2813 if used with eta = 0.01
res = xgb.cv(xgb_params,
             dtrain,
             nrounds=3000,
             nfold=5,
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 1,
             feval=xg_eval_mae,
             maximize=FALSE)

best_nrounds = res$best_iteration # for xgboost v0.6 users 
#best_nrounds = which.min(res$dt[, test.error.mean]) # for xgboost v0.4-4 users 

# established best _nrounds with eta=0.05 from a local cv run 
best_nrounds = 2813 # comment this out when doing local 1113 run

cv_mean = res$evaluation_log$test_error_mean[best_nrounds]
cv_std = res$evaluation_log$test_error_std[best_nrounds]
cat(paste0('CV-Mean: ',cv_mean,' ', cv_std))

gbdt = xgb.train(xgb_params, dtrain, nrounds=as.integer(best_nrounds/0.8))

submission = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
submission$loss = exp(predict(gbdt,dtest)) - SHIFT
write.csv(submission,'C:/Study/kaggle_Pro/xgb_starter_v3.sub.csv',row.names = FALSE)
