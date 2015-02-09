### 0. let's just define that myplclust function here...
#myplclust <- function( hclust, lab=hclust$labels, lab.col=rep(1,length(hclust$labels)), hang=0.1,...){
#    ## http://rafalab.jhsph.edu/batch/myplclust.R
#    ## modifiction of plclust for plotting hclust objects *in colour*!
#    ## Copyright Eva KF Chan 2009
#    ## Arguments:
#    ##    hclust:   hclust object
#    ##    lab:      a character vector of labels of the leaves of the tree
#    ##    lab.col:  colour for the labels; NA=default device foreground colour
#    ##    hang:     as in hclust & plclust
#    ## Side effect:
#    ##    A display of hierarchical cluster with coloured leaf labels.
#    y <- rep(hclust$height,2)
#        x <- as.numeric(hclust$merge)
#        y <- y[which(x<0)]
#        x <- x[which(x<0)]
#        x <- abs(x)
#        y <- y[order(x)]
#        x <- x[order(x)]
#        plot( hclust, labels=FALSE, hang=hang, ... )
#        text( x=x, y=y[hclust$order]-(max(hclust$height)*hang), 
#              labels=lab[hclust$order], 
#              col=lab.col[hclust$order], 
#              srt=90, 
#              adj=c(1,0.5), 
#              xpd=NA, ... )
#}

library(gplots)

##  1. read in raw data
samsung.raw <- read.table('data/train/X_train.txt')
dim(samsung.raw)
names(samsung.raw)


##  2. clean up feature names first,
##     and then set them in our samsung.raw
##     NOTE chars like -() cannot be used as col names
##          data.frame() will clean them up, by substituting with '.'
tmp  <- read.table('data/features.txt')
names(samsung.raw) <- tmp$V2
samsung.raw <- data.frame(samsung.raw)
names(samsung.raw)


##  3. Further scrubbing is needed, as there are many columns
##     related to the FFT derived values that seem to be for X, Y and Z,
##     but do not have any discerning marker. These column names are duplicated!
## 303-344: fBodyAcc-bandsEnergy()-
## 382-423: fBodyAccJerk-bandsEnergy()-
## 461-502: fBodyGyro-bandsEnergy()-
samsung.raw <- subset(samsung.raw, select=-c(303:344,382:423,461:502))


##  4. read in the corresponding activity labels,
##     and add them in, too...
tmp <- read.table('data/train/y_train.txt')
labels <- c('walking', 'walkingup', 'walkingdown', 'sitting', 'standing', 'landing')
samsung.raw$activity <- as.factor(labels[tmp[,1]])
table(samsung.raw$activity)


##  5. read in the subject labels as well,
##     and add them in
tmp <- read.table('data/train/subject_train.txt')
samsung.raw$subject <- as.factor(tmp$V1)
head(samsung.raw)
table(samsung.raw$subject)
summary(samsung.raw)


##  6. split up the samsung.raw into
##     - training
##     - test
##     NOTE  let's leave out the column of subject identifiers
##           (that's the last column!)
set.seed(777)
r <- nrow(samsung.raw)
c <- ncol(samsung.raw)
idx <- sample(seq(1:r), r/2, replace=F) 
data.train <- samsung.raw[idx , -c]
data.test  <- samsung.raw[-idx, -c]


##  7. before modelling, some way to measure accuracy? 
##     https://github.com/fontanon/samsung-activity-analysis/blob/master/samsung-activity/src/reproducible_code.R#L32
confusion_matrix <- function(model, outcome, dataset, predict_type="class") {
    as.matrix(table(outcome, predict(model, dataset, type=predict_type)))
}

accuracy <- function(model, outcome, dataset, predict_type="class") {
    confusion.matrix <- confusion_matrix(model, outcome, dataset, predict_type)
    sum(diag(confusion.matrix)/sum(confusion.matrix))
}


### Modelling
##  8. logistic regression
model.logistic <- glm(data.train$activity ~., family="binomial", data=data.train)
summary(model.logistic)
 accuracy(model.logistic, data.test$activity, data.test, "response")

png('images/01_logistic.png', width=1000, height=1000)
par(mfrow=c(2,2), oma=c(0,0,2,0))
plot(model.logistic,
     sub.caption=paste("Classification: Logistic Regression, accuracy:", 
                        accuracy(model.logistic,
                                 data.test$activity,  
                                 data.test, "response")))  
dev.off()


##  9. tree function in {{tree}}
library(tree)
model.tree <- tree(activity~., data=data.train)
summary(model.tree)

png('images/02_tree_1.png', width=1000, height=800)
plot(model.tree);text(model.tree)
title("Classification: tree")
dev.off()

png('images/02_tree_2.png', width=1000, height=800)
par(mfrow=c(2,1), oma=c(0,0,2,0))
plot(cv.tree(model.tree, FUN=prune.tree, method="misclass"))
plot(cv.tree(model.tree))
title(main="Classification: tree",
      sub="Validation")
dev.off()

## better than logistic??
png('images/02_tree_3.png', width=1000, height=800)
textplot(confusion_matrix(model.tree, data.test$activity, data.test))
title(main="Classification: tree",
      sub=paste("Accuracy:", accuracy(model.tree,
                                      data.test$activity,  data.test)))  
dev.off()


## 10. rpart function in {{rpart}}
library(rpart)
model.rpart <- rpart(activity ~., data=data.train)
printcp(model.rpart)
summary(model.rpart)

png('images/03_rpart_1.png', width=1000, height=800)
plot(model.rpart, uniform=T)
text(model.rpart, use.n=T, all=T, cex=.8)
title("Classification: rpart")
dev.off()

png('images/03_rpart_2.png', width=1000, height=800)
plotcp(model.rpart)
title(main="Classification: rpart",
      sub="Validation")
dev.off()

png('images/03_rpart_3.png', width=1000, height=800)
textplot(confusion_matrix(model.rpart, data.test$activity, data.test))
title(main="Classification: rpart",
      sub=paste("Accuracy:", accuracy(model.rpart,
                                      data.test$activity,  data.test)))  
dev.off()

## 11. Boostrap AGGregatING tree
library(ipred)
model.baggedtree <- bagging(activity ~., data=data.train, coob=T)
#print(model.baggedtree)
png('images/04_bagged_tree.png', width=1000, height=800)
textplot(confusion_matrix(model.baggedtree, data.test$activity, data.test))
title(main="Classification: bagged tree",
      sub=paste("Accuracy:", accuracy(model.baggedtree,
                                      data.test$activity,  data.test)))  
dev.off()


## 12. Random Forest
library(randomForest)
model.forest <- randomForest(activity ~., data=data.train, prox=T)
#print(model.forest)
png('images/05_random_forest.png', width=1000, height=800)
textplot(confusion_matrix(model.forest, data.test$activity, data.test))
title(main="Classification: random forest",
      sub=paste("Accuracy:", accuracy(model.forest,
                                      data.test$activity,  data.test)))  
dev.off()
