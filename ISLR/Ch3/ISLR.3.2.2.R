library(ISLR)

data = read.csv('../Data/Advertising.csv', row.names=1)

idx  = seq(1:nrow(data))

set.seed(322)
tr   = sample(idx, 3*length(idx)/4)

data.train = data[tr,]
data.test  = data[! idx %in% tr,]

# For this exercise, we don't really need to know
# what the intercept-only model looks like...
#
#lm.0 = lm(Sales ~ . -(TV+Radio+Newspaper), data)
#summary(lm.0)

# Step 1: Select the 1st predictor
lm.1 = lm(Sales ~ TV, data.train)
lm.2 = lm(Sales ~ Radio, data.train)
lm.3 = lm(Sales ~ Newspaper, data.train)

#         ... some plots ...
library(ggplot2)
# png(file='images/fig_1-1_1.png', height=600, width=600)
# plot(data.train$TV, data.train$Sales, pch=19, col='blue', main='TV vs Sales')
# abline(lm.1, col='red')
# dev.off()
p = ggplot(data.train, aes(x=data.train$TV,y=data.train$Sales)) +
      geom_point(size=3, color='#E69F00') +
      geom_smooth(method='lm', color='#56B4E9') +
      ggtitle('Advertising') +
      scale_x_continuous('TV') +
      scale_y_continuous('Sales')
ggsave(file='images/fig_1-1_1.png', plot=p)

png(file='images/fig_1-1_2.png', height=400, width=800)
par(mfrow=c(1,2))
plot(lm.1, which=c(1,2))
mtext('TV vs Sales', cex=1.5, side=3, line=-2, outer=T)
dev.off()

# png(file='images/fig_1-2_1.png', height=600, width=600)
# plot(data.train$Radio, data.train$Sales, pch=19, col='blue', main='Radio vs Sales')
# abline(lm.2, col='red')
# dev.off()
p = ggplot(data.train, aes(x=data.train$Radio,y=data.train$Sales)) +
  geom_point(size=3, color='#E69F00') +
  geom_smooth(method='lm', color='#56B4E9') +
  ggtitle('Advertising') +
  scale_x_continuous('Radio') +
  scale_y_continuous('Sales')
ggsave(file='images/fig_1-2_1.png', plot=p)

png(file='images/fig_1-2_2.png', height=400, width=800)
par(mfrow=c(1,2))
plot(lm.2, which=c(1,2))
mtext('Radio vs Sales', cex=1.5, side=3, line=-2, outer=T)
dev.off()

# png(file='images/fig_1-3_1.png', height=600, width=600)
# plot(data.train$Newspaper, data.train$Sales, pch=19, col='blue', main='Newspaper vs Sales')
# abline(lm.3, col='red')
# dev.off()
p = ggplot(data.train, aes(x=data.train$Newspaper,y=data.train$Sales)) +
  geom_point(size=3, color='#E69F00') +
  geom_smooth(method='lm', color='#56B4E9') +
  ggtitle('Advertising') +
  scale_x_continuous('Newspaper') +
  scale_y_continuous('Sales')
ggsave(file='images/fig_1-3_1.png', plot=p)

png(file='images/fig_1-3_2.png', height=400, width=800)
par(mfrow=c(1,2))
plot(lm.3, which=c(1,2))
mtext('Newspaper vs Sales', cex=1.5, side=3, line=-2, outer=T)
dev.off()

#         Which single predictor has lowest RSS?
summary(lm.1)[[6]]
# [1] 3.258656       ....    go with TV as 1st choice!
summary(lm.2)[[6]]
# [1] 4.274944
summary(lm.3)[[6]]
# [1] 5.09248

# Step 2: 2nd choice predictors?
lm.4 = lm(Sales ~ TV + Radio, data)
lm.5 = lm(Sales ~ TV + Newspaper, data)

#         Which pair of predictors has lowest RSS?
summary(lm.4)[[6]]
# [1] 1.681361      ....    go with TV+Radio!
summary(lm.5)[[6]]
# [1] 3.12072

# Step 3: 3d plot of Sales ~ TV + Radio
library(rgl)
open3d()
plot3d(x=data.train$TV, 
       y=data.train$Radio,
       z=data.train$Sales, 
       col='red', size=3, 
       xlab='TV', ylab='Radio', zlab='Sales')
planes3d(a=coef(lm.4)[2], 
         b=coef(lm.4)[3], 
         c=-1, 
         d=coef(lm.4)[1], 
         alpha=.5)
