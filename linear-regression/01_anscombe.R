X1 = c(10.0,8.0,13.0,9.0,11.0,14.0,6.0,4.0,12.0,7.0,5.0)
Y1 = c(8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68)
D1 = data.frame(X1,Y1)

X2 = c(10.0,8.0,13.0,9.0,11.0,14.0,6.0,4.0,12.0,7.0,5.0)
Y2 = c(9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74)
D2 = data.frame(X2,Y2)

X3 = c(10.0,8.0,13.0,9.0,11.0,14.0,6.0,4.0,12.0,7.0,5.0)
Y3 = c(7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73)
D3 = data.frame(X3,Y3)

X4 = c(8.0,8.0,8.0,8.0,8.0,8.0,8.0,19.0,8.0,8.0,8.0)
Y4 = c(6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.50,5.56,7.91,6.89)
D4 = data.frame(X4,Y4)

lm1 = lm(Y1~X1)
lm2 = lm(Y2~X2)
lm3 = lm(Y3~X3)
lm4 = lm(Y4~X4)

# graph these 4 datasets
library(ggplot2)
library(grid)
library(gridExtra)

multiplot <- function(..., title='', cols=1) {
  grid.newpage()
  
  # Make a list from the ... argument
  plots <- c(list(...))
  
  pushViewport(viewport(layout=grid.layout(3, cols, heights=unit(c(1, 4), "null"))))

  grid.text(title,  vp=viewport(layout.pos.row=1, layout.pos.col=1:cols), gp=gpar(fontsize=22))
 
  print(plots[[1]], vp=viewport(layout.pos.row=2, layout.pos.col=1))
  print(plots[[2]], vp=viewport(layout.pos.row=2, layout.pos.col=2))
}

fix <- function(n,k) {
  format(round(n,k), nsmall=k)
}

png(file='images/fig_1-0.png', width=600, height=600)
t1 <- tableGrob(fix(t(D1),2), core.just='right')
t2 <- tableGrob(fix(t(D2),2), core.just='right')
t3 <- tableGrob(fix(t(D3),2), core.just='right')
t4 <- tableGrob(fix(t(D4),2), core.just='right')
grid.arrange(t1,t2,t3,t4, main="Anscombe's Quartet", ncol=1)
dev.off()

# Dataset I
# plot for points, linear regression model 
p1 <- ggplot(D1, aes(x=X1,y=Y1)) +
        geom_point(size=3, color='#E69F00') +
        geom_smooth(method='lm', color='#56B4E9') +
        ggtitle('Linear Regression')
# plot for points vs residuals
p2 <- ggplot(data.frame(X1, lm1$residuals), aes(x=X1,y=lm1$residuals)) +
        geom_point(size=3, color='#E69F00') +
        geom_hline(y=0, color='#56B4E9') +
        ggtitle('Residuals') +
        scale_y_continuous('Residuals')
png(file='images/fig_1-1.png', width=800, height=400)
multiplot(p1, p2, title="Anscombe's Quartet, Dataset I", cols=2)
dev.off()

# Dataset II
# plot for points, linear regression model 
p1 <- ggplot(D2, aes(x=X2,y=Y2)) +
        geom_point(size=3, color='#E69F00') +
        geom_smooth(method='lm', color='#56B4E9') +
        ggtitle('Linear Regression')
# plot for points vs residuals
p2 <- ggplot(data.frame(X2, lm2$residuals), aes(x=X2,y=lm2$residuals)) +
        geom_point(size=3, color='#E69F00') +
        geom_hline(y=0, color='#56B4E9') +
        ggtitle('Residuals') +
        scale_y_continuous('Residuals')
png(file='images/fig_1-2.png', width=800, height=400)
multiplot(p1, p2, title="Anscombe's Quartet, Dataset II", cols=2)
dev.off()

# Dataset III
# plot for points, linear regression model 
p1 <- ggplot(D3, aes(x=X3,y=Y3)) +
        geom_point(size=3, color='#E69F00') +
        geom_smooth(method='lm', color='#56B4E9') +
        ggtitle('Linear Regression')
# plot for points vs residuals
p2 <- ggplot(data.frame(X3, lm3$residuals), aes(x=X3,y=lm3$residuals)) +
        geom_point(size=3, color='#E69F00') +
        geom_hline(y=0, color='#56B4E9') +
        ggtitle('Residuals') +
        scale_y_continuous('Residuals')
png(file='images/fig_1-3.png', width=800, height=400)
multiplot(p1, p2, title="Anscombe's Quartet, Dataset III", cols=2)
dev.off()

# Dataset IV
# plot for points, linear regression model 
p1 <- ggplot(D4, aes(x=X4,y=Y4)) +
        geom_point(size=3, color='#E69F00') +
        geom_smooth(method='lm', color='#56B4E9') +
        ggtitle('Linear Regression')
# plot for points vs residuals
p2 <- ggplot(data.frame(X4, lm4$residuals), aes(x=X4,y=lm4$residuals)) +
        geom_point(size=3, color='#E69F00') +
        geom_hline(y=0, color='#56B4E9') +
        ggtitle('Residuals') +
        scale_y_continuous('Residuals')
png(file='images/fig_1-4.png', width=800, height=400)
multiplot(p1, p2, title="Anscombe's Quartet, Dataset IV", cols=2)
dev.off()
