library(tm)

data = read.csv("titles0.csv", header=T, row.names=1)

corp = Corpus(DataframeSource(data))

tdm = TermDocumentMatrix(corp, control=list(stopwords=T, 
                                            stemming=T,
                                            bounds=list(global = c(2,Inf)), 
                                            removePunctuation=T))
#inspect(tdm)
library(gridExtra)
grid.table(as.matrix(tdm))
set.seed(1)
A = svd(as.matrix(tdm))
U = A$u      # term vectors
D = A$d      # singular values
V = A$v      # document vectors

# visualize the percentage of variance explained
# given the number of singular vectors
# sv = D^2/sum(D^2)
# plot(cumsum(sv) * 100.0,
#     pch=19,
#     col='blue',
#     xlab='singular vectors',
#     ylab='% var explained')


# #     LSI: plotting with reduced rank matrix, k=2
# #     ... which, by the way, explains over 50% of
# #         the variation... 
U = U[,1:2]  # k=2
D = D[1:2]   # k=2
V = V[,1:2]  # k=2
print(V)

# #          Terms
terms = data.frame(U[,1]*D[1], U[,2]*D[2], tdm$dimnames$Terms)
colnames(terms) <- c('x', 'y', 'terms')
plot(terms$x, terms$y, 
     col="#d8b365", 
     pch=16, 
     xlim=c(-3.5, 0.5), ylim=c(-1.0, 2.0), 
     main="A Demonstration of Latent Semantic Indexing", 
     sub=expression(paste(hat(q), ": query on differential, equations, and theory in ", A['k=2'], " space")),
     xlab="", ylab="")
abline(v=0,h=0,lty=3)
text(terms$x, terms$y, terms$terms, col="#d8b365", cex=0.9, pos=4)

# #          Documents
docs  = data.frame(V[,1]*D[1], V[,2]*D[2], tdm$dimnames$Docs)
colnames(docs) <- c('x', 'y', 'docs')
points(docs$x, docs$y, col="#5ab4ac", pch=17)
text(docs$x, docs$y, docs$docs, col="#5ab4ac", cex=0.9, pos=4)

# #          Calculating a new point (query)
q = rep(0, length(terms$terms))
# # query will contain "differenti" or "equat" or "theori"
q[which(terms$terms=="differenti" | terms$terms=="equat" | terms$terms=="theori")]=5
q_hat = q %*% U %*% solve(diag(D))
points(q_hat[1], q_hat[2], col="#ef6548", pch=16)
text(q_hat[1], q_hat[2], expression(hat(q)), col="#ef6548", cex=0.7, pos=4)
points(terms$x[c(5,7,17)], terms$y[c(5,7,17)], col="#ef6548", pch=16)
text(terms$x[c(5,7,17)], terms$y[c(5,7,17)], terms$terms[c(5,7,17)], col="#ef6548", cex=0.9, pos=4)

lines(c(0,q_hat[1]), c(0,q_hat[2]), col="#ef6548")

# #          Calculating cosine similarity
# #          (clear command window)
# library(lsa)
# nV <- rbind(V, n)
# cosine(t(nV))[,11]
# q
# #          Plot of 3 most similar customers
# c2 <- V[2,1:2]
# c4 <- V[4,1:2]
# c7 <- V[7,1:2]
# lines(c(0,c2[1]), c(0,c2[2]), col="red")
# lines(c(0,c4[1]), c(0,c4[2]), col="red")
# lines(c(0,c7[1]), c(0,c7[2]), col="red")