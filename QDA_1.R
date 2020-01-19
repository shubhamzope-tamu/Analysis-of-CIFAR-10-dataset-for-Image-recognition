#Preprocessing of data from List to Data Frame
processed_image = list()
for (index in 1:50000){
  images = unlist(images.rgb[[index]],FALSE,FALSE)
  processed_image[[index]] = images
}
##head(processed_image)
processed_image1 = as.data.frame(do.call(rbind,processed_image))
p2<-as.data.frame(processed_image1)

#Dividing Colors from the whole Data Frame
p2_r<-p2[,1:1024]
p2_g<-p2[,1025:2048]
p2_b<-p2[,2049:3072]

#Using PCA for Different Colors
pr.out_r=prcomp(p2_r,scale=TRUE)
pr.out_g=prcomp(p2_g,scale=TRUE)
pr.out_b=prcomp(p2_b,scale=TRUE)
str(pr.out_r)
str(pr.out_g)
str(pr.out_b)
n=200
install.packages("devtools")
library(devtools)
install_github("vqv/ggbiplot")
pr.var_r=pr.out_r$sdev^2
pve_r=pr.var_r/sum(pr.var_r)
sum(pve_r[1:n])
plot(cumsum(pve_r),xlab="Principal components of Red",ylab="Cummulative Proportion of Variance Explained",ylim=c(0,1),type="b")
pr.var_g=pr.out_g$sdev^2
pve_g=pr.var_g/sum(pr.var_g)
sum(pve_g[1:n])
plot(cumsum(pve_g),xlab="Principal components of Green",ylab="Cummulative Proportion of Variance Explained",ylim=c(0,1),type="b")
pr.var_b=pr.out_b$sdev^2
pve_b=pr.var_b/sum(pr.var_b)
sum(pve_b[1:n])
plot(cumsum(pve_b),xlab="Principal components of Blue",ylab="Cummulative Proportion of Variance Explained",ylim=c(0,1),type="b")
p2_r_pca_x<-pr.out_r$x[,c(1:n)]
p2_g_pca_x<-pr.out_g$x[,c(1:n)]
p2_b_pca_x<-pr.out_b$x[,c(1:n)]
#Changing Labels for all Colors
for(i in 1:n){colnames(p2_r_pca_x)[i]<-paste('R',i,sep="")}
for(i in 1:n){colnames(p2_g_pca_x)[i]<-paste('G',i,sep="")}
for(i in 1:n){colnames(p2_b_pca_x)[i]<-paste('B',i,sep="")}
#Creating Final Data frame with most effective predictors
p2_pca<-cbind(p2_r_pca_x,p2_g_pca_x,p2_b_pca_x)
p3<-unlist(images.lab)
p4<-cbind(p2_pca,p3)
p4<-as.data.frame(p4)
#Splitting data to Train  and Test
size=sample(1:nrow(p4), nrow(p4) /(10/9))
train=p4[size,]
test=p4[-size,]
#Implementing Qda
library(MASS)
qda.fit<-qda(p3~.-p3,data=train)
summary(qda.fit)
qda.pred=predict(qda.fit,test)
qda.class=qda.pred$class
names(qda.pred)
table(qda.class,test$p3)
mean(qda.class==test$p3)
qda.pred=predict(qda.fit,train)
qda.class=qda.pred$class
table(qda.class,train$p3)
mean(qda.class==train$p3)