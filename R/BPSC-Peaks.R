data <- read.csv(file.choose()) ## C:\Users\HP\Desktop\上交\acadamic\lungCancer\AD\data  
tmp<-t(data[,-2])
tmp0<-data.frame(tmp[,tmp[1,]==0])
tmp1<-data.frame(tmp[,tmp[1,]==1])
names(tmp0)<-paste("HC",1:ncol(tmp0),sep="")
names(tmp1)<-paste("LUNG",1:ncol(tmp1),sep="")
final<-cbind(tmp0[-1,],tmp1[-1,])

library(BPSC)
group=c(rep(1,ncol(tmp0)),rep(2,ncol(tmp1)))
controlIds=which(group==1)
design=model.matrix(~group)
coef=2
res=BPglm(as.matrix(final), controlIds=controlIds, design=design, coef=coef, estIntPar=FALSE, useParallel=FALSE)
ss=summary(res)
bpres<-data.frame(ss$topTable)
names(bpres)<-c("tval","pval","fdr")
bpres<- na.omit(bpres)
bpres<-bpres[bpres$pval<0.05,]
depeak<-row.names(bpres)

######################################################################################

data2 <- read.csv(file.choose()) ## C:\Users\HP\Desktop\上交\acadamic\lungCancer\AD\data  
tmp2<-t(data2[,-2])
tmp20<-data.frame(tmp2[,tmp2[1,]==0])
tmp21<-data.frame(tmp2[,tmp2[1,]==1])
names(tmp20)<-paste("HC",1:ncol(tmp20),sep="")
names(tmp21)<-paste("LUNG",1:ncol(tmp21),sep="")
final2<-cbind(tmp20[-1,],tmp21[-1,])

group2=c(rep(1,ncol(tmp20)),rep(2,ncol(tmp21)))
controlIds2=which(group2==1)
design2=model.matrix(~group2)
coef2=2
res2=BPglm(as.matrix(final2), controlIds=controlIds2, design=design2, coef=coef2, estIntPar=FALSE, useParallel=FALSE)
ss2=summary(res2)
bpres2<-data.frame(ss2$topTable)
names(bpres2)<-c("tval","pval","fdr")
bpres2 <- na.omit(bpres2)
bpres2<-bpres2[bpres2$pval<0.05,]
depeak2<-row.names(bpres2)

######################################################################################

data3 <- read.csv(file.choose()) ## C:\Users\HP\Desktop\上交\acadamic\lungCancer\AD\data  
tmp3<-t(data3[,-2])
tmp30<-data.frame(tmp3[,tmp3[1,]==0])
tmp31<-data.frame(tmp3[,tmp3[1,]==1])
names(tmp30)<-paste("HC",1:ncol(tmp30),sep="")
names(tmp31)<-paste("LUNG",1:ncol(tmp31),sep="")
final3<-cbind(tmp30[-1,],tmp31[-1,])

group3=c(rep(1,ncol(tmp30)),rep(2,ncol(tmp31)))
controlIds3=which(group3==1)
design3=model.matrix(~group3)
coef3=2
res3=BPglm(as.matrix(final3), controlIds=controlIds3, design=design3, coef=coef3, estIntPar=FALSE, useParallel=FALSE)
ss3=summary(res3)
bpres3<-data.frame(ss3$topTable)
names(bpres3)<-c("tval","pval","fdr")
bpres3 <- na.omit(bpres3)
bpres3<-bpres3[bpres3$pval<0.05,]
depeak3<-row.names(bpres3)

######################################################################################

library(VennDiagram)
venn.diagram(x=list(Batch1=depeak,Batch2=depeak2,Batch3=depeak3),filename = "DExingeng.tiff",col = "black",fill = c("red","green","blue"),alpha = 0.50,cat.col =c("red","green","blue"),cat.cex = 1.2,cat.fontface = "bold",margin = 0.1)
