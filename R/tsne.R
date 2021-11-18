suppressMessages(require(Seurat))
suppressMessages(require(ggplot2))
suppressMessages(require(cowplot))
suppressMessages(require(scater))
suppressMessages(require(scran))
suppressMessages(require(BiocParallel))
suppressMessages(require(BiocNeighbors))

af <- read.csv(file.choose(),row.names=1,header=FALSE)
#af<-af[,-1]
#af<-af[,-1]
af<-t(af)

af.data<-CreateSeuratObject(data.frame(af), meta.data = col)
library(SingleCellExperiment)
af.data <- as.SingleCellExperiment(af.data)

zzz<-runPCA(af.data, BSPARAM = BiocSingular::RandomParam())
zz2 <- runTSNE(zzz, dimred = "PCA")

p1<-plotTSNE(zz2, colour_by = "plate")+ggtitle("By batch before calibration")+theme(plot.title = element_text(size=16,face="bold"), axis.text = element_blank(), axis.ticks = element_blank(), legend.title = element_text(size=15), legend.key.size = unit(1.3,"lines"), legend.text = element_text(size=13))
p2<-plotTSNE(zz2, colour_by = "label")+ggtitle("By category before calibration")+theme(plot.title = element_text(size=16,face="bold"), axis.text = element_blank(), axis.ticks = element_blank(), legend.title = element_text(size=15), legend.key.size = unit(1.3,"lines"), legend.text = element_text(size=13))
gridExtra::grid.arrange(p1, p2, ncol = 2)