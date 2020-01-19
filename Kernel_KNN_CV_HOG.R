labels <- read.table("batches.meta.txt")
images.rgb <- list()
images.lab <- list()
num.images = 10000 # Set to 10000 to retrieve all images per file to memory

# Cycle through all 5 binary files
for (f in 1:5) {
  to.read <- file(paste("data_batch_", f, ".bin", sep=""), "rb")
  for(i in 1:num.images) {
    l <- readBin(to.read, integer(), size=1, n=1, endian="big")
    r <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    g <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    b <- as.integer(readBin(to.read, raw(), size=1, n=1024, endian="big"))
    index <- num.images * (f-1) + i
    images.rgb[[index]] = data.frame(r, g, b)
    images.lab[[index]] = l+1
  }
  close(to.read)
  remove(l,r,g,b,f,i,index, to.read)
}

#function to run sanity check on photos & labels import
drawImage <- function(index) {
  # Testing the parsing: Convert each color layer into a matrix,
  # combine into an rgb object, and display as a plot
  img <- images.rgb[[index]]
  img.r.mat <- matrix(img$r, ncol=32, byrow = TRUE)
  img.g.mat <- matrix(img$g, ncol=32, byrow = TRUE)
  img.b.mat <- matrix(img$b, ncol=32, byrow = TRUE)
  img.col.mat <- rgb(img.r.mat, img.g.mat, img.b.mat, maxColorValue = 255)
  dim(img.col.mat) <- dim(img.r.mat)
  
  # Plot and output label
  library(grid)
  grid.raster(img.col.mat, interpolate=FALSE)
  
  # clean up
  remove(img, img.r.mat, img.g.mat, img.b.mat, img.col.mat)
  
  labels[[1]][images.lab[[index]]]
}

drawImage(sample(1:(num.images*5), size=1))

processed_image = list()
for (index in 1:50000){
  images = unlist(images.rgb[[index]],FALSE,FALSE)
  processed_image[[index]] = images
  
}

x = as.data.frame(do.call(rbind,processed_image))
y = unlist(images.lab)
df = cbind.data.frame(x,y)
New_10 = df
x = New_10[, -ncol(New_10)]
y = New_10[, ncol(New_10)]


acc = function (ytrue, ycap) {
  
  out = table(ytrue, max.col(ycap, ties.method = "random"))
  
  acc = sum(diag(out))/sum(out)
  
  acc
}


install.packages("irlba")
install.packages("KernelKnn")
install.packages("OpenImageR")
library(irlba)
library(KernelKnn)
library(OpenImageR)


hog = HOG_apply(x, cells = 6, orientations = 9, rows = 32,
                
                columns = 96, threads = 6)

fit_hog = KernelKnnCV(hog, y, k = 20, folds = 4, method = 'braycurtis',
                      
                      weights_function = 'biweight_tricube_MULT', regression = F,
                      
                      threads = 6, Levels = sort(unique(y)))

acc_fit_hog = unlist(lapply(1:length(fit_hog$preds), 
                            
                            function(x) acc(y[fit_hog$folds[[x]]], 
                                            
                                            fit_hog$preds[[x]])))


acc_fit_hog



cat('mean accuracy for hog-features using cross-validation :', mean(acc_fit_hog), '\n')
  
