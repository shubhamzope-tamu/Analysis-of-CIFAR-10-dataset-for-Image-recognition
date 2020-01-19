#install.packages('devtools',repos = "http://cran.us.r-project.org")
#devtools::install_github("rstudio/keras")
#install.packages("tensorflow",repos = "http://cran.us.r-project.org")
#devtools::install_github("rstudio/tensorflow")
library(tensorflow)
#install_tensorflow()
#install_tensorflow(gpu = T)
#install.packages("ramify",repos="http://cran.us.r-project.org")
#devtools::install_github("bgreenwell/ramify")

library(reticulate)
#py_install("pandas")
#conda_install("r-reticulate", "Scipy")
#use_condaenv("r-reticulate")

library(keras)
#loading the keras inbuilt cifar10 dataset
labels <- read.table("batches.meta.txt")
#images.rgb <- list()
images.rgb <- array(numeric(),c(50000,32,32,3)) 
images.lab <- list()

images1.rgb <- array(numeric(),c(10000,32,32,3)) 
images1.lab <- list()



num.images = 10000 # Set to 10000 to retrieve all images per file to memory

# Cycle through all 5 binary files
for (f in 1:5) {
  to.read <- file(paste("data_batch_", f, ".bin", sep=""), "rb")
  for(i in 1:num.images) {
    l <- readBin(to.read, integer(), size=1, n=1, endian="big")
    r <- readBin(to.read, integer(), size=1, n=1024, endian="big")
    g <- readBin(to.read, integer(), size=1, n=1024, endian="big")
    b <- readBin(to.read, integer(), size=1, n=1024, endian="big")
    #r <- as.matrix(r,nrow=32,ncol=32)
    #g <- as.matrix(g,nrow=32,ncol=32)
    #b <-as.matrix(b,nrow=32,ncol=32)
    r <- r/255
    g <- g/255
    b <- b/255
    index <- num.images * (f-1) + i
    #images.rgb[[index]] = data.frame(r, g, b)
    images.rgb[index,,,] <- array( c( r ,g, b ),dim=c(32,32,3) )
    images.lab[[index]] = l
  }
  close(to.read)
  remove(l,r,g,b,f,i,index, to.read)
}

images.lab <- to_categorical(images.lab,num_classes = 10)
#train_x <- array(images.rgb,dim=c(50000,32,32,3))
train_x <- images.rgb
train_y <- array(images.lab,dim=c(50000,10))



##############################################

to.read <- file("test_batch.bin", "rb")
for(i in 1:num.images) {
    l <- readBin(to.read, integer(), size=1, n=1, endian="big")
    r <- readBin(to.read, integer(), size=1, n=1024, endian="big")
    g <- readBin(to.read, integer(), size=1, n=1024, endian="big")
    b <- readBin(to.read, integer(), size=1, n=1024, endian="big")
    #r <- as.matrix(r,nrow=32,ncol=32)
    #g <- as.matrix(g,nrow=32,ncol=32)
    #b <-as.matrix(b,nrow=32,ncol=32)
    r <- r/255
    g <- g/255
    b <- b/255
    index <- i
    #images.rgb[[index]] = data.frame(r, g, b)
    images1.rgb[index,,,] <- array( c( r ,g, b ),dim=c(32,32,3) )
    images1.lab[[index]] = l
  }
  close(to.read)
  remove(l,r,g,b,i, to.read)
cat("index")
cat(index)
images1.lab <- to_categorical(images1.lab,num_classes = 10)
#train_x <- array(images.rgb,dim=c(50000,32,32,3))
test_x <- images1.rgb
test_y <- array(images1.lab,dim=c(10000,10))




#set.seed(123)
#smp_size <- floor(0.9 * nrow(train_x))
#train_ind <- sample(seq_len(nrow(train_x)), size = smp_size)

#train_x <- train_x[train_ind,,, ]
#valid_x <- train_x[-train_ind,,, ]
#train_y <- train_y[train_ind,]
#valid_y <- train_y[-train_ind,]




 
cat("No of training samples\t",dim(train_x)[[1]],"\tNo of test samples\t",dim(test_x)[[1]])
model<-keras_model_sequential()
#configuring the Model
model %>%  
  #defining a 2-D convolution layer
  
  layer_conv_2d(filter=64,kernel_size=c(3,3),padding="same",                input_shape=c(32,32,3) ) %>%  
  layer_activation("relu") %>%  
  #another 2-D convolution layer
  
  layer_conv_2d(filter=64 ,kernel_size=c(3,3))  %>%  layer_activation("relu") %>%
  #Defining a Pooling layer which reduces the dimentions of the #features map and reduces the computational complexity of the model
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  #dropout layer to avoid overfitting
  layer_dropout(0.25) %>%
  layer_conv_2d(filter=32 , kernel_size=c(3,3),padding="same") %>% layer_activation("relu") %>%  layer_conv_2d(filter=32,kernel_size=c(3,3) ) %>%  layer_activation("relu") %>%  
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  layer_dropout(0.25) %>%
  #flatten the input  
  layer_flatten() %>%  
  layer_dense(512) %>%  
  layer_activation("relu") %>%  
  layer_dropout(0.5) %>%  
  #output layer-10 classes-10 units  
  layer_dense(10) %>%  
  #applying softmax nonlinear activation function to the output layer #to calculate cross-entropy  
  layer_activation("softmax")

  opt<-optimizer_adam( lr= 0.0001 , decay = 1e-6 )

model %>%
  compile(loss="categorical_crossentropy",
          optimizer=opt,metrics = "accuracy")
summary(model)



data_augmentation <- FALSE 
if(!data_augmentation) {  
  model %>% fit( train_x,train_y ,batch_size=10,
                 epochs=40,validation_data = list(test_x, test_y),
                 shuffle=TRUE)
} else {  
  #Generating images
  
  gen_images <- image_data_generator(featurewise_center = TRUE,
                                     featurewise_std_normalization = TRUE,
                                     rotation_range = 20,
                                     width_shift_range = 0.30,
                                     height_shift_range = 0.30,
                                     horizontal_flip = TRUE  )
  #Fit image data generator internal statistics to some sample data
  gen_images %>% fit_image_data_generator(train_x)
  #Generates batches of augmented/normalized data from image data and #labels to visually see the generated images by the Model
  model %>% fit_generator(
    flow_images_from_data(train_x, train_y,gen_images,
                          batch_size=32),
    steps_per_epoch=as.integer(50000/32),epochs =40,
    validation_data = list(test_x, test_y) )
}


#?dataset_cifar10 #to see the help file for details of dataset
#mean(keras_predict_classes(model,test_x) == (apply(test_y, 1, which.max) - 1))
#evaluate(model,test_x, test_y, verbose = 0)
model %>% save_model_tf("model_100")
new_model <- load_model_tf("model_100")
score <- new_model %>% evaluate(test_x, test_y, batch_size = 32)
score
library(ramify)
y_pred <- predict(new_model,test_x)
pred_class <- argmax(y_pred)
#pred_class
#cat("number of correct prediction")
true_class <- argmax(test_y)
#true_class

sum(pred_class == true_class) 
