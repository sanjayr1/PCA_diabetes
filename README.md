# PCA_diabetes
Using PCA to Examine Factors Predicting the Transition from Pre-diabetes to Diabetes


### Load Data

``` r
get_dataset <- function(target){
  dm_original <- read.csv("clean_dm_2_27_19.csv")[,-1]
  
  # define target
  available_targets <- c("diabetes_6","diabetes_12","diabetes_18","diabetes_24")
  non_targets <- Filter(Negate(is.null), c("genpatientid",lapply(available_targets, function(i) if (i != target) {i})))

  # drop non_target targets
  dm_one_target <- dm_original[,-which(names(dm_original) %in% non_targets)]
  
  # make target a string so random forest treats as classification not regression
  dm_one_target$diabetes_24 <- sapply(dm_one_target$diabetes_24,as.factor)
  return(dm_one_target)
}

target <- "diabetes_24"
dm <- get_dataset(target)
```

### Split Into Test and Train

``` r
set.seed(123454321) 
split_perc <- 0.7

# get split-percent number of randomly sampled row indices
indices <- sample(nrow(dm), split_perc*nrow(dm), replace = FALSE)

# split train from target and test from train
train        <- dm[indices,-which(names(dm) == target)]
train_target <- dm[indices, which(names(dm) == target)]
test         <- dm[-indices,-which(names(dm) == target)]
test_target  <- dm[-indices, which(names(dm) == target)]
```

### Create PCA Train and Test from Original

``` r
get_train_pca <- function(dat){
  # PCA
  dat[,1:17] <- log(dat[,1:17])
  dat_pca <- prcomp(dat, scale. = TRUE, center = TRUE)
  
  # add a variance and variance-percent feature
  dat_pca$var     <- dat_pca$sdev^2
  dat_pca$var_per <- round(dat_pca$var/sum(dat_pca$var)*100, 1)
  
  return(dat_pca)
}

get_test_pca <- function(dat,train_pca){
  # scale the first 17 on-one-hot-encoded columns
  dat[,1:17] <- log(dat[,1:17])
  return(predict(train_pca, newdata = dat))
}

train_pca <- get_train_pca(train)
test_pca  <- get_test_pca(test,train_pca)
```

### Determine Number of Principal Components for 95% Variance

``` r
# visualize the percent of variance by prinicipal components
barplot(train_pca$var_per, main="Scree Plot", xlab="Principal Component", ylab="Percent Variation")
```

![](project_dm_pca%5B6%5D_files/figure-markdown_github/unnamed-chunk-4-1.png)

``` r
plot(train_pca, type = 'l')
```

![](project_dm_pca%5B6%5D_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
plot(cumsum(train_pca$var_per), xlab = "Principal Component",
  ylab = "Cumulative Proportion of Variance Explained",type = "b")
```

![](project_dm_pca%5B6%5D_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
train_pca$var_per
```

    ##  [1] 19.4 13.2 10.7  8.7  6.6  5.7  4.9  4.5  3.4  3.4  3.3  3.2  2.9  2.6
    ## [15]  2.2  2.1  1.5  0.6  0.5  0.2  0.1  0.1  0.1  0.1  0.1  0.0  0.0  0.0
    ## [29]  0.0  0.0

### Count of Principal Components with 95% Variance

``` r
# from the graphs above it looks like the first 15 PC contribute 95% of the variance
num_comp_95_per_var <- 15
cat("the first", num_comp_95_per_var, "columns account for",
    sum(train_pca$var_per[1:num_comp_95_per_var]), "% of the variance")
```

    ## the first 15 columns account for 94.7 % of the variance

### Test Hypothesis

``` r
create_and_test_model <- function(variance,model_data,pred_data,num_comp){
  # train model
  start.time <- Sys.time()
  random_forest_model <- randomForest(train_target ~ ., data = model_data, importance = TRUE)
  time.taken <- Sys.time() - start.time
  
  # predict and get accuracy
  prediction <- predict(random_forest_model, pred_data, type = "class")
  accuracy   <- mean(prediction == test_target)
  print(list(comp_count=num_comp, variance=variance, execution_time=extract_numeric(time.taken), accuracy=accuracy))
  return(list(comp_count=num_comp, variance=variance, execution_time=extract_numeric(time.taken), accuracy=accuracy))
}

get_experiment_results <- function(num_comps,train,test,train_pca,test_pca){
  results <- list()
  for (num_comp in (num_comps+1):1){
    # first, run without PCA
    if (num_comp == num_comps+1)
      results[[num_comp]] <- create_and_test_model(100.0,train,test,num_comp)
    # special case, random forest will not accept just one column to train on, we pair PCA1 with PCA30 to approximate PCA1
    else if (num_comp == 1)
      results[[num_comp]] <- create_and_test_model(sum(train_pca$var_per[c(1,30)]),train_pca$x[,c(1,30)],test_pca[,c(1,30)],1)
    else
      results[[num_comp]] <- create_and_test_model(sum(train_pca$var_per[1:num_comp]),train_pca$x[,1:num_comp],test_pca[,1:num_comp],num_comp)
  }
  return(do.call(rbind.data.frame, results))
}
```

``` r
results <- get_experiment_results(num_comp_95_per_var,train,test,train_pca,test_pca)
```

    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 16
    ## 
    ## $variance
    ## [1] 100
    ## 
    ## $execution_time
    ## [1] 19.30469
    ## 
    ## $accuracy
    ## [1] 0.8587946

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 15
    ## 
    ## $variance
    ## [1] 94.7
    ## 
    ## $execution_time
    ## [1] 8.778408
    ## 
    ## $accuracy
    ## [1] 0.8464721

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 14
    ## 
    ## $variance
    ## [1] 92.5
    ## 
    ## $execution_time
    ## [1] 14.76981
    ## 
    ## $accuracy
    ## [1] 0.846159

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 13
    ## 
    ## $variance
    ## [1] 89.9
    ## 
    ## $execution_time
    ## [1] 7.81523
    ## 
    ## $accuracy
    ## [1] 0.8465839

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 12
    ## 
    ## $variance
    ## [1] 87
    ## 
    ## $execution_time
    ## [1] 7.081213
    ## 
    ## $accuracy
    ## [1] 0.8467628

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 11
    ## 
    ## $variance
    ## [1] 83.8
    ## 
    ## $execution_time
    ## [1] 9.970885
    ## 
    ## $accuracy
    ## [1] 0.8467852

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 10
    ## 
    ## $variance
    ## [1] 80.5
    ## 
    ## $execution_time
    ## [1] 5.828022
    ## 
    ## $accuracy
    ## [1] 0.8460919

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 9
    ## 
    ## $variance
    ## [1] 77.1
    ## 
    ## $execution_time
    ## [1] 5.206997
    ## 
    ## $accuracy
    ## [1] 0.8463826

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 8
    ## 
    ## $variance
    ## [1] 73.7
    ## 
    ## $execution_time
    ## [1] 4.414806
    ## 
    ## $accuracy
    ## [1] 0.846159

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 7
    ## 
    ## $variance
    ## [1] 69.2
    ## 
    ## $execution_time
    ## [1] 3.794467
    ## 
    ## $accuracy
    ## [1] 0.8463379

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 6
    ## 
    ## $variance
    ## [1] 64.3
    ## 
    ## $execution_time
    ## [1] 3.362526
    ## 
    ## $accuracy
    ## [1] 0.8449961

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 5
    ## 
    ## $variance
    ## [1] 58.6
    ## 
    ## $execution_time
    ## [1] 2.825038
    ## 
    ## $accuracy
    ## [1] 0.844437

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 4
    ## 
    ## $variance
    ## [1] 52
    ## 
    ## $execution_time
    ## [1] 2.422465
    ## 
    ## $accuracy
    ## [1] 0.8414402

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 3
    ## 
    ## $variance
    ## [1] 43.3
    ## 
    ## $execution_time
    ## [1] 1.938106
    ## 
    ## $accuracy
    ## [1] 0.8357151

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 2
    ## 
    ## $variance
    ## [1] 32.6
    ## 
    ## $execution_time
    ## [1] 1.713713
    ## 
    ## $accuracy
    ## [1] 0.7875433

    ## extract_numeric() is deprecated: please use readr::parse_number() instead
    ## extract_numeric() is deprecated: please use readr::parse_number() instead

    ## $comp_count
    ## [1] 1
    ## 
    ## $variance
    ## [1] 19.4
    ## 
    ## $execution_time
    ## [1] 1.709683
    ## 
    ## $accuracy
    ## [1] 0.615364

    ## extract_numeric() is deprecated: please use readr::parse_number() instead

### Visualize Results

``` r
# save and load
write.csv(file="results[6].csv", x=results)
results <- read.csv("results[6].csv")
```

``` r
results[1:3,]
```

    ##    X comp_count variance execution_time  accuracy
    ## 1  2          1     19.4       1.709683 0.6153640
    ## 2 21          2     32.6       1.713713 0.7875433
    ## 3  3          3     43.3       1.938106 0.8357151

``` r
ggplot(results, aes(x=comp_count, y=accuracy)) +
  geom_point(aes(color = variance))
```

![](project_dm_pca%5B6%5D_files/figure-markdown_github/unnamed-chunk-13-1.png)

``` r
ggplot(results, aes(x=comp_count, y=execution_time)) +
  geom_point(aes(color = variance))
```

![](project_dm_pca%5B6%5D_files/figure-markdown_github/unnamed-chunk-14-1.png)

``` r
ggplot(results, aes(x=variance, y=accuracy)) +
  geom_point(aes(color = variance))
```

![](project_dm_pca%5B6%5D_files/figure-markdown_github/unnamed-chunk-15-1.png)
