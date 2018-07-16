

1 단계 : 데이터 수집
 Pregnancies             : int  4 5 1 2 1 1 1 5 1 2 ...
 Glucose                 : int  99 88 114 91 88 172 144 124 97 89 ...
 BloodPressure           : int  72 78 66 62 62 68 82 74 66 90 ...
 SkinThickness           : int  17 30 36 0 24 49 46 0 15 30 ...
 Insulin                 : int  0 0 200 0 44 579 180 0 140 0 ...
 BMI                     : num  25.6 27.6 38.1 27.3 29.9 42.4 46.1 34 23.2 33.5 ...
 DiabetesPedigreeFunction: num  0.294 0.258 0.289 0.525 0.422 0.702 0.335 0.22 0.487 0.292 ...
 Age                     : int  28 37 21 22 23 28 46 38 22 42 ...
 Outcome                 : int  0 0 0 0 0 1 1 1 0 0 ...

 
2 단계  : 데이터 준비


```{r}
diabets <- read.csv("c:/r/diabetes.csv", header = T, stringsAsFactors = F)
diabets$Outcome <- as.factor(diabets$Outcome)
levels(diabets$Outcome) <- c("No", "Yes")
## 난수
set.seed(0)
diabets <- diabets[order(runif(768)), ] 

str(diabets)
summary(diabets)
sum(is.na(diabets)) # 결측값 확인
```




## 상관관계 확인
```{r}
diabets_cor<- read.csv("c:/r/diabetes.csv", header = T, stringsAsFactors = F)
corr<-round(cor(diabets_cor),1)  # cor함수는 수치형만 받음

library(ggcorrplot)
ggcorrplot(corr, hc.order = T,
           type = "lower", 
           lab = T, 
           lab_size = 4, 
           method="square",
           colors = c("red", "white", "blue"), 
           title="Correlogram of Diabetes data", 
           ggtheme=theme_bw)

```



## 상관관계로 Overcome과 상관관계가 높은 변수만 그래프로 도출
```{r}
ggplot(diabets,aes(x=Glucose,fill=Outcome))+
  geom_density(alpha=0.5)+
  scale_fill_manual(values=c("red", "blue"))+
   labs(title="Distribution of Glucose")

ggplot(diabets,aes(x=BMI,fill=Outcome))+
  geom_density(alpha=0.5)+
  scale_fill_manual(values=c("red", "blue"))+
  labs(title="Distribution of Glucose")

```

```{r}
### k 값에 따른 정확도

a <- NULL
k <- NULL
for (i in 1:8) { 
     pred<-knn(d_train, d_test, d_train_label, k= 23+i, prob = TRUE)      
     a[i] <- round(confusionMatrix(pred, d_test_label)$overall[1], 4) * 100
     k[i] <- 23+i
     print(paste("k=", k[i], ", ","accuracy =", a[i], "%",sep = ""))
}

df <- data.frame("k" = k, "accuracy" = a)


library(ggplot2)

ggplot(df, aes(x=k, y=accuracy)) + 
  geom_point(aes())+
    geom_line()+
      geom_text(aes(label=accuracy), vjust=-0.2 ,size = 5)


```


## train data, test data 분류


```{r}
nrow(diabets)*0.8
nrow(diabets)*0.2

d_train <- diabets[1:614, -9]
d_test <- diabets[615:768, -9]

d_train_label <- diabets[1:614, 9]
d_test_label <- diabets[615:768, 9]

```




3단계 : 데이터로 모델 훈련

############  KNN ##############

## k값 지정
```{r}
sqrt(nrow(diabets))    # k = 28

library(class)
pred <- knn(d_train, d_test, d_train_label, k= 28, prob = TRUE)
pred

```



4 단계 : 모델 성능 평가 
```{r}
install.packages("gmodels")
library(gmodels)

CrossTable(x = pred, y = d_test_label, prop.chisq=T)      ## 예측값과 실제값의 교차표 생성


# 정확도
install.packages("caret")
library(caret)
knn <- confusionMatrix(pred, d_test_label)


###정규화
normalize <- function(x) { 
  return ((x - min(x)) / (max(x) - min(x))) }

d_train_n <-as.data.frame(lapply(d_train, normalize))
d_test_n <- as.data.frame(lapply(d_test, normalize))

pred_n <- knn(d_train_n, d_test_n, d_train_label, k= 28, prob = TRUE)
pred_n

CrossTable(x = pred_n, y = d_test_label, prop.chisq=T)
knn_n <- confusionMatrix(pred_n, d_test_label)

###표준화
d_train_z <- scale(d_train)
d_test_z <- scale(d_test)

pred_z <- knn(d_train_z, d_test_z, d_train_label, k= 28, prob = TRUE)
pred_z

CrossTable(x = pred_z, y = d_test_label, prop.chisq=T)
knn_z <- confusionMatrix(pred_z, d_test_label)



```


############################## 시각화 ###############################
```{r}
## knn, 정규화, 표준화 정확도 비교
col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(1,3))
fourfoldplot(knn$table, color = col, conf.level = 0, margin = 1,
             main=paste("knn (",round(knn$overall[1],2)*100, "%)", sep="" ))
fourfoldplot(knn_n$table, color = col, conf.level = 0, margin =1,
             main=paste("knn_정규화 (",round(knn_n$overall[1],2)*100, "%)", sep="" ))
fourfoldplot(knn_z$table, color = col, conf.level = 0, margin = 1,
             main=paste("knn_표준화 (",round(knn_z$overall[1],2)*100, "%)", sep="" ))

```



######################### SVM ############################

# 선형SVM 훈련
```{r}
library(e1071)

diabets <- read.csv("c:/r/diabetes.csv", header = T, stringsAsFactors = F)
diabets$Outcome <- as.factor(diabets$Outcome)
levels(diabets$Outcome) <- c("No", "Yes")


```


## 난수
```{r}
set.seed(0)
diabets <- diabets[order(runif(768)), ] 
```


## train, test data set
```{r}
d_train <- diabets[1:614, ]
d_test <- diabets[615:768, ]
```


# 모델 테스트
```{r}
d_svm <- svm(Outcome~., data = d_train, kernel="linear")   
p <- predict(d_svm, d_test, type="class")
table(p, d_test[,9])
```


## 분류 결과 확인

```{r}
mean(p == d_test[, 9])
library(caret)
svm <- confusionMatrix(p, d_test[,9])
svm

```






############ 나이브 베이즈 ###########
```{r}
diabets <- read.csv("c:/r/diabetes.csv", header = T, stringsAsFactors = F)
diabets$Outcome <- as.factor(diabets$Outcome)
levels(diabets$Outcome) <- c("No", "Yes")
```



## 난수
```{r}
set.seed(0)
diabets <- diabets[order(runif(768)), ] 

```



## 트레인, 테스트 셋 분류
```{r}
d_train <- diabets[1:614, -9]
d_test <- diabets[615:768, -9]

d_train_label <- diabets[1:614, 9]
d_test_label <- diabets[615:768, 9]
```



## 모델 테스트
```{r}
a <- naiveBayes(d_train, d_train_label, laplace = 1)
pred_naive <- predict(a, d_test)
```



## 분류 결과 확인
```{r}
CrossTable(x = pred_naive, y = d_test_label, prop.chisq=T)
naive <- confusionMatrix(pred_naive, d_test_label)
naive
```




## knn, svm, 나이브베이지안정확도 비교
```{r}
col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(1,3))
fourfoldplot(knn$table, color = col, conf.level = 0, margin = 1,
             main=paste("knn (",round(knn$overall[1],2)*100, "%)", sep="" ))
fourfoldplot(svm$table, color = col, conf.level = 0, margin = 1,
             main=paste("svm (",round(svm$overall[1],2)*100, "%)", sep="" ))
fourfoldplot(naive$table, color = col, conf.level = 0, margin = 1,
             main=paste("naive (",round(naive$overall[1],2)*100, "%)", sep="" ))


```



