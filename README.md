# Predictive Analythics : Loan Status Analysis

### **Background Project**

---
![loan-vector-icon](https://github.com/user-attachments/assets/c51029d1-c7a5-4a1f-a7b1-dca6a4c41221)

A loan is a type of debt that can involve any kind of tangible asset, although it is typically associated with monetary or financial loans. Like other debt instruments, a loan requires the redistribution of financial assets over time between the borrower (debtor) and the lender (creditor).

The borrower initially receives a sum of money from the lender, which is to be repaid over a period of time according to the original agreement, often in regular installments, to the lender. This service is usually provided at a cost, known as interest on the debt. The borrower may also be subject to certain conditions set forth in the form of loan terms.

## **Business Understanding**
### **Problem Statements**
Based on the background that gave rise to this research, the details of the problems that can be solved in this project are as follows:

*   What is the best machine learning model for predicting loan status?
*   Can gender influence the success of a loan?
*   
### **Goals**
The aims of carrying out this experiment are:
*   Pengolahan data dari beberapa variabel yang telah ditentukan dalam memprediksi penyakit stroke.
*   Sistem ini dapat digunakan oleh masyarakat untuk memprediksi penyakit stroke berdasarkan gejala-gejala yang ada.
*   Membandingkan beberapa algoritma yang digunakan dalam memprediksi penyakit stroke guna mendapatkan performa yang terbaik.

### **Solution Statements**
In order to achieve the existing research objectives, the author will build a prediction model with 3 different algorithm models. All models will be compared and the best one will be selected with the best performance and accuracy used:

*   **Logistic Regression**
Logistic Regression is used to model the probability of a binary outcome based on one or more predictor variables. It estimates the probability that a given input point belongs to a particular class.

*   **XGBoost Classifier**
XGBoost is a gradient boosting framework designed to improve the performance of machine learning models. It builds on the concept of boosting, which combines multiple weak learners (typically decision trees) to create a strong predictive model. 

*   **Support Vector Machines**
Support Vector Machine (SVM) is a powerful algorithm that aims to find the optimal hyperplane which best separates different classes in the feature space. The core concept behind SVM involves finding the hyperplane that maximizes the margin between different classes.

## **Data Understanding**
The dataset that the author uses in this project is the Dataset with the title Loan Approvals which is taken from the Kaggle page (https://www.kaggle.com/datasets/prateekmaj21/loan-approvals). The dataset contains 614 data with 12 columns. The following is more detailed information from each dataset column:

*   `Gender` : Gender of Person
*   `Married` : Marital Status
*   `Dependents` : Number of dependents
*   `Education` : Education Level
*   `Self_Employed` : If they are self employed.
*   `Applicant_Income` : Income of applicant
*   `Coapplicant_Income` : Income of co applicant.
*   `Loan_Amount` : Loan Amount
*   `Term` : Term of loan (total of week)
*   `Credit_History` : If they have good/bad credit history.
*   `Area` : Geographical Area
*   `Status` : Loan approval status

![image](https://user-images.githubusercontent.com/55022521/189842969-a4ab8c9e-35a1-4b66-97cf-7ffc5827943d.png)
 
 **Correlation Matrix**
 
![corelational matrix](https://user-images.githubusercontent.com/55022521/189872685-aa01ed7b-78b6-4f1c-b8b5-e77be697dca7.png)

## **Data Cleansing**
At this stage, the author performs data cleaning. Data cleaning is done by filling in the missing values ​​in the dataset with the `fillna` method which is filled with the `mode` or `median` value.

## **Data Preparation**
The technique that the author uses in the Data Preparation stage is as follows:

*  **Encoding Categorical Feature**

At this stage, the author carries out the encoding process for the category features using the `LabelEncoder` technique in the Scikitlearn library.

*   **Standardization**

At this stage, the author performs standardization using the `StandarScaler` found in the sckitlearn library. This standardization is very useful in leveling the scale of data, especially numerical data.

*   **Train-Test-Split**
At this stage, the author divides the dataset into training data and test data using `train_test_split` from the Scikitlearn library. This dataset division aims to be used later to train and evaluate model performance. In this project, 90% of the dataset is used to train the model, and the remaining 10% is used to evaluate model performance.
.

## **Modeling**

At this stage, the author builds prediction models using three different algorithms: Logistic Regression, XGBoost Classifier, and Support Vector Machine. Since this study involves classification, the author uses accuracy metrics to assess the performance of the models created with these three algorithms.

For each of the algorithms, the author will explain the performance of the trained models and the testing results as follows:

In the training phase, the accuracy of each model is as follows:

1. The model using the Logistic Regression algorithm achieved an accuracy score of 0.9998, or approximately 99%.
2. The model using the XGBoost Classifier algorithm achieved an accuracy score of 0.7959, or approximately 80%.
3. The model using the Support Vector Machine algorithm achieved an accuracy score of 0.9146, or approximately 91%.

In the testing phase, the accuracy of each model is as follows:

1. The model using the Logistic Regression algorithm achieved an accuracy score of 0.9998, or approximately 99%.
2. The model using the XGBoost Classifier algorithm achieved an accuracy score of 0.7959, or approximately 80%.
3. The model using the Support Vector Machine algorithm achieved an accuracy score of 0.9146, or approximately 91%.

The highest accuracy in the training phase was achieved by the XGBoost Classifier algorithm, followed by Logistic Regression and Support Vector Machine. For model testing accuracy, the highest result was achieved by the Logistic Regression algorithm, followed by Support Vector Machine, with the lowest accuracy being from the XGBoost Classifier algorithm. Based on the results from training and testing the models, the author concludes that the best and most stable performance was achieved using the K-Neighbors Classifier algorithm.

## **Evaluation**
Because the model used is a classification model, the model that has been built will be evaluated using the confusion matrix method. Confusion matrix is ​​a tabular summary of the number of correct and incorrect predictions made by the classification model. Confusion matrix is ​​used to measure the performance of the classification model. 

![cfx](https://user-images.githubusercontent.com/55022521/189931611-5867b84b-98b2-4ab9-b053-661a0c053425.png)

The following is an explanation of each value contained in the confusion matrix:

*   **Prediction Value**: keluaran dari program dimana nilainya Positif dan Negatif.
*   **Actual Value**: nilai sebenarnya dimana nilainya True dan False.
*   **True Positive** (TP): Nilai aktual Positif dan diprediksi juga Positif.
*   **True Negative** (TN): Nilai actual Negatif dan prediksi juga Negatif.
*  **False Positive** (FP): Nilai actual negatif tetapi prediksi positif. Istilah lain nya dikenal sebagai 'Type 1 error' atau kesalahan Tipe 1
*   **False Negative** (FN) :Nilai actual Positif tetapi prediksinya Negatif. Istilah lain nya sebagai 'Type 2 error' atau kesalahan Tipe 2

Confusion matrix can also be used to evaluate the performance of a classification model by calculating performance metrics such as `accuracy`, `precision`, `recall or sensitivity`, and `F-1 Score`.

**Accuracy**: 
Describes how accurate the model is in correctly classifying

```
Accuracy = (TP+TN) / (TP+FP+FN+TN)
```

**Precision**: 

Describes the accuracy between the requested data and the predicted results provided by the model.
```
Precision = (TP) / (TP + FP)
```
**Recall atau sensitivity**: 

Describes the success of the model in rediscovering information.

```
Recall  = TP / (TP + FN)
```

**F-1 Score**: describes the comparison of the average precision and weighted recall. Accuracy is exactly what we use as a reference for algorithm performance if our dataset has a very close (symmetric) number of False Negative and False Positive data. However, if the number is not close, then we should use F1 Score as a reference.

```
F-1 Score  = (2 * Recall * Precision) / (Recall + Precision)
```

Let's calculate it manually:

- Accuracy = (TP+TN) / (TP+FP+FN+TN)

Accuracy = (261 + 757) / (261 + 218 + 709 + 757) = 0.523
 
- Precision = (TP) / (TP + FP)

Precision = 261 / (261 + 218) = 0.544

- Recall  = TP / (TP + FN)

Recall = 261 /(261 + 709 ) = 0.269

- F-1 Score  = (2 * Recall * Precision) / (Recall + Precision)

F-1 Score  = ( 2 * 0.269 * 0.544 ) / ( 0.269 * 0.544) = 0.360


## **Conclusion**

Based on the research results above, it was found that when training the model, the author obtained very good model accuracy from the 4 models. However, when the model was tested, the accuracy results were quite good.

However, there are other things that make this model training less good. There are `missing values` in several `feature` columns, `imbalance dataset`, and the minimal number of datasets used in model training

