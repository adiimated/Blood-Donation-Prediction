# Blood-Donation-Prediction

## Project Description :
Forecasting blood supply is a serious and recurrent problem for blood collection managers. In this Project, we will work with data collected from the donor database of Blood Transfusion Service Centre. The dataset, obtained from the Machine Learning Repository, consists of a random sample of 748 donors. Our task will be to predict if a blood donor will donate within a given time window. We will look at the full model-building process: from inspecting the dataset to using the tpot library to automate your Machine Learning pipeline.

## Steps :
Blood transfusion saves lives - from replacing lost blood during major surgery or a serious injury to treating various illnesses and blood disorders. Ensuring that there's enough blood in supply whenever needed is a serious challenge for the health professionals. According to WebMD, "about 5 million Americans need a blood transfusion every year".

Our dataset is from a mobile blood donation vehicle in Taiwan. The Blood Transfusion Service Center drives to different universities and collects blood as part of a blood drive. We want to predict whether or not a donor will give blood the next time the vehicle comes to campus.

The data is stored in datasets/transfusion.data and it is structured according to RFMTC marketing model (a variation of RFM). We'll explore what that means later in this notebook. First, let's inspect the data.

### 1. Loading the blood donations data
We now know that we are working with a typical CSV file (i.e., the delimiter is ,, etc.). We proceed to loading the data into memory.

### 2. Inspecting transfusion DataFrame
Let's briefly return to our discussion of RFM model. RFM stands for Recency, Frequency and Monetary Value and it is commonly used in marketing for identifying your best customers. In our case, our customers are blood donors.

RFMTC is a variation of the RFM model. Below is a description of what each column means in our dataset:

    R (Recency - months since the last donation)
    F (Frequency - total number of donation)
    M (Monetary - total blood donated in c.c.)
    T (Time - months since the first donation)
    a binary variable representing whether he/she donated blood in March 2007 (1 stands for donating blood; 0 stands for not donating blood)

It looks like every column in our DataFrame has the numeric type, which is exactly what we want when building a machine learning model. Let's verify our hypothesis.
