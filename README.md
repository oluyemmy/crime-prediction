# Crime Prediction
## Overview
Predictive policing, which aims to forecast criminal activity using historical data and advanced analytics, has garnered attention as a potential tool for law enforcement agencies to optimize resource allocation and enhance crime prevention efforts. It relies on the premise that patterns and trends in past criminal behavior can inform predictions about future incidents.

## Aim
The aim of predictive policing using historical data and machine learning algorithms is to proactively identify areas and times with a higher likelihood of criminal activity, enabling law enforcement agencies to allocate resources more effectively and prevent crime before it occurs.

## Objectives
The specific objectives of the study are;
* Crime Prevention: Identify high-risk areas and times where criminal activity is likely to occur, enabling law enforcement to deploy resources preemptively and deter crime before it happens.
* Building five Machine learning models for crime prediction.
* Deployment of the machine learning model

## DataSet 

The dataset for the crime prediction model is sourced from [crime dataset](https://www.kaggle.com/code/umangaggarwal/predict-crime-rate)

> Dataset Attributes 
The dataset consists of 878,049 samples and 9 columns.
* Dates - timestamp of the crime incident
* Category - category of the crime incident. (This is our target variable.)
* Descript - detailed description of the crime incident
* DayOfWeek - the day of the week
* PdDistrict - the name of the Police Department District
* Resolution - The resolution of the crime incident
* Address - the approximate street address of the crime incident
* X - Longitude
* Y - Latitude

## Methodology 

Five machine models were developed and compared using classification evaluation metrics.
The models are;
* KNN
* Random Forest
* LGBM

## Findings 
Light GBM model showed the best accuracy (83%) and balanced precision-recall metrics, so it was preferred for crime tagging. The KNN models scored similarity in accuracy and performance 80% of accuracies were achieved, and the Random Forest model had high precision but it was lacking with respect to accuracy or recall.

The Model was deployed into a serveless web