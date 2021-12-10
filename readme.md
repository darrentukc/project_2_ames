## Problem Statement

There are many variables that determine how much a home can fetch.

Using the Ames (IA) dataset (train.csv, test.csv), we want to find out which variables matter for home sale prices and produce accurate sale price predictions for interested parties, ie sellers, buyers and real estate agents. This will allow them to make informed choices that are in their interest.

## Executive Summary

We will be using the train.csv dataset to train our model. The features inside the dataset will train our model to find the relationship between certain features with the sale price. Once the model is trained. Once the model is trained, it will then be used to predict sale prices using the data in the test set. After which, the predicted results will be uploaded to kaggle to see how it performs on data that is unknown.


## Contents:

1. [Datasets Used](#1-Datasets-Used)
2. [Data Dictionary](#2.-Data-Dictionary)
3. [Cleaning Train Dataset](#3.-Cleaning-Train-Dataset)
4. [Exploratory Data Analysis and Data Visualization](#4.-Exploratory-Data-Analysis)
5. [Train and Score Models](#5.-Cleaning-Test_Dataset)
6. [Conclusions](#6.-Conclusions)
7. [References](#7-References)


### 1. Datasets Used :

The following datasets were used for this projects:
- train.csv
- test.csv


### 2. Data Dictionary:

The below is a data dictionary containing all the data features, type and its description:


|Feature|Type|Description|
|---|---|---|
|**SalePrice**|*float64*|the property's sale price in dollars.| <br>
|**MSSubClass**|*Object*|The building class.| <br>
|**MSZoning**|*Object*|Identifies the general zoning classification of the sale.| <br>
|**LotFrontage**|*float4*|Linear feet of street connected to property| <br>
|**LotArea**|*float64*|Lot size in square feet.|  <br>
|**Street**|*Object*|Type of road access to property.|  <br>
|**Alley**|*Object*|Type of alley access to property.|  <br>
|**LotShape**|*Object*|General shape of property|  <br>
|**LandContour**|*Object*|Flatness of the property|   <br>
|**Utilities**|*Object*|Type of utilities available|  <br>
|**LotConfig**|*Object*|Lot configuration|  <br>
|**LandSlope**|*Object*|Slope of property|  <br>
|**Neighborhood**|*Object*|Physical locations within Ames city limits|  <br>
|**Condition1**|*Object*|Proximity to main road or railroad| <br>
|**Condition2**|*Object*|Proximity to main road or railroad (if a second is present)| <br>
|**BldgType**|*Object*|Type of dwelling| <br>
|**HouseStyle**|*Object*|Style of dwelling| <br>
|**OverallQual**|*int64*|Overall material and finish quality| <br>
|**OverallCond**|*int64*|Overall condition rating| <br>
|**YearBuilt**|*int64*|Original construction date| <br>
|**YearRemodAdd**|*int64*|Remodel date (same as construction date if no remodeling or additions)| <br>
|**RoofStyle**|*Object*|Type of roof| <br>
|**RoofMatl**|*Object*|Roof material| <br>
|**Exterior1st**|*Object*|aExterior covering on house| <br>
|**Exterior2nd**|*Object*|Exterior covering on house (if more than one material)| <br>
|**MasVnrType**|*Object*|Masonry veneer type| <br>
|**MasVnrArea**|*int64*|Masonry veneer area in square feet| <br>
|**ExterQual**|*Object*|Exterior material quality| <br>
|**ExterCond**|*Object*|Present condition of the material on the exterior| <br>
|**Foundation**|*Object*|Type of foundation| <br>
|**BsmtQual**|*Object*|Height of the basement| <br>
|**BsmtCond**|*Object*|General condition of the basement| <br>
|**BsmtExposure**|*Object*|Walkout or garden level basement walls| <br>
|**BsmtFinType1**|*Object*|Quality of basement finished area| <br>
|**BsmtFinSF1**|*int64*|Type 1 finished square feet| <br>
|**BsmtFinType2**|*Object*|Quality of second finished area (if present)| <br>
|**BsmtFinSF2**|*int64*|Type 2 finished square feet| <br>
|**BsmtUnfSF**|*int64*|Unfinished square feet of basement area| <br>
|**TotalBsmtSF**|*int64*|Total square feet of basement area| <br>
|**Heating**|*Object*|Type of heating| <br>
|**HeatingQC**|*Object*|Heating quality and condition| <br>
|**CentralAir**|*Object*|Central air conditioning| <br>
|**Electrical**|*Object*|Electrical system| <br>
|**1stFlrSF**|*int64*|First Floor square feet| <br>
|**2ndFlrSF**|*int64*|Second floor square feet| <br>
|**LowQualFinSF**|*int64*|Low quality finished square feet (all floors)| <br>
|**GrLivArea**|*int64*|Above grade (ground) living area square feet| <br>
|**BsmtFullBath**|*int64*|Basement full bathrooms| <br>
|**BsmtHalfBath**|*int64*|Basement half bathrooms| <br>
|**FullBath**|*int64*|Full bathrooms above grade| <br>
|**HalfBath**|*int64*|Half baths above grade| <br>
|**Bedroom**|*int64*|Number of bedrooms above basement level| <br>
|**Kitchen**|*int64*|Number of kitchens| <br>
|**KitchenQual**|*Object*|Kitchen quality| <br>
|**TotRmsAbvGrd**|*int64*|Total rooms above grade (does not include bathrooms)| <br>
|**Functional**|*Object*|Home functionality rating| <br>
|**Fireplaces**|*int64*|Number of fireplaces| <br>
|**FireplaceQu**|*Object*|Fireplace quality| <br>
|**GarageType**|*Object*|Garage location| <br>
|**GarageYrBlt**|*int64*|Year garage was built| <br>
|**GarageFinish**|*Object*|Interior finish of the garage| <br>
|**GarageCars**|*int64int64*|Size of garage in car capacity| <br>
|**GarageArea**|*int64*|Size of garage in square feet| <br>
|**GarageQual**|*Object*|Garage quality| <br>
|**GarageCond**|*Object*|Garage condition| <br>
|**PavedDrive**|*Object*|Paved driveway| <br>
|**WoodDeckSF**|*int64*|Wood deck area in square feet| <br>
|**OpenPorchSF**|*int64*|Open porch area in square feet| <br>
|**EnclosedPorch**|*int64*|Enclosed porch area in square feet| <br>
|**3SsnPorch**|*int64*|Three season porch area in square feet| <br>
|**ScreenPorch**|*int64*|Screen porch area in square feet| <br>
|**PoolArea**|*int64*|Pool area in square feet| <br>
|**PoolQC**|*Object*|Quality of pool| <br>
|**Fence**|*Object*|Fence quality| <br>
|**MiscFeature**|*Object*|Miscellaneous feature not covered in other categories| <br>
|**MiscVal**|*int64*|$Value of miscellaneous feature| <br>
|**MoSold**|*int64*|Month Sold| <br>
|**YrSold**|*int64*|Year Sold| <br>
|**SaleType**|*Object*|Type of sale| <br>

### 3. Cleaning Train Dataset

Here, we clean up the dataset in order to do exploratory data analysis. We replaced all spaces in column names to underscore and all caps to lower case and converted year columns to age.

### 4. Exploratory Data Analysis and Data Visualization

In this section, we worked with the data to find trends and relations to our target variable, sale price.
- Ordinal features
-- Convert catagorical features that were ordinal in nature in ordinal features
- Pearson's Correlation
-- Drop features with low correlation to sale price
- Multicollinearity
-- Dropping features with high correlation to another feature
- Train Dataset Null Values
-- Imputing Null Values for train dataset
- Scatterplot
-- To check for outliers
- Histogram
-- To see distribution of each numerical feature
- Boxplot
-- To see if there is any relationship between catagorical feature with target variable
- Matching features between train and test set
- -Test Dataset Null Values
-- Impute Null Values for Test Dataset
- Get Dummies

### 5. Train and Score Models

Here, we trained and scored 3 models to see how well the models were able to predict the sale price.

We trained the model with train test split and scored the model with our hold out set and the test.csv, which was totally unseen by our model.

### 6. Conclusions

We conclude that using Linear Regression with Lasso will give the best result, as it will zero out features that are low in coefficients. In the scoring of our models, ( Linear Regression, Ridge, Lasso), lasso and ridge both had good scores and were close to each other. However, lasso edged out a little with slightly higher metrics and spread, which will result in models with a higher predicted accuracy, which is an important point for our model to solve the problem statement.

We also found that the features that affect the sale price the most in a positive way are:
- gr_living_area ( above grade living area square feet )
- overall_qual ( overall material and finish quality )
- functional_typ ( home functionality rating: house with typical functionality )
- functional_min1 ( home functionality rating: house with minor deductions 1)
- functional_min2 ( home functionality rating : house with deductions 2)

We are able to predict the sale price of a house given its features with an error of about $20,000. In this way, buyers, sellers and real estate agents are able to make the best decision in their favor.

### 7. References

###### kaggle

https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge

##### lot frontage definition

https://vhal.org/wp-content/uploads/sites/120/2019/04/Zoning-Code-revised-2016-w-2331A-and-2331B.pdf
