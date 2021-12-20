<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# Project 2 - Ames Housing Data and Kaggle Challenge

## Background

In this project, we will use the well known Ames housing data (from 2006-10) to create a regression model that can predict the price of houses in Ames, Iowa.

## Problem Statement

##### What we plan to do

There are are many variables that determine how much a home can fetch.
Using the Ames (IA) housing data ( from 2006 to 2010 ), we want to find out which variables matter for home sale prices and produce accurate sale price predictions.

##### What models will we be exploring

We will be focusing our project on supervised machine learning models, in particular, linear models and regularization.

##### How will success be evaluated?

The success of our models will be determined by the highest R2 and lowest RMSE scores. This will represent the accuracy and the margin or error of our models. Using these scores, we aim to create a model that will outperform the baseline model that we have identified.

##### Is the scope of the project appropriate? Who are our important stakeholders and why is this important to investigate?

This model will provide the Outside View, helping to reduce information asymmetry between potential home buyers, sellers and real estate agents.<br><br>
An Inside View involves making a prediction based on knowledge that the predictor already has, (eg insider news for stocks and shares), while an Outside View prediction ignores these details and makes an estimation based on a class of rougly similar previous cases. This is to cut out any bias that may exist.

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
1. Ordinal features
    - Convert catagorical features that were ordinal in nature in ordinal features
2. Pearson's Correlation
    - Drop features with low correlation to sale price
3. Multicollinearity
    - Dropping features with high correlation to another feature
4. Train Dataset Null Values
    - Imputing Null Values for train dataset
5. Scatterplot
    - To check for outliers
6. Histogram
    - To see distribution of each numerical feature
7. Boxplot
    - To see if there is any relationship between catagorical feature with target variable
8. Matching features between train and test set
9. Test Dataset Null Values
    - Impute Null Values for Test Dataset
10. Get Dummies

### 5. Train and Score Models

Here, we trained and scored 3 models to see how well the models were able to predict the sale price.

We trained the model with train test split and scored the model with our hold out set and the test.csv, which was totally unseen by our model.

We found that using Linear Regression with Lasso will give the best result, as it will zero out features that are low in coefficients. In the scoring of our models, ( Linear Regression, Ridge, Lasso), lasso and ridge both had good scores and were close to each other. However, lasso edged out a little with slightly higher metrics and spread, which will result in models with a higher predicted accuracy, which is an important point for our model to solve the problem statement.


### 6. Conclusions

Based on previously similar cases (2006 to 2010 Ames housing data), we are able to predict the sale price of a house given its features with an error of about $20,000, allowing buyers, sellers and realtors to get an outside view of the market rate of a house without bias.<br>
Recalling our problem statement: 'There are many variables that determine how much a home can fetch, and there is traditionally a lot of information asymmetry between buyers, sellers and realtors.
<br>
Our lasso regression model zeros out features that do not impact the saleprice.

We also found that the features that affect the sale price the most in a positive way are:

- gr_living_area ( above grade living area square feet )
  - The larger the above ground living area, the higher the saleprice
- overall_qual ( overall material and finish quality )
  - The better the material and finish of a house, the higher the saleprice
- functional_typ ( home functionality rating: house with typical functionality )
  - Functions of rooms are working as expected (eg, toilet with working sink and show, rooms with windows that are in working condition etc)
- functional_min1 ( home functionality rating: house with minor deductions 1)
  - Functions of rooms are working as expected, except for some very minor issues
- functional_min2 ( home functionality rating : house with deductions 2)
  - Functions of rooms are working as expected, except for some issues

 Features that affect the sale price negatively are:<br>

- reno_age ( number of years since last renovation )
  - Houses with a low reno age usually hints that it is in newer and better condition
- house_age ( age of the house )
  - The older the house, the lower the saleprice
- neighborhood_MeadowV ( Physical locations within Ames city limits: Meadow Village )
  - The closer the house is to Meadow Village, the lower the saleprice
- ms_zoning_c (  Identifies the general zoning classification of the sale: Commercial )
  - Sale of house for commercial purposes will bring a lower saleprice
- paved_drive_N ( Paved driveway: Dirt or gravel )
  - Houses without paved driveway will fetch a lower saleprice compared to a house with a paved driveway.

##### Limitations of Model

Every model has its limitations:
- Geographical constraints: The dataset is restricted to Ames, a city in Iowa with a relatively small population
- The dataset contains data from 2006 to 2010. Trends and demands may have changed since then.
- The model is able to predict well for homes within the range of 100k to 300k, as that is where the majority of the data is.
- Other features suchs as amenities / educational institutions could better train the model.
- Level of crime rates for different neighborhoods may also affect saleprice.<br><br>

##### Future Improvements

In future, this model can be further improved by introducing some potentially impactful features mentioned above, and adding data from neighboring cities. Doing so will expand the coverage of the model and allows the identification of trends between cities.

### 7. References

###### kaggle

https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge

##### lot frontage definition

https://vhal.org/wp-content/uploads/sites/120/2019/04/Zoning-Code-revised-2016-w-2331A-and-2331B.pdf
