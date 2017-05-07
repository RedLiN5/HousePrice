## House Price Prediction

### Final Score: 0.13348 





The feature importance is calculated by **Pearson Correlation Coefficents** and **Mutual Information**.

![Feature Importance](plots/feature_impact.png)





This plot cannot show significant difference between important features and less important ones. 

So I accumulate feature impact ratios to see if it's more clear.

![Accumulative Feature Impact](plots/accum_feature_impact.png)

No! It's more ambiguous!



Price distribution

![Price Distribution](plots/price_distribution.png)



### Features with severe collinearity





### Feature Engineering with domain knowledge

#### Interpolation

**LotFrontage**: Interpolation by linear regression on MSSubClass, MSZoning, LotArea.



#### Encoding (Do more importance features first)

**LotShap**: Reg ——— 4; IR1 ——— 3; IR2 ——— 2; IR3 ——— 1

**Utilities**: AllPub ——— 4; NoSewr ——— 3; NoSeWa ——— 2; ELO ——— 1

**LandSlope**: Gtl —— 3; Mod —— 2; Sev —— 1

**BldgType**: 1Fam ——— 5; 2FmCon ——— 4; Duplx ——— 3; TwnhsE ——— 2; TwnhsI ——— 1

**HouseStyle**: SLvl —— 6; SFoyer —— 5; 2.5Fin —— 4; 2.5Unf —— 3.5; 2Story —— 3; 1.5Fin —— 2; 1.5Unf —— 1.5; 1Story —— 1

**ExterQual**: Ex —— 5; Gd —— 4; TA —— 3; Fa —— 2; Po —— 1

**ExterCond**: Ex —— 5; Gd —— 4; TA —— 3; Fa —— 2; Po —— 1

**Foundation**: PConc —— 4; CBlock —— 3; BrkTil —— 2; Slab, Stone, Wood —— 1

**BsmtQual**: Ex —— 5; Gd —— 4; TA —— 3; Fa —— 2; Po —— 1, NA ——0

**BsmtCond**: Ex —— 5; Gd —— 4; TA —— 3; Fa —— 2; Po —— 1, NA —— 0

**BsmtExposure**: Gd —— 4; Av —— 3; Mn —— 2; No —— 1; NA —— 0

**BsmtFinType1**: GLQ —— 6; ALQ —— 5; BLQ —— 4; Rec —— 3; LwQ —— 2; Unf —— 1; NA —— 0

**BsmtFinType2**: GLQ —— 6; ALQ —— 5; BLQ —— 4; Rec —— 3; LwQ —— 2; Unf —— 1; NA —— 0

**HeatingQC**: Ex —— 5; Gd —— 4; TA —— 3; Fa —— 2; Po —— 1

**CentralAir**: Y —— 1; N —— 0

**KitchenQual**: Ex —— 5; Gd —— 4; TA —— 3; Fa —— 2; Po —— 1

**FireplaceQu**: Ex —— 5; Gd —— 4; TA —— 3; Fa —— 2; Po —— 1, NA ——0

**GarageType**: 2Types —— 6; Attchd —— 5; Basment —— 4; BuiltIn —— 3; CarPort —— 2; Detchd —— 1; NA —— 0

**GarageFinish**: Fin —— 3; RFn —— 2; Unf —— 1; NA —— 0

**GarageQual**: Ex —— 5; Gd —— 4; TA —— 3; Fa —— 2; Po —— 1, NA —— 0

**GarageCond**: Ex —— 5; Gd —— 4; TA —— 3; Fa —— 2; Po —— 1, NA —— 0

