# Here is a DataProvider for the Ames Houses Prices competition from
#
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
#
# It is based on my solution which EDA is at https://www.kaggle.com/code/avibrazil/robust-feature-engineering-bagging-ensamble
#
# Avi Alkalay
# 2023-09-06
#

import pandas
import xingu

class DPAmes(xingu.DataProvider):
    id='ames_house_prices'


    # Feature names for this estimator
    x_estimator_features = """
    """

    train_dataset_sources         = dict(
        ames_train=dict(
            url="https://storage.googleapis.com/kagglesdsdata/competitions/5407/868283/train.csv",
        ),
    )



    batch_predict_dataset_sources = dict(
        ames_test=dict(
            url="https://storage.googleapis.com/kagglesdsdata/competitions/5407/868283/test.csv",
        ),
    )



    def clean_data_for_train(self, datasets: dict) -> pandas.DataFrame:
        def ddebug(input_df, message):
            self.log(message,level=logging.DEBUG)
            return input_df

        categoricalFeatures = """
             BldgType
             Condition1
             Condition2
             Electrical
             Exterior1st
             Exterior2nd
             Foundation
             Heating
             MSZoning
             MasVnrType
             MiscFeature
             Neighborhood
             RoofMatl
             RoofStyle
             SaleCondition
             SaleType
        """.split()

        # According to data_description.txt, most NaN in following features have a meaning
        categoricalOrderedFeatures = dict(
            Alley =        "_UNAVAILABLE Grvl  Pave".split(),
            BsmtCond =     "_UNAVAILABLE="" Po Fa TA Gd Ex".split(),
            BsmtExposure = "_UNAVAILABLE No  Mn  Av  Gd".split(),
            BsmtFinType1 = "_UNAVAILABLE Unf LwQ Rec BLQ ALQ GLQ".split(),
            BsmtFinType2 = "_UNAVAILABLE Unf LwQ Rec BLQ ALQ GLQ".split(),
            BsmtQual =     "_UNAVAILABLE Po Fa TA Gd Ex".split(),
            CentralAir =   "N Y".split(),
            ExterCond =    "Po Fa TA Gd Ex".split(),
            ExterQual =    "Po Fa TA Gd Ex".split(),
            Fence =        "_UNAVAILABLE MnWw GdWo MnPrv GdPrv".split(),
            FireplaceQu =  "_UNAVAILABLE Po Fa TA Gd Ex".split(),
            Functional =   "_UNKNOWN Sal Sev Maj2 Maj1 Mod Min2 Min1 Typ".split(),
            GarageCond =   "_UNAVAILABLE Po Fa TA Gd Ex".split(),
            GarageFinish = "_UNAVAILABLE Unf RFn Fin".split(),
            GarageQual =   "_UNAVAILABLE Po Fa TA Gd Ex".split(),
            GarageType =   "_UNAVAILABLE Detchd CarPort BuiltIn Basment Attchd 2Types".split(),
            HeatingQC =    "Po Fa TA Gd Ex".split(),
            HouseStyle =   "SLvl  SFoyer  2.5Unf  2.5Fin 2Story  1.5Unf  1.5Fin  1Story".split(),
            KitchenQual =  "_UNKNOWN Po Fa TA Gd Ex".split(),
            LandContour =  "Low HLS Bnk Lvl".split(),
            LandSlope =    "Sev Mod Gtl".split(),
            LotConfig =    "Inside Corner CulDSac FR2 FR3".split(),
            LotShape =     "IR3 IR2 IR1 Reg".split(),
            PavedDrive =   "N P Y".split(),
            PoolQC =       "_UNAVAILABLE Po Fa TA Gd Ex".split(),
            Street =       "Grvl Pave".split(),
            Utilities =    "_UNKNOWN NoSeWa AllPub".split(),
        )

        na_map = dict(
            _UNAVAILABLE="MiscFeature MasVnrType".split(),
            _UNKNOWN="Utilities MSZoning Electrical Exterior1st Exterior2nd KitchenQual SaleType".split(),
            Typ=['Functional'],
        )

        # Features which values might be unknown before house sale
        futureFeatures = """
            MoSold
            SaleType
            SaleCondition
            SalePrice
            YrSold
        """.split()

        unv='_UNAVAILABLE'
        unk='_UNKNOWN'

        df = (
            datasets['ames_train']
            .set_index('Id')

            # The 2 points where GrLivArea>4000 and SalePrice<300000 are outliers; remove
            .pipe(lambda table: table.drop(table.query("GrLivArea>4000 and SalePrice<300000").index))

            # Convert columns to categorical and fill NaNs with whats in the documentation
            .assign(
                # Fill NaNs with what is in the documentation

                # Convert columns to categorical
                BldgType     = lambda table: table.BldgType.astype(pandas.api.types.CategoricalDtype()),
                Condition1   = lambda table: table.Condition1.astype(pandas.api.types.CategoricalDtype()),
                Condition2   = lambda table: table.Condition2.astype(pandas.api.types.CategoricalDtype()),
                Electrical   = lambda table: table.Electrical.fillna(unk).astype(pandas.api.types.CategoricalDtype()),
                Exterior1st  = lambda table: table.Exterior1st.fillna(unk).astype(pandas.api.types.CategoricalDtype()),
                Exterior2nd  = lambda table: table.Exterior2nd.fillna(unk).astype(pandas.api.types.CategoricalDtype()),
                Foundation   = lambda table: table.Foundation.astype(pandas.api.types.CategoricalDtype()),
                Heating      = lambda table: table.Heating.astype(pandas.api.types.CategoricalDtype()),
                MSZoning     = lambda table: table.MSZoning.fillna(unk).astype(pandas.api.types.CategoricalDtype()),
                MasVnrType   = lambda table: table.MasVnrType.fillna(unv).astype(pandas.api.types.CategoricalDtype()),
                MiscFeature  = lambda table: table.MiscFeature.fillna(unv).astype(pandas.api.types.CategoricalDtype()),
                Neighborhood = lambda table: table.Neighborhood.astype(pandas.api.types.CategoricalDtype()),
                RoofMatl     = lambda table: table.RoofMatl.astype(pandas.api.types.CategoricalDtype()),
                RoofStyle    = lambda table: table.RoofStyle.astype(pandas.api.types.CategoricalDtype()),
                SaleCondition= lambda table: table.SaleCondition.astype(pandas.api.types.CategoricalDtype()),
                SaleType     = lambda table: table.SaleType.fillna(unk).astype(pandas.api.types.CategoricalDtype()),

                # Convert columns to ordered categorical
                Alley        = lambda table: table.Alley.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['Alley'],ordered=True)),
                BsmtCond     = lambda table: table.BsmtCond.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['BsmtCond'],ordered=True)),
                BsmtExposure = lambda table: table.BsmtExposure.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['BsmtExposure'],ordered=True)),
                BsmtFinType1 = lambda table: table.BsmtFinType1.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['BsmtFinType1'],ordered=True)),
                BsmtFinType2 = lambda table: table.BsmtFinType2.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['BsmtFinType2'],ordered=True)),
                BsmtQual     = lambda table: table.BsmtQual.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['BsmtQual'],ordered=True)),
                CentralAir   = lambda table: table.CentralAir.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['CentralAir'],ordered=True)),
                ExterCond    = lambda table: table.ExterCond.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['ExterCond'],ordered=True)),
                ExterQual    = lambda table: table.ExterQual.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['ExterQual'],ordered=True)),
                Fence        = lambda table: table.Fence.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['Fence'],ordered=True)),
                FireplaceQu  = lambda table: table.FireplaceQu.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['FireplaceQu'],ordered=True)),
                Functional   = lambda table: table.Functional.fillna('Typ').astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['Functional'],ordered=True)),
                GarageCond   = lambda table: table.GarageCond.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['GarageCond'],ordered=True)),
                GarageFinish = lambda table: table.GarageFinish.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['GarageFinish'],ordered=True)),
                GarageQual   = lambda table: table.GarageQual.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['GarageQual'],ordered=True)),
                GarageType   = lambda table: table.GarageType.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['GarageType'],ordered=True)),
                HeatingQC    = lambda table: table.HeatingQC.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['HeatingQC'],ordered=True)),
                HouseStyle   = lambda table: table.HouseStyle.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['HouseStyle'],ordered=True)),
                KitchenQual  = lambda table: table.KitchenQual.fillna(unk).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['KitchenQual'],ordered=True)),
                LandContour  = lambda table: table.LandContour.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['LandContour'],ordered=True)),
                LandSlope    = lambda table: table.LandSlope.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['LandSlope'],ordered=True)),
                LotConfig    = lambda table: table.LotConfig.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['LotConfig'],ordered=True)),
                LotShape     = lambda table: table.LotShape.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['LotShape'],ordered=True)),
                PavedDrive   = lambda table: table.PavedDrive.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['PavedDrive'],ordered=True)),
                PoolQC       = lambda table: table.PoolQC.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['PoolQC'],ordered=True)),
                Street       = lambda table: table.Street.fillna(unv).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['Street'],ordered=True)),
                Utilities    = lambda table: table.Utilities.fillna(unk).astype(pandas.api.types.CategoricalDtype(categories=categoricalOrderedFeatures['Utilities'],ordered=True)),
            )

            .assign(
                # Replace unknwon values with feature mode
                KitchenQual=lambda table: table.KitchenQual.str.replace(unk, table.KitchenQual.mode().head(1).values[0]),
                Utilities=lambda table: table.Utilities.str.replace(unk, table.KitchenQual.mode().head(1).values[0]),
                MSZoning=lambda table: table.MSZoning.str.replace(unk, table.KitchenQual.mode().head(1).values[0]),
                Electrical=lambda table: table.Electrical.str.replace(unk, table.KitchenQual.mode().head(1).values[0]),
                Exterior1st=lambda table: table.Exterior1st.str.replace(unk, table.KitchenQual.mode().head(1).values[0]),
                Exterior2nd=lambda table: table.Exterior2nd.str.replace(unk, table.KitchenQual.mode().head(1).values[0]),
                SaleType=lambda table: table.SaleType.str.replace(unk, table.KitchenQual.mode().head(1).values[0]),

                # Replace unknown values with Zero
                MasVnrArea=lambda table: table.MasVnrArea.fillna(0),
                GarageCars=lambda table: table.GarageCars.fillna(0),
                GarageArea=lambda table: table.GarageArea.fillna(0),
                BsmtFinSF1=lambda table: table.BsmtFinSF1.fillna(0),
                BsmtFinSF2=lambda table: table.BsmtFinSF2.fillna(0),
                BsmtHalfBath=lambda table: table.BsmtHalfBath.fillna(0),
                BsmtFullBath=lambda table: table.BsmtFullBath.fillna(0),
                BsmtUnfSF=lambda table: table.BsmtUnfSF.fillna(0),
                TotalBsmtSF=lambda table: table.TotalBsmtSF.fillna(0),

                # Fill NaNs with values from other feature
                GarageYrBlt=lambda table: table.GarageYrBlt.combine_first(table.YearBuilt),

                # Estimate LotFrontage as median of LotFrontage from same Neighborhood
                LotFrontage=lambda table: table.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median())),

                # New column
                TotalSF=lambda table: table.TotalBsmtSF + table.n1stFlrSF + table.n2ndFlrSF,

                # Features to log
                logTotalSF=lambda table: numpy.log1p(table.TotalSF),
                logLotArea=lambda table: numpy.log1p(table.LotArea),
                logn1stFlrSF=lambda table: numpy.log1p(table.n1stFlrSF),
                logGrLivArea=lambda table: numpy.log1p(table.GrLivArea),
            )
        )