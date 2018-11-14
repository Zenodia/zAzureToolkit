/***create the table to hold the ML model ***/
DROP TABLE IF EXISTS bc_models
CREATE TABLE bc_models (
  model_name VARCHAR(50) NOT NULL DEFAULT('default model') PRIMARY KEY,
  model VARBINARY(MAX) NOT NULL
);
GO


SELECT * from bc_models


/*** create train test split***/
DROP PROCEDURE IF EXISTS [TrainTestSplit];
GO
CREATE PROCEDURE [dbo].[TrainTestSplit] (@pct int)
AS

DROP TABLE IF EXISTS dbo.bc_train
SELECT * into bc_train FROM [BreastCancer].[dbo].[BreastCancer]
WHERE (ABS(CAST(BINARY_CHECKSUM(ID)  as int)) % 100) < @pct


DROP TABLE IF EXISTS dbo.bc_test
SELECT * into bc_test FROM [BreastCancer].[dbo].[BreastCancer]
WHERE (ABS(CAST(BINARY_CHECKSUM(ID)  as int)) % 100) > @pct

GO
EXEC TrainTestSplit 75
Go

/***check what is in the train or test table ***/
SELECT TOP (10) [ID]
      ,[Class]
      ,[age]
      ,[menopause]
      ,[tumor_size]
      ,[inv_nodes]
      ,[node_caps]
      ,[deg_malig]
      ,[breast]
      ,[breast_quad]
      ,[irradiat]
  FROM [BreastCancer].[dbo].[bc_train]


DROP PROCEDURE IF EXISTS TrainBreastCancerModel;
GO
CREATE PROCEDURE TrainBreastCancerModel (@trained_model varbinary(max) OUTPUT)
AS
BEGIN
EXEC sp_execute_external_script @language = N'Python',
@script = N'
import pickle
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
##Create SciKit-Learn logistic regression model
X = InputDataSet[[2,3,4,5,6,7,8,9,10]]
y = InputDataSet[[1]]
trained_model = pickle.dumps(GNB.fit(X,y))
'
, @input_data_1 = N'SELECT [ID]
	  ,[Class]
      ,[age]
      ,[menopause]
      ,[tumor_size]
      ,[inv_nodes]
      ,[node_caps]
      ,[deg_malig]
      ,[breast]
      ,[breast_quad]
      ,[irradiat]
  FROM [BreastCancer].[dbo].[bc_train]'
, @input_data_1_name = N'InputDataSet'
, @params = N'@trained_model varbinary(max) OUTPUT'
, @trained_model = @trained_model OUTPUT;
END;
GO



/***insert your model to the bc_models table ***/
DECLARE @model varbinary(max);
EXEC TrainBreastCancerModel @model OUTPUT;
INSERT INTO bc_models (model_name, model) values('Naive Bayes', @model);


DECLARE @model varbinary(max);
DECLARE @new_model_name varchar(50)
--SET @new_model_name = 'Naive Bayes ' + CAST(GETDATE()as varchar)
SET @new_model_name = 'Naive Bayes ' + CAST(FLOOR(RAND()*(20-10+1))+10 as varchar)
SELECT @new_model_name 
EXEC TrainBreastCancerModel @model OUTPUT;
INSERT INTO bc_models (model_name, model) values(@new_model_name, @model);

/*** check the model indeed stored in the bc_models table ***/
SELECT * from bc_models


/***make batch prediction for the test set ***/
DROP PROCEDURE IF EXISTS predict_breastcancer;
GO
CREATE PROCEDURE predict_breastcancer (@model varchar(100))
AS
BEGIN
DECLARE @nb_model varbinary(max) = (SELECT model FROM bc_models WHERE model_name = @model);
EXEC sp_execute_external_script @language = N'Python', 
@script = N'
import pickle
loaded_model = pickle.loads(nb_model)
pred = loaded_model.predict(test_data[[2,3,4,5,6,7,8,9,10]])
test_data["prediction"] = pred
#OutputDataSet = test_data.query( ''prediction != Class'' )[[0,1,11]]
OutputDataSet = test_data.query( ''prediction <=2'' )[[0,1,11]]

print(OutputDataSet)
'
, @input_data_1 = N'SELECT Top(150) [ID]
	  ,[Class]
      ,[age]
      ,[menopause]
      ,[tumor_size]
      ,[inv_nodes]
      ,[node_caps]
      ,[deg_malig]
      ,[breast]
      ,[breast_quad]
      ,[irradiat]
  FROM [BreastCancer].[dbo].[bc_test]'
, @input_data_1_name = N'test_data'
, @params = N'@nb_model varbinary(max)'
, @nb_model = @nb_model

WITH RESULT SETS (("ID" int, "Class" int, "PredictedClass" int));
END;
GO
/***execute the procedure for batch prediction ***/
EXEC predict_breastcancer 'Naive Bayes';
GO

/***create a temporary table to hold the prediction***/
DROP TABLE IF EXISTS bc_predict
CREATE TABLE bc_predict (
  ID int NOT NULL PRIMARY KEY,
  Class int NOT NULL,
  prediction int NOT NULL,
);
GO

/***insert the predicted data into a new table ***/
declare @temp table
(
    ID int,
	Class int,
	PredictedClass int
   
);

INSERT @temp  Exec predict_breastcancer 'Naive Bayes';
INSERT INTO bc_predict 
select * from @temp;


/***confirm the data exist in the new table ***/
select * from bc_predict

/***get a random number ***/ 
SELECT FLOOR(RAND()*(20-10+1))+10;