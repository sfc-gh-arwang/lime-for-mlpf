-- ADAPTED FROM 
-- https://quickstarts.snowflake.com/guide/lead_scoring_with_ml_powered_classification/index.html?index=..%2F..index#0
-- https://towardsdatascience.com/decrypting-your-machine-learning-model-using-lime-5adc035109b5

--USE YOUR OWN APPROPRIATE ROLE
use role ml_classification_analyst;

--USE YOUR OWN DB, SCHEMA, & WAREHOUSE
USE WAREHOUSE lead_scoring_demo_WH;
USE DATABASE lead_scoring_test;
USE SCHEMA lead_scoring_test.DEMO;

-- create table to hold the generated data
CREATE OR REPLACE TABLE customers AS
SELECT 'user'||seq4()||'_'||uniform(1, 3, random(1))||'@email.com' as email,
dateadd(minute, uniform(1, 525600, random(2)), ('2023-03-11'::timestamp)) as join_date,
round(18+uniform(0,10,random(3))+uniform(0,50,random(4)),-1)+5*uniform(0,1,random(5)) as age_band,
case when uniform(1,6,random(6))=1 then 'Less than $20,000'
       when uniform(1,6,random(6))=2 then '$20,000 to $34,999'
       when uniform(1,6,random(6))=3 then '$35,000 to $49,999'
       when uniform(1,6,random(6))=3 then '$50,000 to $74,999'
       when uniform(1,6,random(6))=3 then '$75,000 to $99,999'
  else 'Over $100,000' end as household_income,
    case when uniform(1,10,random(7))<4 then 'Single'
       when uniform(1,10,random(7))<8 then 'Married'
       when uniform(1,10,random(7))<10 then 'Divorced'
  else 'Widowed' end as marital_status,
  greatest(round(normal(2.6, 1.4, random(8))), 1) as household_size,
  0::float as total_order_value,
  FROM table(generator(rowcount => 100000));

  -- set total order values for longer-term customers
update customers
set total_order_value=round((case when uniform(1,3,random(9))=1 then 0
    else abs(normal(15, 5, random(10)))+
        case when marital_status='Married' then normal(5, 2, random(11)) else 0 end +
        case when household_size>2 then normal(5, 2, random(11)) else 0 end +
        case when household_income in ('$50,000 to $74,999', '$75,000 to $99,999', 'Over $100,000') then normal(5, 2, random(11)) else 0 end
    end), 2)
where join_date<'2024-02-11'::date;

-- set total order value for more recent customers
update customers
set total_order_value=round((case when uniform(1,3,random(9))<3 then 0
    else abs(normal(10, 3, random(10)))+
        case when marital_status='Married' then normal(5, 2, random(11)) else 0 end +
        case when household_size>2 then normal(5, 2, random(11)) else 0 end +
        case when household_income in ('$50,000 to $74,999', '$75,000 to $99,999', 'Over $100,000') then normal(5, 2, random(11)) else 0 end
    end), 2)
where join_date>='2024-02-11'::date;

-- verify the data loading
select * from customers;

--create training view
create or replace view customer_training
as select age_band, household_income, marital_status, household_size, case when total_order_value<10 then 'BRONZE'
    when total_order_value<=25 and total_order_value>10 then 'SILVER'
    else 'GOLD' END as segment
from customers
where join_date<'2024-02-11'::date;

select * from customer_training;

-- SNOWPARK PYTHON TO ENCODE TRAINING VIEW
CREATE OR REPLACE PROCEDURE encode(table_to_encode string, table_to_write string)
  RETURNS VARIANT
  LANGUAGE PYTHON
  RUNTIME_VERSION = 3.9
  PACKAGES = ('snowflake-snowpark-python', 'scikit-learn')
  HANDLER = 'main'
AS $$
from sklearn.preprocessing import LabelEncoder
def main(session, table_to_encode, table_to_write):
    le = LabelEncoder()
    df_training_data = session.table(table_to_encode).to_pandas()
    feat = ['AGE_BAND', 'HOUSEHOLD_INCOME_le', 'MARITAL_STATUS_le', 'HOUSEHOLD_SIZE']
    # label encoding textual data
    df_training_data['HOUSEHOLD_INCOME_le'] = le.fit_transform(df_training_data['HOUSEHOLD_INCOME'])
    df_training_data['MARITAL_STATUS_le'] = le.fit_transform(df_training_data['MARITAL_STATUS']) 
    df_training_data = df_training_data.drop(columns=["HOUSEHOLD_INCOME","MARITAL_STATUS"])
    #save dataframe
    sp_df = session.create_dataframe(df_training_data)
    sp_df.write.mode("overwrite").save_as_table(table_to_write)
    return 'ENCODING SUCCESSFUL';
$$;

CALL encode('customer_training', 'customer_training_encoded');

--CHECK RESULTS
SELECT * FROM customer_training_encoded;

-- create the classification model
CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION customer_classification_model(
    INPUT_DATA => SYSTEM$REFERENCE('table', 'customer_training_encoded'),
    TARGET_COLNAME => 'segment'
);

-- run prediction and save results
CREATE OR REPLACE TEMPORARY TABLE customer_predictions_prep AS
SELECT age_band, household_income, marital_status, household_size
from customers;

---ENCODE PREDICTION VALUES
CALL encode('customer_predictions_prep', 'customer_predictions_encoded');

-- run prediction and save results
CREATE OR REPLACE TABLE customer_predictions AS
SELECT AGE_BAND, "HOUSEHOLD_INCOME_le", "MARITAL_STATUS_le", HOUSEHOLD_SIZE,
customer_classification_model!PREDICT(INPUT_DATA => object_construct(*)) as predictions,
PARSE_JSON(PREDICTIONS)['class']::String as predicted_segment
from customer_predictions_encoded;

--Python Procedure to generate LIME explanations
CREATE OR REPLACE PROCEDURE explain_classification(predict_table string)
  RETURNS VARIANT
  LANGUAGE PYTHON
  RUNTIME_VERSION = 3.9
  PACKAGES = ('snowflake-snowpark-python', 'scikit-learn', 'lime', 'pandas', 'numpy')
  HANDLER = 'main'
AS $$
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np

#ML Powered Classification does not have a predict_proba function, so we need to create a custom function
#to generate the probabilities for each class and return them in a 2D numpy array
def prob(data):
        return_array = []
        for i in data:     
            cmd ="""CREATE OR REPLACE TABLE run_classification AS 
            SELECT customer_classification_model!PREDICT(INPUT_DATA => object_construct('AGE_BAND', ?, 'HOUSEHOLD_INCOME_le', ?, 'MARITAL_STATUS_le', ?, 'HOUSEHOLD_SIZE', ?)) as predictions"""
    
            age = float(i[0])
            hh_income = float(i[1])
            mstatus = float(i[2])
            hh_size = float(i[3])
            
            sp_session.sql(cmd, params=[age, hh_income, mstatus, hh_size]).collect()
    
            cmd ="""
                CREATE OR REPLACE TABLE predict_proba_classificationone AS
                SELECT
                    predictions:probability:BRONZE::FLOAT AS bronze_proba,
                    predictions:probability:SILVER::FLOAT AS silver_proba,
                    predictions:probability:GOLD::FLOAT AS gold_proba,
                FROM run_classification,
                LATERAL FLATTEN(input => predictions);
            """
            sp_session.sql(cmd).collect()
            
            proba_array = sp_session.table('predict_proba_classificationone').to_pandas().iloc[0].tolist()
            #print(proba_array)
            return_array.append(proba_array)
            #print(return_array)
        return np.array(return_array)

def main(session, predict_table):
    global sp_session
    sp_session = session
    df_prediction_data = session.table(predict_table).to_pandas()
    feat = ['AGE_BAND', 'HOUSEHOLD_INCOME_le', 'MARITAL_STATUS_le', 'HOUSEHOLD_SIZE']
    explainer = lime.lime_tabular.LimeTabularExplainer(df_prediction_data[feat].astype(int).values,
    mode='classification',training_labels=np.array(df_prediction_data['PREDICTED_SEGMENT']),feature_names=feat)
    # asking for explanation for LIME model, first row
    i = 1
    exp = explainer.explain_instance(df_prediction_data.loc[i,feat].astype(int).values, prob, num_features=4, num_samples=5)
    i=1
    df_prediction_data.loc[i,feat].astype(int).values
    return exp.as_list()
$$;

CALL explain_classification('customer_predictions');