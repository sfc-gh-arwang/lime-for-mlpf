--Code Sourced here: https://quickstarts.snowflake.com/guide/lead_scoring_with_ml_powered_classification/index.html?index=..%2F..index#0

-- create table to hold the generated data
create table daily_impressions(day timestamp, impression_count integer);-- create the example customer table
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