-- Merging the 6 datasets
CREATE TABLE sales_data AS
SELECT * FROM sales_Canada
UNION ALL
SELECT * FROM sales_China
UNION ALL
SELECT * FROM sales_India
UNION ALL
SELECT * FROM sales_Nigeria
UNION ALL
SELECT * FROM sales_UK
UNION ALL
SELECT * FROM sales_US;

-- Checking for missing values
SELECT *
FROM sales_data
WHERE
    Country IS NULL
    OR Price_per_Unit IS NULL
    OR Quantity_Purchased IS NULL
    OR Cost_Price IS NULL
    OR Discount_Applied IS NULL;
    
-- Updating “Quantity Purchased”    
SET SQL_SAFE_UPDATES = 0;    
UPDATE sales_data
SET Quantity_Purchased = 3
WHERE Transaction_ID = '00a30472-89a0-4688-9d33-67ea8ccf7aea';

-- Updating “Price Per Unit”
UPDATE sales_data
SET Price_per_Unit = (
    SELECT avg_price
    FROM (
        SELECT AVG(Price_per_Unit) AS avg_price
        FROM sales_data
        WHERE Price_per_Unit IS NOT NULL
    ) AS temp
)
WHERE Transaction_ID = '001898f7-b696-4356-91dc-8f2b73d09c63';

-- Checking for duplicate values
SELECT Transaction_ID, COUNT(*) AS duplicate_count
FROM sales_data
GROUP BY Transaction_ID
HAVING COUNT(*) > 1;

-- Adding “Total Amount” column
ALTER TABLE sales_data
ADD COLUMN Total_Amount DECIMAL(10,2);

UPDATE sales_data
SET Total_Amount = (Price_per_Unit * Quantity_Purchased) - Discount_Applied;

-- Adding “Profit” column
ALTER TABLE sales_data
ADD COLUMN Profit DECIMAL(10,2);

UPDATE sales_data
SET Profit = Total_Amount - (Cost_Price * Quantity_Purchased);

-- Sales Revenue & Profit by Country (Combined Query)
SELECT 
    Country,
    SUM(Total_Amount) AS Total_Revenue,
    SUM(Profit) AS Total_Profit
FROM sales_data
WHERE Date BETWEEN '10/2/2025' AND '14/2/2025'
GROUP BY Country
ORDER BY Total_Revenue DESC;

-- Top 5 Best-Selling Products (During the Period)
SELECT 
    Product_Name,
    SUM(Quantity_Purchased) AS Total_Units_Sold
FROM sales_data
WHERE Date BETWEEN '10/2/2025' AND '14/2/2025'
GROUP BY Product_Name
ORDER BY Total_Units_Sold DESC
LIMIT 5;

-- Best Sales Representatives (During the Period)
SELECT 
    Sales_Rep,
    SUM(Total_Amount) AS Total_Sales
FROM sales_data
WHERE Date BETWEEN '10/2/2025' AND '14/2/2025'
GROUP BY Sales_Rep
ORDER BY Total_Sales DESC
LIMIT 5;

-- store locations generated the highest sales?
SELECT 
    Store_Location,
    SUM(Total_Amount) AS Total_Sales,
    SUM(Profit) AS Total_Profit
FROM sales_data
WHERE Date BETWEEN '10/2/2025' AND '14/2/2025'
GROUP BY Store_Location
ORDER BY Total_Sales DESC
LIMIT 5;
-- key sales and profit insights for the selected period?
SELECT 
    MIN(Total_Amount) AS Min_Sales_Value,
    MAX(Total_Amount) AS Max_Sales_Value,
    AVG(Total_Amount) AS Avg_Sales_Value,
    SUM(Total_Amount) AS Total_Sales_Value,
    MIN(Profit) AS Min_Profit,
    MAX(Profit) AS Max_Profit,
    AVG(Profit) AS Avg_Profit,
    SUM(Profit) AS Total_Profit
FROM sales_data
WHERE Date BETWEEN '10/2/2025' AND '14/2/2025';