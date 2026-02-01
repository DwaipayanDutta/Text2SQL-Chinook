<div align="center">
  <img width="70" height="70" alt="Text2SQL" src="https://github.com/user-attachments/assets/d2387972-61ef-4b72-8cbf-e4073d8111cb" />
  <h1>Text2SQL-Chinook </h1>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=yellow" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/🤗-Transformers-FF6200?style=for-the-badge&logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTM1IiBoZWlnaHQ9IjMyIiB2aWV3Qm94PSIwIDAgMTM1IDMyIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xMzUgMEgxMjEuMDJDMTIwLjYyIDAgMTIwIDEuMTg0IDEyMCAyLjIzNTdWMi4yMzU3QzEyMCAzLjI4NzMgMTIwLjYyIDQuNDcxNCAxMjEuMDIgNS4wMDAySDEzNVYwaFYwWk0xMzUgMzJIMTIxLjAyQzEyMC42MiAzMiAxMjAgMzAuODE2IDEyMCAyOS43NjQzVjI5Ljc2NDNDMTIwIDI4LjcxMTcgMTIwLjYyIDI3LjQ5NzIgMTIxLjAyIDI2Ljk5OThIMTM1VjMyWk0xMjEuMDIgNUgxMzVWNzUuNUgxMjEuMDJDMTE5LjUgNzUuNSAxMTguOTUgNzcuMDUwMiAxMTguOTUgNzguNzY0M1Y4NC4yMzU3QzExOC45NSA4NS45MTg4IDExOS41IDg3LjQ2OSAxMjEuMDIgODcuNUgxMzVWMzJIMTIxLjAyQzEyMy41IDMyIDEyNS4wNSA0MC41IDEyNS4wNSA0My4yMzU3VjQ4Ljc2NDNDMTI1LjA1IDUxLjUwMTMgMTIzLjUgNjAgMTIxLjAyIDYwSDEwMFYxMDBIMzVWMzJIMTIxLjAyWiIgZmlsbD0iIzY2RkY2NiIvPjwvc3ZnPg==" alt="License">
</div>

# <div align="center"> • **Chinook DB** • **Streamlit Demo**</div>

<div align="center">
  <img alt="Demo" src="./streamlit/demo.gif" width="800"/>
</div>

**Transform natural language questions into executable SQL** - Production-ready Text2SQL pipeline trained on Chinook database.

## **Setup**

```bash
# 1. Clone & Install
git clone https://github.com/DwaipayanDutta/Text2SQL-Chinook.git
cd text2sql-chinook
pip install -r requirements.txt

# 2. Get Chinook DB
curl -L -o data/chinook.db "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip" && \
unzip -o data/chinook.zip -d data/ && mv data/Chinook_Sqlite.sqlite data/chinook.db

# 3. Launch Demo ✨
streamlit run streamlit/app.py


---
## Instruction
In order to help the team at Chinook music store, you need to answer the following 5 queries
- Q1 Use the Invoice table to determine the countries that have the most invoices. Provide a table of BillingCountry and Invoices ordered by the number of invoices for each country. The country with the most invoices should appear first.
- Q2 We would like to throw a promotional Music Festival in the city we made the most money. Write a query that returns the 1 city that has the highest sum of invoice totals. Return both the city name and the sum of all invoice totals.

**check your solution:** The top city for Invoice dollars was Prague with an amount of 90.24
- Q3 The customer who has spent the most money will be declared the best customer. Build a query that returns the person who has spent the most money. I found the solution by linking the following three: Invoice, InvoiceLine, and Customer tables to retrieve this information, but you can probably do it with fewer!

**check your solution:** The customer who spent the most according to invoices was Customer 6 with 49.62 in purchases.
- Q4 The team at Chinook would like to identify all the customers who listen to Rock music. Write a query to return the email, first name, last name, and Genre of all Rock Music listeners. Return your list ordered alphabetically by email address starting with 'A'.

**Check your solution:** Your final table should have 59 rows and 4 columns.
- Q5 Write a query that determines the customer that has spent the most on music for each country. Write a query that returns the country along with the top customer and how much they spent. For countries where the top amount spent is shared, provide all customers who spent this amount.
You should only need to use the Customer and Invoice tables.

**Check Your Solution:** Though there are only 24 countries, your query should return 25 rows because the United Kingdom has 2 customers that share the maximum.

---
## Queries
- Q1 :
```SQL
SELECT BillingCountry, COUNT(*) AS invoices_number
FROM Invoice
GROUP BY 1
ORDER BY 2 DESC
```
- Q2:
```SQL
SELECT BillingCity, SUM(Total) AS invoices_total
FROM Invoice
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1
```
- Q3:
```SQL
SELECT c.CustomerId,
       c.FirstName,
       c.LastName,
       sum(inv.UnitPrice) AS invoices
FROM Invoice i
JOIN InvoiceLine il
ON il.Invoiceid = i.Invoiceid
JOIN Customer c
ON c.CustomerId = i.CustomerId
GROUP BY 1,2,3
ORDER BY i.Total DESC
LIMIT 1
```
- Q4:
```SQL
SELECT c.Email,
       c.FirstName,
       c.LastName,
       g.Name
FROM Customer c
JOIN Invoice i
ON c.CustomerId= i.CustomerId
JOIN InvoiceLine il
ON i.InvoiceId= il.InvoiceId
JOIN Track t
ON t.TrackId = il.TrackId
JOIN Genre g
ON g.GenreId = t.GenreId
WHERE g.Name = 'Rock'
GROUP BY 1
```
- Q5:
```SQL
WITH c AS (SELECT Invoice.CustomerId AS id_cst, 
                 Invoice.BillingCountry AS Country, 
                 SUM(Invoice.Total) AS som 
           FROM Invoice
           JOIN Customer 
           ON Invoice.BillingCountry = Customer.Country AND Invoice.CustomerId = Customer.CustomerId
           GROUP BY 1,2
           ORDER BY 2 ),
          
    Customers AS (SELECT Customer.CustomerId as cust_id, 
                         Customer.FirstName as name_customer, 
                         Customer.LastName as lastname_customer 
              FROM Customer)

SELECT customers.cust_id, 
       customers.name_customer,
       customers.lastname_customer, 
       b.country, 
       b.max_som 
FROM Customers, (SELECT a.country AS country, 
                        max(a.som) AS max_som 
                 FROM (SELECT Invoice.CustomerId AS id_cst, 
                              Invoice.BillingCountry AS Country, 
                              SUM(Invoice.Total) AS som 
                       FROM Invoice 
                       JOIN Customer 
                       ON Invoice.BillingCountry = Customer.Country AND Invoice.CustomerId = Customer.CustomerId
                       GROUP BY 1,2
                       ORDER BY 2 ) AS a
                 GROUP BY 1
                 ORDER BY 2 ) AS b
JOIN c
ON c.country = b.country AND c.som = b.max_som
WHERE Customers.cust_id = c.id_cst
```
