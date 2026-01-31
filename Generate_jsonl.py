import json
import random

base_pairs = [
    ("List all artists", "SELECT Name FROM Artist;"),
    ("Count the total number of albums", "SELECT COUNT(*) FROM Album;"),
    ("List all tracks with their album title",
     "SELECT t.Name, al.Title FROM Track t JOIN Album al ON t.AlbumId=al.AlbumId;"),
    ("Find total sales per country",
     "SELECT BillingCountry, SUM(Total) FROM Invoice GROUP BY BillingCountry;"),
    ("List customers from Canada",
     "SELECT FirstName, LastName FROM Customer WHERE Country='Canada';"),
    ("Top 5 highest priced tracks",
     "SELECT Name, UnitPrice FROM Track ORDER BY UnitPrice DESC LIMIT 5;"),
    ("List genres and number of tracks in each",
     "SELECT g.Name, COUNT(t.TrackId) FROM Genre g JOIN Track t ON g.GenreId=t.GenreId GROUP BY g.Name;"),
    ("Customers with invoices over 50 total",
     "SELECT CustomerId, Total FROM Invoice WHERE Total > 50;"),
    ("Total number of invoice lines",
     "SELECT COUNT(*) FROM InvoiceLine;"),
    ("Albums by AC/DC",
     "SELECT al.Title FROM Album al JOIN Artist a ON a.ArtistId=al.ArtistId WHERE a.Name='AC/DC';"),
    ("Sum of all invoice totals", "SELECT SUM(Total) FROM Invoice;"),
    ("Average invoice total", "SELECT AVG(Total) FROM Invoice;"),
]

countries = ["Canada","USA","Brazil","France","Germany","Norway","UK","India"]
limits = [3,5,10,20]
amounts = [10,20,30,40,50,75,100]

def varied_question_sql():
    c = random.choice(countries)
    n = random.choice(limits)
    amt = random.choice(amounts)

    variants = [
        (f"List customers from {c}",
         f"SELECT FirstName, LastName FROM Customer WHERE Country='{c}';"),

        (f"Top {n} most expensive tracks",
         f"SELECT Name, UnitPrice FROM Track ORDER BY UnitPrice DESC LIMIT {n};"),

        (f"Invoices over {amt}",
         f"SELECT InvoiceId, Total FROM Invoice WHERE Total > {amt};"),

        (f"Tracks with genre names",
         "SELECT t.Name, g.Name FROM Track t JOIN Genre g ON t.GenreId=g.GenreId;"),

        (f"Albums and their artists",
         "SELECT al.Title, ar.Name FROM Album al JOIN Artist ar ON al.ArtistId=ar.ArtistId;"),

        (f"Total tracks per album",
         "SELECT AlbumId, COUNT(*) FROM Track GROUP BY AlbumId;"),

        (f"Customers and invoice counts",
         "SELECT CustomerId, COUNT(*) FROM Invoice GROUP BY CustomerId;"),
    ]

    return random.choice(variants)


out = []

# keep originals
for q, s in base_pairs:
    out.append({"question": q, "sql": s})

# generate until 1000
while len(out) < 1000:
    if random.random() < 0.4:
        q, s = random.choice(base_pairs)
    else:
        q, s = varied_question_sql()
    out.append({"question": q, "sql": s})

with open("chinook_qa.jsonl", "w", encoding="utf-8") as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("✅ Created chinook_qa.jsonl with", len(out), "lines")
