query_api = client.query_api()

query = """from(bucket: "film-resell")
 |> range(start: -10m)
 |> filter(fn: (r) => r._measurement == "measurement1")"""
tables = query_api.query(query, org="code4lang")

for table in tables:
  for record in table.records:
    print(record)

