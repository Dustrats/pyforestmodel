from sqlalchemy import create_engine

engine = create_engine('postgresql://admin:Orion123@127.0.0.1:5433/mydb')
query = "SELECT * FROM employees_final;"
df = pd.read_sql(query, engine)