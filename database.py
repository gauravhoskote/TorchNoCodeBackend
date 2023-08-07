import psycopg2
from contextlib import closing

# Change the following variables based on your database configuration
DB_HOST = "localhost"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "admin"


def get_connection():
    """
    Function to establish a connection to the PostgreSQL database.
    Modify the connection parameters as per your database configuration.
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,  # Your database host
            database=DB_NAME,  # Your database name
            user=DB_USER,  # Your database username
            password=DB_PASSWORD,  # Your database password
        )
        return conn
    except Exception as e:
        print("Error: Unable to connect to the database.")
        print(e)
        raise


def close_connection(conn):
    """
    Function to close the database connection.
    """
    try:
        conn.close()
    except Exception as e:
        print("Error: Unable to close the database connection.")
        print(e)
        raise


def create_table():
    """
    Function to create the User_table in the database.
    """
    try:
        conn = get_connection()  # Function to get a database connection
        cursor = conn.cursor()

        create_table_query = """
            CREATE TABLE IF NOT EXISTS User_table (
                user_id SERIAL PRIMARY KEY,
                first_name VARCHAR(50) NOT NULL,
                last_name VARCHAR(50) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                occupation VARCHAR(100),
                username VARCHAR(30) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """
        cursor.execute(create_table_query)
        conn.commit()
        close_connection(conn)  # Function to close the database connection

        print("Table User_table created successfully.")
    except Exception as e:
        print("Error: Unable to create the table User_table.")
        print(e)
        raise


if __name__ == "__main__":
    create_table()