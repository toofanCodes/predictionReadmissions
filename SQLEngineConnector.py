from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path
import os

def connectToDB():
    # Define the path to the .env file
    env_path = Path('.') / '.env'
    # print(f"Looking for .env at: {env_path.resolve()}")

    # Check if the .env file exists
    if env_path.exists():
        # print('.env file exists')
        load_dotenv(env_path)

        # Check for required environment variables
        if not all([os.getenv('DB_USER'), os.getenv('DB_PASS'), os.getenv('DB_HOST'), os.getenv('DB_NAME')]):
            raise ValueError("Missing required environment variables: DB_USER, DB_PASS, DB_HOST, or DB_NAME")

        try:
            # Create the database URL
            db_url = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:3306/{os.getenv('DB_NAME')}"
            # Create and return the SQLAlchemy engine
            engine = create_engine(db_url)
            # print(f"SQLAlchemy Engine created with {os.getenv('DB_NAME')}!")
            return engine
        except Exception as e:
            print(f"Engine creation failed: {e}")
            raise
    else:
        print(f".env file NOT found at {env_path.resolve()}")
        raise FileNotFoundError(".env file does not exist")

