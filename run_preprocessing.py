from src.data_loader import load_data
from src.preprocess import text_cleaner, preprocessing

def main():
    df = load_data()
    cleaned_df = preprocessing(df)
    cleaned_df_text = text_cleaner(cleaned_df).cleaner()
    cleaned_df_text.to_csv("processed_data.csv", index=False)
    print("Data preprocessing completed and saved successfully")

if __name__ == "__main__":
    main()
