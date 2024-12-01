# from cleaning import clean_data
from trainmodel import train_model
from visualization import visualize_and_causal_analysis

def main():
    # # Step 1: Clean the data
    # print("\n=== Step 1: Cleaning Data ===")
    # clean_data("./data/raw/train.csv")  # Clean raw training data and save as 'cleaned_train.csv'

    # Step 2: Perform Causal Analysis
    print("\n=== Step 2: Causal Analysis ===")
    visualize_and_causal_analysis()  # Analyze causal relationships in cleaned data

    # Step 3: Train the model
    print("\n=== Step 3: Training Model ===")
    train_model("cleaned_train.csv")  # Train the model using cleaned data


if __name__ == "__main__":
    main()
