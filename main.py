# main.py
import cleaning

# Clean the train and test files and save them to new files
cleaning.clean_data("train.csv")
cleaning.clean_data("test.csv")
print("Cleaning completed")

# Call the model function with the cleaned files
import trainmodel
trainmodel.train_model("cleaned_train.csv")
print("Training completed")

import testmodel
testmodel.test_model("cleaned_test.csv")
print("Testing completed")
