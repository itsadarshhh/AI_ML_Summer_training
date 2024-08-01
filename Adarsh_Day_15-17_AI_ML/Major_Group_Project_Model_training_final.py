import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch
from sklearn.model_selection import train_test_split

# Load the dataset from a CSV file into a pandas DataFrame
data_path = "C:\\AI Major\\Major_Group_Project\\translation_dataset.csv"
df = pd.read_csv(data_path)

# Rename columns if necessary
df.rename(columns={'source_column_name': 'en', 'target_column_name': 'hi'}, inplace=True)

# Check the DataFrame to make sure it loaded correctly
print(df.head())

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert DataFrames to Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

# Define the tokenize function
def tokenize_function(examples):
    # Tokenize inputs and targets
    inputs = tokenizer(examples['en'], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(examples['hi'], truncation=True, padding="max_length", max_length=128)
    inputs['labels'] = targets['input_ids']
    return inputs

# Apply the tokenizer to the datasets with batching
train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=8)
val_dataset = val_dataset.map(tokenize_function, batched=True, batch_size=8)

# Remove unnecessary columns if they exist
def remove_columns_if_exists(dataset, columns):
    existing_columns = set(dataset.column_names)
    columns_to_remove = [col for col in columns if col in existing_columns]
    return dataset.remove_columns(columns_to_remove)

train_dataset = remove_columns_if_exists(train_dataset, ['__index_level_0__'])
val_dataset = remove_columns_if_exists(val_dataset, ['__index_level_0__'])

# Create a DatasetDict to hold train and validation datasets
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# Verify splits
print(dataset_dict)

# Define training arguments with updated parameter name
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Use eval_strategy instead of evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the model and tokenizer
output_dir = "-mode./temp_fine-tunedl"

try:
    # Save the model and tokenizer
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully.")
except Exception as e:
    print(f"Error while saving model and tokenizer: {e}")

try:
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error while loading model and tokenizer: {e}")
    
    
# Evaluate the model
results = trainer.evaluate()
print("Results: ",results)

# Define the translation function
def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Test the fine-tuned model
text = "Hello, how are you?"
translated_text = translate_text(text, tokenizer, model)
print("Translated Text (Hindi):", translated_text)
