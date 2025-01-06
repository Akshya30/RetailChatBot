import matplotlib.pyplot as plt
from datasets import Dataset

# Use double backslashes
dataset_path = "C:\\Retail Bot\\bitext_retail_commerce\\train"


# Load the Arrow file
dataset = Dataset.from_file(f"{dataset_path}\\data-00000-of-00001.arrow")

# Explore the dataset
print("Columns:", dataset.column_names)
print("First 2 rows:", dataset[:2])

df = dataset.to_pandas()

print(df.head())
print(df.info())

print(df.isnull().sum())

print("Duplicate Rows:")
print(df.duplicated().sum())

missing_responses = df[df['response'].isnull()]

print("missing_responses: ", missing_responses)

# Count unique queries
unique_queries = df['instruction'].nunique()
print(f"Number of unique queries: {unique_queries}")

# Most common queries
print("Top 10 most common queries:")
print(df['instruction'].value_counts().head(10))

# Count unique intent
unique_intents = df['intent'].nunique()
print(f"Number of unique intents: {unique_intents}")

# Most common intent
print("Top 10 most common intents:")
print(df['intent'].value_counts().head(10))

# Count unique categories
unique_categories = df['category'].nunique()
print(f"Number of unique categories: {unique_categories}")

# Most common categories
print("Top 10 most common categories:")
print(df['category'].value_counts().head(10))


# Count unique tags
unique_tags = df['tags'].nunique()
print(f"Number of unique tags: {unique_tags}")

# Most common tags
print("Top 10 most common tags:")
print(df['tags'].value_counts().head(10))


# Count unique responses
unique_responses = df['response'].nunique()
print(f"Number of unique responses: {unique_responses}")

# Most common responses
print("Top 10 most common responses:")
print(df['response'].value_counts().head(10))


sample= df.sample(15)

print("Sample Query-Response Pairs:")

for i, row in sample.iterrows():
    print(f"Query: {row['instruction']}")
    print(f"Response: {row['response']}")
    print("---")


shared_responses = df[df['response'].duplicated(keep=False)]
print("Queries with identical responses:")
print(shared_responses[['instruction', 'response']])

print(shared_responses['response'].head(1))

print(df['intent'].value_counts())
print(df['category'].value_counts())


# Visualize intent distribution
df['intent'].value_counts().plot(kind='bar', title='Intent Distribution')
plt.show()

# Visualize category distribution
df['category'].value_counts().plot(kind='bar', title='category Distribution')
plt.show()

print(df['tags'].value_counts().head(10))  # Most common tags
print(df['tags'].value_counts().tail(10))  # Least common tags


# Determine if certain tags are strongly associated with specific categories.
tag_category_relation = df.groupby('tags')['category'].value_counts()
print(tag_category_relation.head(30))

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['tags_encoded'] = label_encoder.fit_transform(dataset['tags'])
