try:
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
except ImportError as e:
    print("❌ Import Error:", e)
    print("👉 Please install required libraries:")
    print("pip install pandas scikit-learn joblib")
    exit(1)

csv_path = r'C:\Users\Akshay\Desktop\anugrahaa\test\data\test (1).csv'

try:
    df = pd.read_csv(csv_path, encoding="latin1")
    print("✅ CSV file loaded successfully")
except FileNotFoundError:
    print(f"❌ File not found: {csv_path}")
    exit(1)
except UnicodeDecodeError:
    print("❌ Encoding error while reading CSV")
    print("👉 Try encoding='utf-8' or 'ISO-8859-1'")
    exit(1)
except Exception as e:
    print("❌ Unexpected error while loading CSV:", e)
    exit(1)

required_columns = {"text", "sentiment"}
if not required_columns.issubset(df.columns):
    print("❌ Missing required columns in CSV")
    print("Found columns:", list(df.columns))
    print("Required columns:", required_columns)
    exit(1)

df = df.dropna(subset=["text", "sentiment"])
df["text"] = df["text"].astype(str).str.strip()
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

df = df[(df["text"] != "") & (df["sentiment"] != "")]
df = df.reset_index(drop=True)

X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved successfully!")
