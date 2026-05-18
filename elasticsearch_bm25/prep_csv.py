import pandas as pd

df = pd.read_csv("marvel.csv")

df["id"] = range(1, len(df) + 1)

df = df.drop(
    columns=[
        "Imprint",
        "Format",
        "Rating",
        "Price",
        "publish_date",
        "active_years",
    ]
)

df = df.fillna("")

df = df[
    (df["comic_name"] != "")
    | (df["issue_title"] != "")
    | (df["issue_description"] != "")
]


columns = [
    "comic_name",
    "issue_title",
    "issue_description",
    "writer",
    "penciler",
    "cover_artist",
]

for col in columns:
    df[col] = df[col].str.lower()

df.to_json("/datasets/dataset.json", index=True, orient="records", indent=2)
