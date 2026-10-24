"""Build the benchmark suite: one cleaned parquet + meta json per dataset.

Each dataset gets: target column 'y', task in {bin, multi, reg}, group in
{numeric, semantic, text, post}, and lists of numeric / categorical / text columns.
Rows are capped at MAX_N with a fixed seed (stratified for classification).
"""
import json, re, numpy as np, pandas as pd
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
OUT = Path(__file__).parent / "data" / "prepared"
OUT.mkdir(parents=True, exist_ok=True)
MAX_N = 2500
rng = np.random.default_rng(0)


def cap(df, task):
    if len(df) <= MAX_N:
        return df.reset_index(drop=True)
    if task == "reg":
        return df.sample(MAX_N, random_state=0).reset_index(drop=True)
    frac = MAX_N / len(df)
    parts = [g.sample(max(1, int(round(len(g) * frac))), random_state=0) for _, g in df.groupby("y")]
    return pd.concat(parts).sample(frac=1, random_state=0).head(MAX_N).reset_index(drop=True)


def save(name, df, task, group, text_cols=(), source="", year=""):
    df = df.dropna(subset=["y"]).copy()
    if task in ("bin", "multi"):
        df["y"] = df["y"].astype(str)
        vc = df["y"].value_counts()
        df = df[df["y"].isin(vc[vc >= 20].index)]
        df["y"] = pd.Categorical(df["y"]).codes
    else:
        df["y"] = df["y"].astype(float)
    df = cap(df, task)
    feats = [c for c in df.columns if c != "y"]
    text_cols = [c for c in text_cols if c in feats]
    num_cols, cat_cols = [], []
    for c in feats:
        if c in text_cols:
            df[c] = df[c].astype("string").fillna("")
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("string")
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float)
            num_cols.append(c)
        else:
            df[c] = df[c].astype("string")
            cat_cols.append(c)
    df.to_parquet(OUT / f"{name}.parquet")
    meta = dict(name=name, task=task, group=group, n=len(df), num_cols=num_cols,
                cat_cols=cat_cols, text_cols=text_cols, n_classes=int(df.y.nunique()) if task != "reg" else None,
                source=source, data_year=year)
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"{name:16s} {task:5s} {group:8s} n={len(df):5d} num={len(num_cols):2d} cat={len(cat_cols):2d} text={len(text_cols)}")


TT = "TidyTuesday (rfordatascience/tidytuesday)"

# ---------------- numeric-only controls ----------------
w = pd.read_csv(RAW / "winequality-white.csv", header=None,
                names=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                       "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"])
w["y"] = (w.pop("quality") >= 6).astype(int)
save("wine_quality", w, "bin", "numeric", source="UCI via jbrownlee/Datasets")

p = pd.read_csv(RAW / "pima-indians-diabetes.csv", header=None,
                names=["pregnancies", "plasma glucose", "diastolic blood pressure", "triceps skinfold", "serum insulin",
                       "body mass index", "diabetes pedigree", "age", "y"])
save("pima_diabetes", p, "bin", "numeric", source="UCI via jbrownlee/Datasets")

a = pd.read_csv(RAW / "abalone.csv", header=None,
                names=["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
                       "shell weight", "y"])
a = a.drop(columns="sex")  # keep purely numeric
save("abalone", a, "reg", "numeric", source="UCI via jbrownlee/Datasets")

h = pd.read_csv(RAW / "housing.csv", header=None,
                names=["crime rate", "residential zoned share", "industrial share", "borders river", "nitric oxides",
                       "rooms per dwelling", "share built before 1940", "distance to employment centres",
                       "highway access index", "property tax rate", "pupil teacher ratio", "B", "lower status share", "y"])
h = h.drop(columns="B")
save("boston_housing", h, "reg", "numeric", source="UCI via jbrownlee/Datasets")

# ---------------- semantic categorical (pre-2024, public, likely memorised) ----------------
ad = pd.read_csv(RAW / "adult.csv", header=None, skipinitialspace=True,
                 names=["age", "workclass", "fnlwgt", "education", "education years", "marital status", "occupation",
                        "relationship", "race", "sex", "capital gain", "capital loss", "hours per week",
                        "native country", "y"]).replace("?", np.nan)
ad = ad.drop(columns="fnlwgt")
save("adult_income", ad, "bin", "semantic", source="UCI Adult via jbrownlee/Datasets")

bk = pd.read_csv(RAW / "bank.csv", sep=";").drop(columns=["duration"])  # duration leaks the outcome
bk = bk.rename(columns={"y": "y", "emp.var.rate": "employment variation rate", "cons.price.idx": "consumer price index",
                        "cons.conf.idx": "consumer confidence index", "euribor3m": "euribor 3 month rate",
                        "nr.employed": "number employed", "day_of_week": "day of week", "pdays": "days since last contact",
                        "poutcome": "previous outcome"})
save("bank_marketing", bk, "bin", "semantic", source="UCI Bank Marketing via selva86/datasets")

ti = pd.read_csv(RAW / "titanic.csv").drop(columns=["PassengerId", "Ticket"])
ti = ti.rename(columns={"Survived": "y", "Pclass": "passenger class", "SibSp": "siblings or spouses aboard",
                        "Parch": "parents or children aboard"})
save("titanic", ti, "bin", "semantic", text_cols=["Name"], source="Kaggle Titanic via datasciencedojo/datasets")

# ---------------- text-rich (pre-2024) ----------------
wm = pd.read_csv(RAW / "winemag-data-130k-v2.csv").drop(columns=["Unnamed: 0", "taster_twitter_handle", "region_2"])
wm = wm.rename(columns={"points": "y"}).sample(20000, random_state=0)
save("wine_reviews", wm, "reg", "text", text_cols=["description", "title", "designation"], source=TT + " 2019-05-28", year="2017")

rm = pd.read_csv(RAW / "ramen_ratings.csv").drop(columns=["review_number"]).rename(columns={"stars": "y"})
save("ramen_ratings", rm, "reg", "text", text_cols=["variety"], source=TT + " 2019-06-04", year="2019")

ch = pd.read_csv(RAW / "chocolate.csv").drop(columns=["ref"]).rename(columns={"rating": "y"})
ch["cocoa_percent"] = ch["cocoa_percent"].str.rstrip("%").astype(float)
save("chocolate_bars", ch, "reg", "text", text_cols=["specific_bean_origin_or_bar_name", "most_memorable_characteristics"],
     source=TT + " 2022-01-18", year="2021")

nf = pd.read_csv(RAW / "netflix_titles.csv").drop(columns=["show_id", "cast", "date_added", "listed_in"])
nf["y"] = nf.pop("rating").isin(["TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G", "PG", "TV-PG"]).astype(int)
save("netflix_family", nf, "bin", "text", text_cols=["title", "description", "director"], source=TT + " 2021-04-20", year="2021")

# ---------------- post-cutoff tables (published 2025-2026, after the embedder's 2024 release) ----------------
lb = pd.read_csv(RAW / "longbeach.csv", parse_dates=["dob", "intake_date"])
lb["age at intake (years)"] = (lb["intake_date"] - lb["dob"]).dt.days / 365.25
lb["intake month"] = lb["intake_date"].dt.month.astype(float)
lb["y"] = lb["was_outcome_alive"].astype(int)
lb = lb[["animal_type", "primary_color", "secondary_color", "sex", "intake_condition", "intake_type", "intake_subtype",
         "reason_for_intake", "jurisdiction", "age at intake (years)", "intake month", "y"]]
save("animal_shelter", lb, "bin", "post", source=TT + " 2025-03-04", year="2025")

rp = pd.read_csv(RAW / "repairs.csv", low_memory=False)
rp = rp[rp.repaired.isin(["yes", "no", "half"])]
rp["y"] = (rp.pop("repaired") == "yes").astype(int)
rp = rp[["country", "kind_of_product", "category", "brand", "estimated_year_of_production", "y"]]
rp = rp.sample(20000, random_state=0)
save("repair_cafes", rp, "bin", "post", source=TT + " 2026-04-07", year="2026")

cs = pd.read_csv(RAW / "world_castles.csv")
cs = cs[["name", "country", "year", "century", "sitelinks", "pageviews", "lat", "lon", "category"]].rename(columns={"category": "y"})
save("world_castles", cs, "multi", "post", text_cols=["name"], source=TT + " 2026-09-01", year="2026")

ex = pd.read_csv(RAW / "exped_tidy.csv", encoding="latin-1")
ex["y"] = ex.pop("SUCCESS1").astype(int)
ex = ex[["PEAKID", "YEAR", "SEASON_FACTOR", "HOST_FACTOR", "ROUTE1", "NATION", "SPONSOR", "TOTMEMBERS", "TOTHIRED",
         "O2USED", "O2NONE", "CAMPS", "ROPE", "AGENCY", "y"]].rename(columns={
    "PEAKID": "peak id", "YEAR": "year", "SEASON_FACTOR": "season", "HOST_FACTOR": "host country", "ROUTE1": "route",
    "NATION": "expedition nationality", "SPONSOR": "sponsor", "TOTMEMBERS": "team members", "TOTHIRED": "hired staff",
    "O2USED": "oxygen used", "O2NONE": "no oxygen", "CAMPS": "camps", "ROPE": "fixed rope (m)", "AGENCY": "trekking agency"})
save("himalaya_expeditions", ex, "bin", "post", text_cols=["sponsor", "route"], source=TT + " 2025-01-21", year="2025")

ms = pd.read_csv(RAW / "monsters.csv")
ms["y"] = np.log1p(ms.pop("cr"))
ms = ms[["size", "type", "descriptive_tags", "alignment", "ac", "hp_number", "speed_base_number", "str", "dex", "con",
         "int", "wis", "cha", "skills", "resistances", "vulnerabilities", "immunities", "senses", "languages", "y"]]
save("dnd_monsters", ms, "reg", "post", text_cols=["skills", "resistances", "immunities", "senses", "languages"],
     source=TT + " 2025-05-27 (D&D 2024 SRD)", year="2025")

rc = pd.read_csv(RAW / "all_recipes.csv")
rc["y"] = np.log1p(rc.pop("calories"))
rc = rc[["name", "ingredients", "servings", "prep_time", "cook_time", "total_time", "total_ratings", "y"]]
rc = rc.sample(20000, random_state=0, replace=False) if len(rc) > 20000 else rc
save("recipe_calories", rc, "reg", "post", text_cols=["name", "ingredients"], source=TT + " 2025-09-16", year="2025")

qc = pd.read_csv(RAW / "qatarcars.csv")
qc["y"] = np.log(qc.pop("price"))
save("qatar_cars", qc, "reg", "post", source=TT + " 2025-12-09", year="2025")

mv = pd.read_csv(RAW / "movies.csv")
mv = mv[mv.views.notna()].drop_duplicates("title")


def runtime_min(s):
    m = re.match(r"(\d+)H (\d+)M", str(s))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan


mv["runtime (minutes)"] = mv["runtime"].map(runtime_min)
mv["release year"] = pd.to_datetime(mv["release_date"], errors="coerce").dt.year
mv["y"] = np.log1p(mv.pop("views"))
mv = mv[["title", "available_globally", "runtime (minutes)", "release year", "y"]]
save("netflix_views", mv, "reg", "post", text_cols=["title"], source=TT + " 2025-07-29", year="2025")
