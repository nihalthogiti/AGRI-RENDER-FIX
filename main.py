
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agricultural Yield Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model and columns
model_columns = joblib.load("model_columns.pkl")
col_index = {c: i for i, c in enumerate(model_columns)}


class LeanForest:
    """Random Forest (log1p target) stored as plain numpy arrays inside
    agricultural_yield_model.pkl. Gives identical predictions to the original
    scikit-learn model but needs ~130 MB RAM instead of 500+ MB."""

    def __init__(self, path):
        d = joblib.load(path)
        self.left, self.right = d["left"], d["right"]
        self.feat, self.thr, self.val = d["feat"], d["thr"], d["val"]
        self.offsets = d["offsets"]

    def predict_one(self, x):
        x = np.asarray(x, dtype=np.float32).astype(np.float64)  # sklearn casts X to float32
        node = self.offsets.copy()
        while True:
            l = self.left[node]
            active = l != -1  # -1 means leaf
            if not active.any():
                break
            idx = node[active]
            go_left = x[self.feat[idx]] <= self.thr[idx]
            node[active] = np.where(go_left, l[active], self.right[idx]) + self.offsets[active]
        return float(np.expm1(self.val[node].mean()))  # target was log1p-transformed


model = LeanForest("agricultural_yield_model.pkl")


# Only required user inputs
class PredictionInput(BaseModel):
    year: int
    temp_max: float
    temp_min: float
    humidity: float
    soil_root: float
    soil_top: float
    precip_mm: float
    rainfall: float
    temp_range: float
    crop: str
    country: str


@app.get("/")
def home():
    return {
        "message": "Agricultural Yield Prediction API is running"
    }

@app.get("/options")
def get_options():

    crops = [
        col.replace("crop_", "", 1)
        for col in model_columns
        if col.startswith("crop_")
    ]

    # Clean country dropdown while preserving original model categories
    country_mapping = {
        "Australia": "Australia",
        "Brazil": "Brazil",
        "Canada": "Canada",
        "China": "China",
        "Egypt": "Egypt",
        "France": "France",
        "Germany": "Germany",
        "India": "India",
        "Indonesia": "Indonesia",
        "Italy": "Italy",
        "Japan": "Japan",
        "Mexico": "Mexico",
        "Nigeria": "Nigeria",
        "Pakistan": "Pakistan",
        "South Africa": "South Africa",
        "Spain": "Spain",
        "Thailand": "Thailand",
        "Turkey": "T Rkiye",
        "United States": "United States Of America"
    }

    available_countries = {
        col.replace("country_", "", 1)
        for col in model_columns
        if col.startswith("country_")
    }

    countries = [
        {"label": label, "value": value}
        for label, value in country_mapping.items()
        if value in available_countries
    ]

    return {
        "crops": crops,
        "countries": countries
    }
@app.post("/predict")
def predict_yield(data: PredictionInput):

    try:
        input_data = data.model_dump()

        crop = input_data.pop("crop").strip()
        country = input_data.pop("country").strip()

        # Initialize all model features with zero
        x = np.zeros(len(model_columns), dtype=np.float64)

        # Fill user-provided numerical values
        for col, value in input_data.items():
            if col not in col_index:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature not found in model: {col}"
                )
            x[col_index[col]] = value

        # Crop / country one-hot encoding
        crop_column = "crop_" + crop
        if crop_column in col_index:
            x[col_index[crop_column]] = 1

        country_column = "country_" + country
        if country_column in col_index:
            x[col_index[country_column]] = 1

        # Prediction
        prediction = model.predict_one(x)

        return {
            "crop": crop,
            "country": country,
            "predicted_yield_kg_per_ha": round(float(prediction), 2),
            "unit": "kg/ha",
            "message": "Prediction successful"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
