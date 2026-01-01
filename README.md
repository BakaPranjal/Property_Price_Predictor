# CDC Open Project 2025

This repository contains the complete workflow for a multimodal machine learning project developed as part of the **CDC Open Project 2025**.  
The project integrates **tabular data** with **satellite/map images** fetched using geographic coordinates to build and evaluate regression models.

---

## Project Workflow

1. Fetch satellite images using latitude and longitude  
2. Preprocess tabular data  
3. Prepare multimodal datasets  
4. Train and evaluate regression models  

---

## 1. Image Fetching (`map_fetcher.py`)

This script downloads satellite images using the **Mapbox Static Images API** and appends image paths to the dataset.

### Requirements
- A valid **Mapbox access token**
- Internet connectivity

### Setup

Open `map_fetcher.py` and replace:

```python
ACCESS_TOKEN = "your access token"
```

with your personal Mapbox access token.

### Expected Input

The input CSV files must contain the following columns:
- `lat`  — latitude  
- `long` — longitude  

### Running the Script

```bash
python map_fetcher.py
```

### What the Script Does
- Fetches satellite images for each row in the dataset  
- Saves images to:
  - `Map_Images/` (training data)
  - `Map_Images_Test/` (test data)
- Adds an `image_path` column to the dataset  
- Outputs:
  - `train_with_images.csv`
  - `test_with_images.csv`

---

## 2. Data Preprocessing (`preprocessing.ipynb`)

This notebook handles:
- Loading raw datasets  
- Data cleaning and filtering  
- Feature selection  
- Scaling numeric features  
- Preparing final inputs for model training  

### How to Use

```bash
jupyter notebook preprocessing.ipynb
```

Run the notebook cells **from top to bottom**.  
Ensure the paths to `train_with_images.csv` and `test_with_images.csv` are correctly specified.

### Output
- Cleaned and scaled tabular features  
- Processed datasets ready for model training  

---

## 3. Model Training (`model_training.ipynb`)

This notebook builds and trains machine learning models using:
- Tabular features  
- Satellite images through a CNN-based architecture (ResNet)  

### Key Features
- Multimodal learning (image + tabular)  
- Transfer learning using pretrained ResNet  
- Proper train–validation split  
- Regularization and early stopping  
- Evaluation using MAE and MSE  

### How to Use

```bash
jupyter notebook model_training.ipynb
```

Run the notebook sequentially to:
1. Load processed data  
2. Load image datasets  
3. Build the multimodal model  
4. Train and validate the model  
5. Evaluate performance  

---

## Important Notes

- Validation and test images should **not** be augmented  
- Ensure there are **no missing values** in the `image_path` column  
- Images act as an auxiliary modality; tabular data remains the primary signal  
- CNN backbones are frozen initially to reduce overfitting  

---

## Dependencies

Recommended Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow requests pillow jupyter
```

---

## Outputs

Depending on execution, the project may generate:
- Trained models  
- Evaluation metrics  
- Loss curves  
- Predictions on test data  

---

## Common Issues

- **Mapbox API errors**: Verify access token and API limits  
- **Missing images**: Ensure image paths exist on disk  
- **Overfitting**: Use pretrained backbones and early stopping  

---

## Author

**Pranjal (BakaPranjal)**  
CDC Open Project 2025
