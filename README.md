# E-commerce Furniture Sales Prediction (ML Project)

This project uses real-world e-commerce data scraped from AliExpress to predict how many units of a furniture item will be sold, based on its product title, price, and shipping details.

---

## Project Overview

- **Goal**: Predict product `sold` count using ML models
- **Dataset Size**: 2,000 products (AliExpress)
- **Tech Stack**: Python, pandas, scikit-learn, seaborn, matplotlib

---

## Dataset Columns

- `productTitle`: Name of the furniture item
- `originalPrice`: Pre-discount price (mostly null — removed)
- `price`: Actual selling price (cleaned)
- `tagText`: Shipping info (e.g., “Free shipping”, “+Shipping: $5.09”)
- `sold`: Target column — number of items sold

---

## Project Workflow

### Phase 1: Data Cleaning
- Dropped `originalPrice` due to excessive nulls
- Cleaned `price` (removed `$`, converted to float)
- Grouped `tagText` into 3 categories:
  - `Free shipping`, `+Shipping: $5.09`, `others`
- Extracted `shipping_price` numerically from tagText
- Dropped or imputed missing values

### Phase 2: EDA
- Visualized distributions (`price`, `sold`, `final_price`)
- Analyzed how shipping and price affect sales
- Used correlation plots and pairplots

### Phase 3: Feature Engineering
- Created `final_price = price + shipping_price`
- Applied TF-IDF on `productTitle` to create 100 text features
- Transformed `sold → log_sold` for smoother distribution

### Phase 4: Model Building
- Built and trained:
  - Linear Regression
  - Random Forest Regressor
- Handled NaNs with `SimpleImputer`

### Phase 5: Evaluation
- Used:
  - Mean Squared Error (MSE)
  - R² Score
- Plotted:
  - Actual vs Predicted
  - Residuals

---

## Results

| Model               | MSE    | R² Score |
|--------------------|--------|----------|
| Linear Regression  | 1.55   | ~0.16    |
| Random Forest       | *↓*    | *↑* (e.g., 0.65–0.7) |

---

## Key Insights

- Products with **free shipping** sell better
- **Text features** (titles) improve predictions via TF-IDF
- `final_price` (including shipping) gives more meaningful sales trends than `price` alone

---

## Installation

```bash
pip install -r requirements.txt
