# Amazon Sales Analysis with Image Quality Metrics

CAP 3764 - Advanced Data Science
Team 1 - Fall 2025

## Problem Statement

In the Amazon marketplace, we want to understand what makes products successful. Lots of people study pricing and reviews, but not many look at how the product images affect sales.

Our main question: Do image quality metrics (visual clutter, colors, and backgrounds) affect sales performance along with numerical features eg. reviews and ratings?

We need to determine  what factors lead to better sales rankings so sellers can improve their product listings.

### Research Question Type: Predictive Analysis

**Question**: Can we predict Amazon product sales performance (BSR) using image quality metrics combined with product features?

**Approach**: Regression modeling (not causal inference)

**Rationale**: This is an observational study using existing marketplace data. We're building predictive models to identify relationships between visual metrics and sales outcomes, but cannot establish causation without controlled experiments.

## Background

Amazon has millions of products, so it is a competitive landscape for sellers. The Best Seller Rank (BSR) tells you how well a product is selling - lower numbers mean better sales. If we can understand what drives BSR, it could help:

- Sellers make better product listings
- Marketers create better images
- Us learn how to build models for e-commerce data

Product images are important because they're the first thing shoppers see. Our dataset has some computer vision metrics that measure image quality:

- Visual clutter - how messy or clean the image looks
- Background - if it uses white/neutral backgrounds (Amazon says this is good)
- Colors - how many colors and how they're distributed
- Edge density - how sharp and detailed the image is

## What We're Trying to Do

1. Do exploratory analysis to see how image quality, reviews, and sales are related
2. Find patterns in what makes products successful
3. Build models to predict sales based on all these features
4. Give recommendations based on what we find

## Dataset

We collected data from over 18,000 Amazon products using the Amazon SP-API with custom scripts we wrote. Here's how we got the data:

1. Used Amazon SP-API to get product info, prices, and sales data
2. Got review counts and ratings
3. Downloaded product images and ran computer vision algorithms on them
4. Combined everything together

The dataset has 18,148 products from 2024-2025, mostly electronics and consumer goods.

### Variables

Numerical variables:
- bsr_best - Best Seller Rank (lower is better)
- units_per_month - estimated monthly sales
- sales_velocity_daily - daily sales rate
- predicted_bsr - our model's prediction
- review_count - number of reviews
- avg_rating - average star rating (1-5)
- edge_density - image sharpness measure
- bg_white_pct - percent of white background
- bg_neutral_pct - percent of neutral background
- n_clusters_sig - number of color clusters
- color_entropy - color diversity
- largest_cluster_pct - how much the main color dominates
- clutter_score - overall visual clutter score
- Plus z-score versions of the image metrics for comparison

Categorical variables:
- asin - Amazon product ID
- keyword - search term/category
- source_file - which batch the data came from
- item_name - product title
- brand - product brand
- image_path - where we saved the image
- has_aplus - if it has A+ content
- has_brand_story - if it has brand story section

## Project Structure

```
CAP_3764_2025_Fall_Team_1/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_collection_and_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_modeling_pipeline.ipynb
├── src/
│   ├── __init__.py
│   └── data_collection.py
├── docs/
├── amazon_sales_env.yml
└── README.md
```

## Setup

Clone the repo:
```
git clone https://github.com/fesalfayed/CAP_3764_2025_Fall_Team_1.git
cd CAP_3764_2025_Fall_Team_1
```

Create environment:
```
conda env create -f amazon_sales_env.yml
conda activate amazon_sales_analysis
```

Run Jupyter:
```
jupyter lab
```

## How We're Doing This

Phase 1 - Data Collection:
- Got data from Amazon SP-API
- Downloaded and analyzed images using OpenCV
- Made a data_collection.py module to load everything
- Cleaned the data (removed duplicates, handled missing values)

Phase 2 - EDA:
- Calculate summary statistics for all numerical variables
- Look at distributions of BSR, ratings, image metrics
- Analyze brands and categories
- Make correlation matrices
- Create visualizations

Phase 3 - Modeling (COMPLETE):
- Engineered interaction features from image quality metrics
- Built XGBoost regression model to predict BSR
- Applied log transformation to handle extreme skewness in sales rank distribution
- Implemented stratified train/test split and hyperparameter tuning
- Evaluated model with comprehensive diagnostics and dual-scale metrics
- Extracted business insights from feature importance analysis

## Our Hypotheses

1. Products with less cluttered images should have better BSR (lower numbers)
2. White/neutral backgrounds should correlate with better sales
3. More reviews and better ratings = better BSR (this should work regardless of image quality)
4. Image quality probably affects how much ratings matter for sales

## Tools We're Using

- Python 3.11
- pandas and numpy for data
- matplotlib, seaborn, plotly for visualizations
- scikit-learn for modeling
- Jupyter Lab
- conda for environment
- Git/GitHub
- Amazon SP-API for data collection
- OpenCV/Pillow for image processing

## What We Hope to Find

1. Stats showing how image quality affects sales
2. Models that can predict BSR from product features
3. Guidelines for making better product listings
4. Good visualizations showing the patterns

## Team

- Fesal Fayed
- Maksim Paklov


## Submission 2 Results (Phase 3 - Modeling)

### Model Performance

We built an XGBoost regression model to predict Best Seller Rank using image quality metrics. The model achieved moderate predictive power:

**Key Metrics:**
- **R² = 0.0823** (explains 8.23% of variance in log-transformed BSR)
- **MAE = 2,708 ranks** (typical prediction error on original scale)
- **Median APE = 85.58%** (median percentage error)

While the R² may seem modest, this quantifies an important finding: image quality alone has limited predictive power for sales success. Most variance in sales performance comes from factors we didn't measure, like product category dynamics, brand reputation, customer reviews, and competitive pricing.

### Top Predictive Features

Our feature importance analysis identified three key visual metrics:

1. **image_count (14.06% importance)** - Number of product photos
   - Products with more images (18-27 photos) show systematically different BSR patterns than those with fewer (1-5 photos)
   - Takeaway: Comprehensive visual documentation matters - sellers should maximize image slots with multiple angles, lifestyle shots, and detail close-ups

2. **clean_sharp (9.73% importance)** - Professional photography quality
   - Interaction of white/neutral backgrounds with sharp, high-resolution images
   - Takeaway: Investment in professional lighting and high-resolution cameras pays off

3. **color_balance (9.63% importance)** - Visual complexity
   - Moderate color variety without single-color dominance
   - Takeaway: Balance visual appeal - avoid both extreme minimalism and chaotic clutter

### What This Means

The model's limited explanatory power (8.2% R²) reveals that **image quality changes alone won't reliably turn a low-seller into a best-seller**. With a median error of 85%, predictions are too noisy for precise forecasting. However, the directional relationships are clear: more images, professional presentation, and balanced visual complexity are associated with better sales ranks.

The remaining 92% of variance likely comes from unmeasured factors:
- Product category and niche dynamics
- Brand strength and reputation
- Customer review quality and quantity
- Competitive pricing strategies
- Seasonal demand fluctuations

### Technical Approach

**Data Preparation:**
- Applied log transformation to Best Seller Rank (reduced skewness from 17.66 to 1.07)
- Stratified train/test split (80/20) using quantile-based bins
- Engineered three interaction features based on business hypotheses

**Model Training:**
- XGBoost gradient boosting regressor
- RandomizedSearchCV hyperparameter tuning (50 iterations, 5-fold cross-validation)
- Optimized learning rate, tree depth, number of estimators, and regularization

**Evaluation:**
- Dual-scale metrics (log scale for optimization, original scale for interpretation)
- Overfitting analysis with train-test gap checks
- Learning curves for bias-variance assessment
- Regression diagnostics: residual plots, Q-Q plots, scale-location plots
- Feature importance analysis

### Recommendations

**For Amazon Sellers:**
- Maximize image count - use all available image slots
- Invest in professional photography with clean backgrounds and sharp focus
- Balance visual complexity - showcase features without overwhelming viewers
- Remember: Image quality is ONE component of success, not a silver bullet

**Limitations:**
- This is an observational study - we can't establish causation
- Review metrics were 100% missing from our dataset
- Cross-category model may mask niche-specific patterns
- Temporal effects and seasonal trends not captured

### Future Work

To improve prediction accuracy and understanding:
1. Build category-specific models (Electronics vs Fashion vs Home Goods)
2. Incorporate review data when available
3. Conduct A/B testing with sellers to establish causal relationships
4. Experiment with deep learning image features (CNNs)
5. Collect longitudinal data for time series analysis

---

## Initial Observations (Phase 1-2)

From our exploratory analysis:
- The dataset has lots of different product categories with different sales levels
- Image quality varies a lot between products
- We have both visual and text features which is good for analysis
- BSR ranges from very popular to very unpopular products
- Strong correlation between image count and sales performance
- Professional image quality (clean backgrounds, sharp edges) associated with better ranks

## References

- Amazon SP-API docs
- Computer vision papers on image quality
- Course materials from CAP 3764
