# Amazon Sales Analysis with Image Quality Metrics

CAP 3764 - Advanced Data Science
Team 1 - Fall 2025

## Problem Statement

In the Amazon marketplace, we want to understand what makes products successful. Lots of people study pricing and reviews, but not many look at how the product images affect sales.

Our main question: Do image quality metrics (like visual clutter, colors, and backgrounds) affect sales performance along with the usual stuff like reviews and ratings?

We're trying to figure out what factors lead to better sales rankings so sellers can improve their product listings.

## Background

Amazon has millions of products, so it's really competitive for sellers. The Best Seller Rank (BSR) tells you how well a product is selling - lower numbers mean better sales. If we can understand what drives BSR, it could help:

- Sellers make better product listings
- Marketers create better images
- Us learn how to build models for e-commerce data

Product images are important because they're the first thing shoppers see. Our dataset has some cool computer vision metrics that measure image quality like:

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

We collected data from over 18,000 Amazon products using the Amazon SP-API and some custom scripts we wrote. Here's how we got the data:

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
│   └── 02_exploratory_data_analysis.ipynb
├── src/
│   ├── __init__.py
│   └── data_collection.py
├── docs/
├── environment.yml
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
conda env create -f environment.yml
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

Phase 3 - Modeling (coming soon):
- Make new features
- Build regression models to predict BSR
- Test the models (RMSE, MAE, R-squared)
- Figure out what the results mean

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


## Initial Observations

From what we've seen so far:
- The dataset has lots of different product categories with different sales levels
- Image quality varies a lot between products
- We have both visual and text features which is good for analysis
- BSR ranges from very popular to very unpopular products

We'll get more detailed insights as we do the full EDA.

## References

- Amazon SP-API docs
- Computer vision papers on image quality
- Course materials from CAP 3764
