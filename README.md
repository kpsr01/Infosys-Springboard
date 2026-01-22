# AI Based Recommendation System

## Overview
This project is an advanced recommendation system utilizing machine learning techniques to provide personalized product suggestions. The application is built with Streamlit, offering an interactive interface for exploring different recommendation strategies.

**Live Demo:** [Check out the live application here!](https://kpsr01-infosys-springboard-app-axqvzb.streamlit.app/)

## Features

### 1. User & Guest Modes
- **Guest Mode (0_guest)**: New users receive popularity-based recommendations (Top Rated Items) to help them discover trending products.
- **Registered Users**: Existing users access a full personalization suite including Collaborative and Hybrid filtering based on their history.

### 2. Recommendation Engines
The system implements four distinct recommendation approaches:

- **Rating Based Filtering**: Suggests top-rated products based on average user ratings and review counts. Ideal for new users or identifying trending items.
- **Content-Based Filtering**: Recommends items similar to a selected product using feature similarity.
- **Collaborative Filtering**: Analyzes similar user behaviors to suggest products that users with similar tastes have liked.
- **Hybrid Approach**: Combines multiple techniques to leverage the strengths of each, providing highly accurate and personalized recommendations.

## Tech Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Logic**: Python

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `app.py`: Main application entry point and UI logic.
- `hybrid_approach.py`: Logic for the hybrid recommendation engine combining content and collaborative methods.
- `content_based_filtering.py`: Logic for content-based recommendation algorithms.
- `collaborative_based_filtering.py`: Logic for collaborative filtering algorithms.
- `rating_based_recommendation.py`: Logic for rating/popularity-based recommendations.
- `preprocess_data.py`: Data cleaning and preprocessing utilities.
- `evaluation_metrics.py`: Metrics for evaluating model performance.
- `clean_data.csv`: The processed dataset used by the recommendation engines.
- `requirements.txt`: Project dependencies.
