# topic_modeling

Topic Modeling Project for Data Mining and Text Analytics Unit

This project demonstrates a topic modeling program using Latent Dirichlet Allocation (LDA) from scikit-learn. It allows users to analyze text summaries from a CSV dataset, and the program returns the discovered topics along with visualizations and detailed analysis results.

## Topic Modeling with LDA on arXiv_scientific dataset

A machine learning project that implements Latent Dirichlet Allocation (LDA) for unsupervised topic extraction from scientific research papers' summaries. The project includes robust text preprocessing, stochastic sampling from large datasets, and interactive visualizations to interpret semantic clusters.

## Project Overview
This project demonstrates a complete NLP pipeline for discovering hidden thematic structures in large archives of scientific text. It includes:
* **Smart Data Loading**: Efficiently handles large CSV datasets using random sampling and chunking.
* **Text Preprocessing Pipeline**: Advanced cleaning, stopword removal, and lemmatization from summaries.
* **LDA Topic Modeling**: Implements Scikit-Learn's Latent Dirichlet Allocation to find semantic patterns.
* **Interactive Visualization**: Generates static charts (Matplotlib) and interactive HTML reports (Plotly).
* **Semantic Analysis**: Automatically identifies key concepts from raw text.

## Project Structure

```
topic_modeling//
├── README.md                              # This file
├── LICENSE                                # MIT License
├── acknowledgment                         # Credits and contributions
├── requirements.txt                       # Python dependencies
├── topic_modeling.py                      # Main script
├── data/
│   └── arxiv_papers.csv                   # Input dataset
└── visualizations/                        # Output folder (generated)
    ├── complete_analysis_results.csv      # Full analysis data
    ├── document_topic_heatmap.png         # Topic probabilities heatmap
    ├── interactive_topic_dist.html        # Interactive visualization
    ├── topic_distribution_pie.png         # Document distribution
    ├── topic_words_details.csv            # Word scores per topic
    └── topic_words.png                    # Key words for each topic
```
## Dataset
 
The `arxiv_papers.csv` file is not included in this repository due to GitHub file size limits (>100MB).
 
You can download the dataset from Kaggle: 
https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset/data
 
After downloading, place the file in the `data/` folder.

## Key features include:
* **Random Sampling**: Efficiently loads a random subset of documents from large datasets.
* **Preprocessing**: Text cleaning, stop-word removal, and lemmatization using NLTK.
* **Analysis**: Topic extraction using Scikit-Learn's CountVectorizer and LDA.
* **Visualization**: Generates word distributions, document-topic heatmaps, and interactive HTML charts.

## Getting Started

### Prerequisites

Make sure you have Python installed on your machine. You can install the required packages using the following command on your terminal or bash shell:

pip install pandas numpy scikit-learn nltk matplotlib seaborn plotly

### Installation

**Clone the repository:**

git clone https://github.com/pcamillacp-create/topic_modeling.git

**Navigate to the project directory:**

cd topic_modeling

## Usage

**Run the script by executing the following command in the terminal:**

python topic_modeling.py

The program will:
1. Load random samples from the CSV dataset
2. Preprocess the text data (lowercase, remove stopwords, lemmatization)
3. Create a document-term matrix using CountVectorizer
4. Train an LDA model to discover topics
5. Generate visualizations (bar charts, pie chart,heatmaps, interactive HTML)
6. Save complete analysis results to CSV files

## Configuration

You can modify the configuration parameters in the script:

- `csv_path`: Path to your CSV dataset
- `text_column`: Name of the column containing text data
- `sample_size`: Number of documents to analyze (default: 2000)
- `n_topics`: Number of topics to discover (default: 5)
- `max_features`: Maximum vocabulary size (default: 2000)

## Output Files

The program generates the following files in the `visualizations/` folder:

- `topic_words.png` - Key words for each topic
- `topic_distribution_pie.png` - Document distribution across topics
- `document_topic_heatmap.png` - Topic probabilities heatmap
- `complete_analysis_results.csv` - Full analysis data with document references
- `topic_words_details.csv` - Word scores per topic

## License
This project is open source and available under the MIT License.