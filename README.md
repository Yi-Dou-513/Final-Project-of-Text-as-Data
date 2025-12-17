## Final Project Materials: _Who Talks About What? Classifying Political Attention Across Levels_

Materials for my final project, "Who Talks About What? Classifying Political Attention Across Levels".

> __Abstract:__
> This paper studies how elected officials communicate across different levels of government under a federal system using social media text data. Drawing on posts from U.S. elected officials on X in 2024, I developed a pipeline that combined an unsupervised topic model with word embeddings to classify texts into federal, state, and local issue levels. Latent topics identified by an LDA model were mapped onto these three dimensions using embedding-based similarity measures with orthogonalization to address semantic overlap. Document-level labels were then constructed as weighted averages of topic-level scores and used to train supervised learning models, including a ridge multinomial logistic regression and a DistilBERT classifier. The models outperformed a naive baseline and produced predictions that align with institutional expectations when applied to an external data set of congressional tweets. Overall, the results suggest that embedding-based labeling offers a scalable and interpretable approach for studying multi-level political communication.


## Tutorial 

This README file provides an overview of the materials for my final project. The R codes used in the project can be found under the folder **codes**. The data I scraped and used to build the models are under the folder **data**. All the images in the report are available at the folder **outputs**.



## Codes
- `01-preprocessing.r`: aggregate and preprocess the raw data.

- `02-topic model.r`: cross validate and train the topic model, and also create figure1.

- `03-embeddings.r`: use word embeddings to score the topics and observations.

- `04-labeling.r`: use the scores to label the observations, and also create figure3

- `05-visualization for topic model.r`: create figure2 and figure4. 

- `06-supervised learning.r`: train the ridge model and compare the perfomance, and also create figure5.

- `07-application.r`: apply the ridge model to the tweets_congress data, and also create figure6.

- `bert_classifiers.py`: train the DistillBERT model. These codes are mainly written by AI.

- `scrape_tweets.py`: scrape data from X. These codes are mainly written by AI.

## Data

- `2024_tweets_raw.csv`: The data I scraped from X. It's used for building the topic model.

- `final_data_for_bert.csv`: used to build the DistillBERT model.

- `official_data.csv`: meta data of the elected officials from the DAPR database. 

Please notice that there are also three data sources used in the project which are not here, because they are too big to be uploaded into this repo. Please see the original tweet ids from DAPR database [here](https://doi.org/10.7910/DVN/A9EPYJ), the tweets_congress data [here](https://www.dropbox.com/scl/fi/mj8ldtyqwpkelnpyv0aja/tweets_congress.csv?rlkey=oltuf3odk3trrwjfl0a6ka25t&e=1&dl=0), and the Glove word embeddings [here](https://nlp.stanford.edu/projects/glove/)


## Report

The final project report is located in the report folder:

- `report.pdf` — PDF version of the report  
- `report.docx` — Word version of the report 

The report summarizes the motivation, data, methods, results and discussions of the project.

