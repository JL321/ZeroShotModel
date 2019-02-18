# ZeroShotModel

Implementation of a zero shot classification model using the cifar-10 dataset.

Uses glove vectors from https://nlp.stanford.edu/projects/glove/
ZSL classes taken from google via scraper.

Zero Shot Classes Used:
- Bicycle, Helicopter, Submarine

Achieves 90+% accuracy for top 3 predictions with zsl classes (true labels were within the top 3 predictions of the model - true label was amongst the closest 3 vectors to the feature prediction), and 40%+ for top 1 prediction (argmin distance).

Training uses random sampling with a semi-deep CNN, restricted due to memory constraints on GPU.

Note: Scaling word embeddings to reduce the weights on the feature prediction layer isn't effective

