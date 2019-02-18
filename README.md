# ZeroShotModel

Implementation of a zero shot classification model using the cifar-10 dataset.

Uses glove vectors from https://nlp.stanford.edu/projects/glove/
ZSL classes taken from google via scraper.

Achieves 78% and 51% accuracy respectively for ZSL classes of helicopter and submarine under a top 3 metric (true labels were within the top 3 predictions of the model - true label was amongst the closest 3 vectors to the feature prediction).

Training uses random sampling with a semi-deep CNN, restricted due to memory constraints on GPU.
