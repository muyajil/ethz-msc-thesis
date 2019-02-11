# Code documentation

## dataset/

- Under this directory we will find the code that is used to generate the dataset object used by the model.
- This will include transformations, batching etc done such that we can directly use the datapoints in the model.
- The dataset object should provide an interface such that the model can iterate through the datapoints directly.
- Since all the data is hosted on GCS the code for a specific commit hash produces the data used by the model directly.

## model/

- In this directory we can find the model implementation.
- Evaluation results should always be correlated with a commit hash, such that we can reconstruct the results.