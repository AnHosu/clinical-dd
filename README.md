# Selecting Sites for Clinical Trials

I recently encountered an interesting dataset of historical data for clinical trials, and thought it would be a great opportunity for some data analysis and Bayesian modelling.

The data set contains information about a few trials, the sites they were run at, and the country they were run in. The original intention behind the data was to facilitate site selection for future clinical trials. Specifically, it was the general hope that the data could be used to build a structured approach to site selection for a an upcoming trial.

At first glance, this should be simple. Given the available data we could arbitrarily pick any measure or measures that would seem to reflect site quality. Then we either use that metric to choose a site from within the dataset or we gather the metric for new sites and pick the best one.

However, there are two major obstacles: data completeness and data quality. We would not be able to compare sites, where the quality metric is not available or unreliable, and very few features in the dataset are complete or completely trustworthy.

# Model-first Approach

Let's arbitrarily pick the quality measure `recruitment_speed` which we will calculate for a given trial site as `no_of_patients / enrolled_months`. This seems a sensible criterium, as we would often want to pick the site, which could recruit patients the fastest, so the trial can start quickly or so we can obtain more data within an allotted timeframe.

However, there are some immediate concerns when choosing this metric:
- Is is properly normalised so it is comparable across sites?
- What about sites where `no_of_patients` or `enrolled_months` arent available?
- What about new sites, where comparable clinical trials have never been run before?
- Are previous trials representative for future trials? What if the trials are for different disease areas?

There are probably more valid concerns, but we'll shelf them for now, and focus on the two issues of extrapolation. Sites with missing data and new sites are two sides of the same problem, i.e. we can treat new sites as having missing data.

The problem then becomes one of extrapolating or imputing missing data. Here is what I have done:
 - Combined the datasets
 - Calculated the proposed target variable where possible
 - Split the dataset into observations that require imputation and observations that will be used for fitting the model
 - Selected some example features to work with
 - Defined a Bayesian Linear Regression model
 - Conditioned the model and done the imputation
 - Estimated leave one out cross validation generalisation

 The code is in `site_selection.ipynb`.

 Here are examples of additional things one could do:
  - Structured feature selection
  - Bespoke model
  - More careful priors
  - Model selection using LOO-CV estimates
  - Looked at the posterior predictions for the target value
