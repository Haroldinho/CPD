# Detecting Anomalies in M/M/1 queue dynamics:

## Resolved Items:

- Validation of M/M/1 DEDS against asymptotic expected wait times and variance
- First phase of detection test: Detection threshold h_t as a function of ARL_0
- Autocorrelation results for different batch sizes
- Normality convergence of m.g.f. (batch sizes 1-80)
- Validation of non-overlapping batch means Autocorrelation
- Deciding on batch size based on Autocorrelation

## Unresolved issues:

- Understanding warm-up period
- How to combine multiple change-point tests with different batch sizes?

Multiple parallel tests with different batch sizes on same data:
The tests are correlated => look at Bonferroni inequality What happens if they don't all detect a change? What happens
if they give different change points?

- What is the overall performance of the test (detection delay/ ARL_1, prob. correct detection, ARL_0,...)?

## Next steps:

1) Re-run the Autocorrelation results with more replications (e.g. 2500) and divide variance by sqrt(# of reps.)
2) Obtain FAR for a single batch size
3) Use Bonferroni inequality to bound the type 1 error for multiple tests Prob that k out of n tests detect a change
   need FAR for each batch size (get it from ARL_0?)

## Future:

- What is the overall performance of the test (detection delay/ ARL_1, prob. correct detection, ARL_0,...)?