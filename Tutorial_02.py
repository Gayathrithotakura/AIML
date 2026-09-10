"""
Tutorial 2 — Probability, Bayes theorem, MLE and MAP
CO1
"""
import numpy as np
from scipy.stats import binom

# 1. Bayes theorem: fraud screening example
# P(Fraud)=0.01, P(Alert|Fraud)=0.95, P(Alert|NotFraud)=0.05
p_fraud = 0.01
p_alert_given_fraud = 0.95
p_alert_given_not = 0.05

p_alert = (p_alert_given_fraud * p_fraud
           + p_alert_given_not * (1 - p_fraud))

p_fraud_given_alert = p_alert_given_fraud * p_fraud / p_alert

print("P(Alert) =", p_alert)
print("P(Fraud | Alert) =", p_fraud_given_alert)

# 2. Binomial probability
# Probability of exactly 3 successes in 10 trials, p=0.2
print("\nP(X=3) =", binom.pmf(3, 10, 0.2))

# 3. MLE for Bernoulli probability
# 7 successes out of 10 observations
x = np.array([1,1,1,1,1,1,1,0,0,0])
p_mle = x.mean()
print("\nBernoulli MLE p =", p_mle)

# 4. MAP estimate with Beta prior
# Prior Beta(alpha,beta), likelihood has s successes and f failures
alpha, beta = 2, 2
s, f = 7, 3
# Posterior is Beta(alpha+s, beta+f)
post_alpha = alpha + s
post_beta = beta + f
map_estimate = (post_alpha - 1) / (post_alpha + post_beta - 2)
posterior_mean = post_alpha / (post_alpha + post_beta)

print("Posterior Beta parameters:", post_alpha, post_beta)
print("MAP estimate =", map_estimate)
print("Posterior mean =", posterior_mean)
