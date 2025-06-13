# Problem

Consider the case of Parameter learning, a special case of hybrid models, 
where a machine learning model, $g_{\phi_g}$, uses known covariates $x_{Mi}$ at site $i$,  
to predict a subset of the parameters, $\theta$ of the process based model, $f$.

We are interested in both,
- the uncertainty of hybrid model predictions, $ŷ$ (predictive posterior), and
- the uncertainty of process-model parameters $\theta$, including their correlations
  (posterior)

For example we have soil organic matter process-model that predicts carbon stocks for 
different sites. We need to parameterize the unknown carbon use efficiency (CUE) of the soil
microbial community that differs by site, but is hypothesized to correlate with climate variables
and pedogenic factors, such as clay content.
We apply a machine learnign model to estimate CUE and fit it end-to-end with other
parameters of the process-model to observed carbon stocks.
In addtion to the predicted CUE, we are interested in the uncertainty of CUE and its correlation with other parameters, such a the capacity of the soil minerals to bind carbon. 
I.e. we are interetes in the entire posterior probability distribution of the model parameters.

