# Interpreting predictive process monitoring 
## Outcome prediction using different buckets and encoding methods.
The <a href="https://github.com/irhete/predictive-monitoring-benchmark">predictive monitoring benchmark</a> is used. We thank the authors of the benchmark on outcome oriented predictions, for providing the source code which enabled further study on the interpretability of the models.

The data is evaluated for the following logs. The feature importances and the generated local explanations are used for model interpretations. The explanations are available in the notebooks.
BPIC 2011, BPIC 2012, and BPIC 2015

Explanations enable us to do the following:

1. Compare and use the relevant encoding and bucketing methods

BPIC 2012 results

<a href="https://github.com/renuka98/benchmark_interpretability/blob/master/bpic2012accepted_explanations_bucket_single_encoding_agg_cls_xgboost.ipynb">BPIC 2012 Single Bucket, Aggregation Encoding</a>

<a href="https://github.com/renuka98/benchmark_interpretability/blob/master/bpic2012_Accepted_explanations_bucket_prefix_encoding_agg_cls_xgboost.ipynb">BPIC 2012 Prefix Bucket, Aggregation Encoding</a>

<a href="https://github.com/renuka98/benchmark_interpretability/blob/master/bpic2012_accepted_explanations_bucket_prefix_encoding_index_cls_xgboost.ipynb">BPIC 2012 Prefix Bucket, Index Encoding</a>

2. Understand the model explanations in the context of process model


<a href="https://github.com/renuka98/benchmark_interpretability/blob/master/bpic2011_explanations_bucket_prefix_encoding_agg_cls_xgboost.ipynb">BPIC 2011 Prefix Bucket , Aggregation Encoding</a>


<a href="https://github.com/renuka98/benchmark_interpretability/blob/master/bpic2011_explanations_bucket_prefix_encoding_index_cls_xgboost.ipynb">BPIC 2011 Prefix Bucket , Index Encoding</a>

3. Detect any issues related to the relevant use of features (e.g. leakage)

<a href="https://github.com/renuka98/benchmark_interpretability/blob/master/bpic2015_explanations_bucket_single_encoding_agg_cls_xgboost.ipynb">BPIC 2015 Single Bucket, Aggregation Encoding</a>

<a href="https://github.com/renuka98/benchmark_interpretability/blob/master/bpic2015_explanations_bucket_prefix_encoding_agg_cls_xgboost.ipynb">BPIC 2015 Prefix Bucket, Aggregation Encoding</a>

<a href="https://github.com/renuka98/benchmark_interpretability/blob/master/bpic2015_explanations_bucket_prefix_encoding_index_cls_xgboost.ipynb">BPIC 2015 Prefix Bucket, Index Encoding</a>


