data {
  int<lower=1> N;
  int<lower=1> D;
  int<lower=1> K;
  matrix[N, D] X;
  int<lower=1> y[N];
  real<lower=0> s;
}

transformed data {
  row_vector[N] zeros = rep_row_vector(0, N);
}

parameters {
  matrix[K-1, D] w_raw;
}

model {
  matrix[K, N] z = append_row(zeros, w_raw * X');
  for(n in 1:N){
    y[n] ~ categorical_logit(z[,n]);
  }
  to_vector(w_raw) ~ normal(0, s);
}