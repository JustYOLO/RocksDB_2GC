#pragma once

void init_zipf_generator(long min, long max, double zipf_const);
double zeta(long st, long n, double initial_sum);
double zetastatic(long st, long n, double initial_sum);
long nextLong(long item_count);
long nextValue();
void setLastValue(long val);
