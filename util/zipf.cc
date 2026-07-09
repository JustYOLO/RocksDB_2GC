#include "util/zipf.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

long items;
long base;
double zipfianconstant;
double alpha;
double zetan;
double eta;
double theta;
double zeta2theta;
long countforzeta;
long lastVal;

void init_zipf_generator(long min, long max, double zipf_const) {
  items = max - min + 1;
  base = min;
  zipfianconstant = zipf_const;
  theta = zipfianconstant;
  zeta2theta = zeta(0, 2, 0);
  alpha = 1.0 / (1.0 - theta);
  zetan = zetastatic(0, max - min + 1, 0);
  countforzeta = items;
  eta = (1 - pow(2.0 / items, 1 - theta)) / (1 - zeta2theta / zetan);

  nextValue();
}

double zeta(long st, long n, double initial_sum) {
  countforzeta = n;
  return zetastatic(st, n, initial_sum);
}

double zetastatic(long st, long n, double initial_sum) {
  double sum = initial_sum;
  for (long i = st; i < n; i++) {
    sum += 1 / (pow(i + 1, theta));
  }
  return sum;
}

long nextLong(long item_count) {
  if (item_count != countforzeta) {
    if (item_count > countforzeta) {
      fprintf(stderr,
              "WARNING: Incrementally recomputing Zipfian distribution. "
              "(item_count=%ld; countforzeta=%ld)\n",
              item_count, countforzeta);
      zetan = zeta(countforzeta, item_count, zetan);
      eta = (1 - pow(2.0 / items, 1 - theta)) / (1 - zeta2theta / zetan);
    }
  }

  double u = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
  double uz = u * zetan;
  if (uz < 1.0) {
    return base;
  }

  if (uz < 1.0 + pow(0.5, theta)) {
    return base + 1;
  }

  long ret =
      base + static_cast<long>(item_count * pow(eta * u - eta + 1, alpha));
  setLastValue(ret);
  return ret;
}

long nextValue() { return nextLong(items); }

void setLastValue(long val) { lastVal = val; }
