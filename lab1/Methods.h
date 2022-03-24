#ifndef LAB1_METHODS_H
#define LAB1_METHODS_H

void DichotomyMethod(double (*f)(double),double a,double b,double eps);
void GoldenRationMethod(double (*f)(double),double a,double b,double eps);
void FibonacciMethod(double (*f)(double),double a,double b,double eps, int n);
void ParabolasMethod(double (*f)(double), double a, double b, double eps);
void BrentMethod(double (*f)(double), double a, double c, double eps);
#endif //LAB1_METHODS_H