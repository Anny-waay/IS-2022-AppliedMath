#include <iostream>
#include <math.h>
#include "Methods.h"

using namespace std;

double f(double x){
    return log(x)*sin(x)*pow(x,2);
}

double func1(double x){
    return (x-0.7)*(x+2.4)*(x+0.9)*(x-3.5);
}

double func2(double x){
    return x*sin(0.5*log(x)*x)+1;
}

double func3(double x){
    return sin(x)-4*pow(x,2)*(x-1)*(x+0.5)*(x-2);
}

void Test(double (*f)(double), double a, double b,  double eps, int n) {
    cout << "------------ Segment [" << a << "," << b << "] ------------" << endl;
    cout << "------------ Eps = " << eps << " ------------" << endl << endl;
    DichotomyMethod(f, a, b, eps);
    GoldenRationMethod(f, a, b, eps);
    FibonacciMethod(f, a, b, eps, n);
    ParabolasMethod(f, a, b, eps);
    BrentMethod(f, a, b, eps);
}
void TestAllEps(double (*f)(double), double a, double b,  int n)
{
    double eps = 0.01;
    cout << "------------ Segment [" << a << "," << b << "] ------------" << endl;
    cout << "------------ Eps = 0.01 ------------" << endl << endl;
    DichotomyMethod(f, a, b, eps);
    GoldenRationMethod(f, a, b, eps);
    FibonacciMethod(f, a, b, eps, n);
    ParabolasMethod(f, a, b, eps);
    BrentMethod(f, a, b, eps);
    eps = 0.0001;
    cout << "------------ Eps = 0.0001 ------------" << endl << endl;
    DichotomyMethod(f, a, b, eps);
    GoldenRationMethod(f, a, b, eps);
    FibonacciMethod(f, a, b, eps, 0);
    ParabolasMethod(f, a, b, eps);
    BrentMethod(f, a, b, eps);
    eps = 0.000001;
    cout << "------------ Eps = 0.000001 ------------" << endl << endl;
    DichotomyMethod(f, a, b, eps);
    GoldenRationMethod(f, a, b, eps);
    FibonacciMethod(f, a, b, eps, 0);
    ParabolasMethod(f, a, b, eps);
    BrentMethod(f, a, b, eps);
    eps = 0.00000001;
    cout << "------------ Eps = 0.00000001 ------------" << endl << endl;
    DichotomyMethod(f, a, b, eps);
    GoldenRationMethod(f, a, b, eps);
    FibonacciMethod(f, a, b, eps, 0);
    ParabolasMethod(f, a, b, eps);
    BrentMethod(f, a, b, eps);
}

int main() {
    TestAllEps(f, 0.5, 0.8, 0);
    TestAllEps(f, 3, 6, 0);
    Test(func1, -5, 5,0.00001, 0);
    Test(func2, 0, 10,0.00001, 0);
    Test(func3, -1, 1,0.00001, 0);
    return 0;
}
