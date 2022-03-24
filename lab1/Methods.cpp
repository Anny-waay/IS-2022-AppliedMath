#include <iostream>
#include <math.h>
#include <vector>

using namespace std;

void DichotomyMethod(double (*f)(double),double a,double b,double eps) {
    int iterCounter =0;
    double delta = ((double)rand()/RAND_MAX)*(eps/2.0);
    double c, x1, x2;
    vector<pair<double, double>> segments;
    double seg = b - a;
    while(b-a>eps){
        c = (a+b)/2;
        x1 = c-delta;
        x2 = c+delta;
        if(f(x1)<f(x2))
            b=x2;
        else
            a=x1;
        ++iterCounter;
        segments.push_back(make_pair(b-a, seg/(b-a)));
        seg = b - a;
    }
    double x = (a+b)/2;
    cout << "Dichotomy" << endl;
    cout << "Min f(x) = " << f(x) << endl;
    cout << "Min x = " << x << endl;
    cout << "Number of iterations = " << iterCounter << endl;
    cout << "Number of function evaluations = " << iterCounter*2 << endl;
    cout << "The segment decreases at iterations:" << endl;
    for (int i = 0; i < segments.size(); i++)
    {
        cout << i+1 << " " << segments[i].first << " " << segments[i].second << endl;
    }
    cout << endl;
}

void GoldenRationMethod(double (*f)(double),double a,double b,double eps){
    int iterCounter =0;
    double phi = (1+sqrt(5))/2;
    double x1=b-(b-a)/phi;
    double x2=a+(b-a)/phi;
    double y1=f(x1);
    double y2=f(x2);
    vector<pair<double, double>> segments;
    double seg = b - a;
    while(b-a>eps){
        if(y1<y2){
            b=x2;
            x2=x1;
            y2=y1;
            x1=b-(b-a)/phi;
            y1=f(x1);
        }
        else
        { a=x1;
            x1=x2; y1=y2;
            x2=a+(b-a)/phi;
            y2=f(x2);
        }
        ++iterCounter;
        segments.push_back(make_pair(b-a, seg/(b-a)));
        seg = b - a;
    }
    double x = (a+b)/2;
    cout << "Golden Ration" << endl;
    cout << "Min f(x) = " << f(x) << endl;
    cout << "Min x = " << x << endl;
    cout << "Number of iterations = " << iterCounter << endl;
    cout << "Number of function evaluations = " << iterCounter+2 << endl;
    cout << "The segment decreases iterations:" << endl;
    for (int i = 0; i < segments.size(); i++)
    {
        cout << i+1 << " " << segments[i].first << " " << segments[i].second << endl;
    }
    cout << endl;
}

void FibonacciMethod(double (*f)(double),double a,double b,double eps, int n){
    vector<long> fib;
    fib.push_back(0);
    fib.push_back(1);
    if (n==0){
        double fib_n_2 = (b-a)/eps;
        int counter = 1;
        while (fib[counter]<fib_n_2){
            counter++;
            fib.push_back(fib[counter-1]+fib[counter-2]);
        }
        n=counter-2;
    }
    else{
        for(int i=2; i<=n+2; i++){
            fib.push_back(fib[i-1]+fib[i-2]);
        }
    }

    double x1 = a + ((double)fib[n]/fib[n+2])*(b-a);
    double x2 = a + ((double)fib[n+1]/fib[n+2])*(b-a);
    double y1 = f(x1);
    double y2 = f(x2);
    vector<pair<double, double>> segments;
    double seg = b - a;
    for(int i=0; i<=n-1; i++){
        if(y1<y2){
            b=x2;
            x2=x1;
            y2=y1;
            x1=a + ((double)fib[n-i]/fib[n-i+2])*(b-a);
            y1=f(x1);
        }
        else{
            a=x1;
            x1=x2;
            y1=y2;
            x2=a + ((double)fib[n-i+1]/fib[n-i+2])*(b-a);
            y2=f(x2);
        }
        segments.push_back(make_pair(b-a, seg/(b-a)));
        seg = b - a;
    }
    double x = (b+a)/2;
    cout << "Fibonacci" << endl;
    cout << "Min f(x) = " << f(x) << endl;
    cout << "Min x = " << x << endl;
    cout << "Number of iterations = " << n << endl;
    cout << "Number of function evaluations = " << n+1 << endl;
    cout << "The segment decreases at iterations: " << endl;
    for (int i = 0; i < segments.size(); i++)
    {
        cout << i+1 << " " << segments[i].first << " " << segments[i].second << endl;
    }
    cout << endl;
}

void ParabolasMethod(double (*f)(double), double a, double b, double eps)
{
    int iterCounter = 0;
    double u, x, fa, fx, fb, fu;
    x = (a + b) / 2;
    fa = f(a);
    fb = f(b);
    fx = f(x);
    vector<double> segments;
    double seg = b - a;
    while (b - a > eps)
    {
        u = x - ((x - a) * (x - a) * (fx - fb) - (x - b) * (x - b) * (fx - fa)) / (2 * ((x - a) * (fx - fb) - (x - b) * (fx - fa)));
        fu = f(u);
        if (fu > fx)
        {
            if (u > x)
            {
                b = u;
                fb = fu;
            }
            else
            {
                a = u;
                fa = fu;
            }
        }
        else
        {
            if (x > u)
            {
                b = x;
                fb = fx;
            }
            else
            {
                a = x;
                fa = fx;
            }
            x = u;
            fx = fu;
        }
        ++iterCounter;
        segments.push_back(seg / (b - a));
        seg = b - a;
    }
    cout << "Parabolas Method" << endl;
    cout << "Min f(x) = " << f(u) << endl;
    cout << "Min x = " << u << endl;
    cout << "Number of iterations = " << iterCounter << endl;
    cout << "Number of function evaluations = " << iterCounter+3 << endl;
    cout << "The segment decreases at iterations:" << endl;
    for (int i = 0; i < segments.size(); i++)
    {
        cout << segments[i] << " ";
    }
    cout << endl << endl;
}

template <class Value>
int Sign(Value Val) {
    if (Val == 0.)  return 0;
    if (Val > 0.)  return 1;
    else return -1;
}

void BrentMethod(double (*f)(double), double a, double c, double eps)
{
    double x, w, v, u, fx, fw, fv;
    double d, e, g;
    int iterCounter = 0;
    double phi = (3 - sqrt(5)) / 2;
    x = (a + c) / 2;
    w = (a + c) / 2;
    v = (a + c) / 2;
    fx = f(x);
    fw = f(w);
    fv = f(v);
    d = c - a;
    e = c - a;
    vector<double> segments;
    double seg = c - a;
    while (d > eps)
    {
        g = e;
        e = d;
        if (!(fx == fw || fx == fv || fv == fw))
        {
            u = x - ((x - w) * (x - w) * (fx - fv) - (x - v) * (x - v) * (fx - fw)) / (2 * ((x - w) * (fx - fv) - (x - v) * (fx - fw)));
            if (u >= a + eps && u <= c - eps && abs(u-x) < g/2)
            {
                d = abs(u - x);
            }
            else
            {
                if (x < (c-a)/2)
                {
                    u = a + phi * (c - x);
                    d = c - x;
                }
                else
                {
                    u = c - phi * (x - a);
                    d = x - a;
                }
            }
        }
        else
        {
            if (x < (c-a)/2)
            {
                u = x + phi * (c - x);
                d = c - x;
            }
            else
            {
                u = x - phi * (x - a);
                d = x - a;
            }
            if (abs(u-x) < eps)
            {
                u = x + Sign(u - x) * eps;
            }
        }
        double fu = f(u);
        if (fu <= fx)
        {
            if (u >= x)
            {
                a = x;
            }
            else
            {
                c = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }
        else
        {
            if ( u >= x)
            {
                c = u;
            }
            else
            {
                a = u;
            }
            if (fu <= fw || w == x)
            {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            }
            else
            {
                if (fu <= fv || v == x || v == w)
                {
                    v = u;
                    fv = fu;
                }
            }
        }
        ++iterCounter;
        segments.push_back(seg / (c - a));
        seg = c - a;
    }
    cout << "Brent Method" << endl;
    cout << "Min f(x) = " << f(x) << endl;
    cout << "Min x = " << x << endl;
    cout << "Number of iterations = " << iterCounter << endl;
    cout << "Number of function evaluations = " << iterCounter + 3 << endl;
    cout << "The segment decreases at iterations:" << endl;
    for (int i = 0; i < segments.size(); i++)
    {
        cout << segments[i] << " ";
    }
    cout << endl << endl;
}
