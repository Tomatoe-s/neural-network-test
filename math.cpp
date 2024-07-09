#include <iostream>
#include <vector>
#include <cmath>
#ifndef MATH_HPP
#define MATH_HPP
namespace math {



    double dot(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of the same dimension to compute dot product.");
        }
        double product = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            product += a[i] * b[i];
        }
        return product;
    }

    double sigmoid(double x) {
        return 1 / (1 + std::exp(-x));
    };
    double deriv_sigmoid(double x){
        double sig = math::sigmoid(x);
        return sig*(1-sig);
    }
    double mse_loss(std::vector<double> yTrue, std::vector<double> yPred){
        double sum = 0;
        for (size_t i =0; i< yTrue.size(); ++i){
            double diff = (yTrue[i]-yPred[i]);
            sum += diff*diff;
        }
       
        return sum/static_cast<double>(yTrue.size());
    }
}
#endif