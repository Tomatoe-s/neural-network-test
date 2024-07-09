#include <iostream>
#include <vector>
#include "math.cpp"
#include "neuron.cpp"
#include <cmath>



int main(){
    std::vector<double> weights = {2,3};
    double bias = 0;
  
    neuralNetwork network = neuralNetwork();
    std::vector<std::vector<double>> data = {
    {-2, -1},   // Alice
    {25, 6},    // Bob
    {17, 4},    // Charlie
    {-15, -6}   // Diana
};

std::vector<double> all_y_trues = {1, 0, 0, 1};

    network.train(data,all_y_trues);
    std::vector<double> emily = {-7,-3};
    std::vector<double> frank = {20,2};
    std::vector<double> who = {20,5};
    std::cout << "Emily: " << network.feedForward(emily) << std::endl;
std::cout << "Frank: " << network.feedForward(frank) << std::endl;
std::cout << "Me: " << network.feedForward(who) << std::endl;



}