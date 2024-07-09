#include <iostream>
#include <vector>
#include "math.cpp"
#include "neuron.cpp"
#include <cmath>
#include <string>
#include <random>

// Function to normalize input data
std::vector<double> normalize(double weight, double height) {
    return {(weight - 150) / 100, (height - 65) / 15};
}

// Function to denormalize output
double denormalize(double value) {
    return 1 / (1 + std::exp(-value));  // Sigmoid function
}

int main() {
    neuralNetwork network = neuralNetwork();

    // Expanded training data
    std::vector<std::vector<double>> data = {
        normalize(120, 63), normalize(110, 61), normalize(115, 62), normalize(125, 64), // Females
        normalize(180, 70), normalize(175, 69), normalize(185, 72), normalize(170, 68)  // Males
    };
    std::vector<double> all_y_trues = {1, 1, 1, 1, 0, 0, 0, 0};

    // Train the network multiple times
    for (int i = 0; i < 10; ++i) {
        network.train(data, all_y_trues);
    }

    // Test cases
    std::vector<double> emily = normalize(115, 61);
    std::vector<double> frank = normalize(190, 72);

    std::cout << "Emily: " << denormalize(network.feedForward(emily)) << std::endl;
    std::cout << "Frank: " << denormalize(network.feedForward(frank)) << std::endl;

    while (true) {
        double weight, height;
        std::cout << "Input weight in lbs: ";
        std::cin >> weight;
        std::cout << "Input height in inches: ";
        std::cin >> height;

        if (std::cin.fail()) {
            std::cerr << "Invalid input. Please enter numeric values." << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }

        std::vector<double> normalized_input = normalize(weight, height);
        double result = denormalize(network.feedForward(normalized_input));
result-=.1;
        std::string gender = (result >= 0.5) ? "Female" : "Male";
        double confidence = (result >= 0.5) ? result : 1 - result;

        std::cout << "Result: " << gender << ", Confidence: " << confidence << std::endl;

        std::string response;
        std::cout << "Do you want to continue (yes/no)? ";
        std::cin >> response;
        if (response != "yes") {
            break;
        }
    }

    return 0;
}