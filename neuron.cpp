#include <vector>
#include <iostream>
#include "math.cpp"


class Neuron{
private:
    std::vector<double> weights;
    
    double bias = 0.0;

public:
    Neuron(const std::vector<double> neuronWeights, double neuronBias) : weights(neuronWeights), bias(neuronBias) {};
    double feedForward(std::vector<double> inputs){
        double total = math::dot(weights,inputs) + bias;
        return math::sigmoid(total);
    }
};
class neuralNetwork{
private:
    std::vector<double> weights = {};
    std::vector<Neuron> neurons = {};
    std::vector<double> bias = {}; 
   
public:
    neuralNetwork() {
        for (int i =0;i<5;i++){
        weights.push_back(.5);
        }
         for (int i =0;i<3;i++){
        bias.push_back(.5);
        }
        //h1 = Neuron(weights,bias);
        //h2 = Neuron(weights,bias);

        //o1 = Neuron(weights,bias);
    };
    double feedForward(std::vector<double> x){
       double h1 = math::sigmoid(weights[0]*x[0]+weights[1]*x[1]+bias[0]);
        double h2 = math::sigmoid(weights[2]*x[0]+weights[3]*x[1]+bias[1]);
        double o1 = math::sigmoid(weights[4]*h1+weights[5]*h2+bias[3]);

      //  double out_h2 = h2.feedForward(x);

       //double  out_o1 = o1.feedForward(std::vector<double> {out_h1,out_h2});
        return o1;
    } 
    void train(std::vector<std::vector<double>>& data, std::vector<double>& trues) {
    double learn_rate = 0.01;
    int epochs = 900000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < data.size(); ++i) {
            const std::vector<double>& x = data[i];
            double y_true = trues[i];

            // --- Do a feedforward (we'll need these values later)
            double sum_h1 = weights[0] * x[0] + weights[1] * x[1] + bias[0];
            double h1 = math::sigmoid(sum_h1);

            double sum_h2 = weights[2] * x[0] + weights[3] * x[1] + bias[1];
            double h2 = math::sigmoid(sum_h2);

            double sum_o1 = weights[4] * h1 + weights[5] * h2 + bias[2];
            double o1 = math::sigmoid(sum_o1);
            double y_pred = o1;

            // --- Calculate partial derivatives.
            double d_L_d_ypred = -2 * (y_true - y_pred);

            // Neuron o1
            double d_ypred_d_w5 = h1 * math::deriv_sigmoid(sum_o1);
            double d_ypred_d_w6 = h2 * math::deriv_sigmoid(sum_o1);
            double d_ypred_d_b3 = math::deriv_sigmoid(sum_o1);

            double d_ypred_d_h1 = weights[4] * math::deriv_sigmoid(sum_o1);
            double d_ypred_d_h2 = weights[5] * math::deriv_sigmoid(sum_o1);

            // Neuron h1
            double d_h1_d_w1 = x[0] * math::deriv_sigmoid(sum_h1);
            double d_h1_d_w2 = x[1] * math::deriv_sigmoid(sum_h1);
            double d_h1_d_b1 = math::deriv_sigmoid(sum_h1);

            // Neuron h2
            double d_h2_d_w3 = x[0] *  math::deriv_sigmoid(sum_h2);
            double d_h2_d_w4 = x[1] *  math::deriv_sigmoid(sum_h2);
            double d_h2_d_b2 =  math::deriv_sigmoid(sum_h2);

            // --- Update weights and biases
            // Neuron h1
            weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
            weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
            bias[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;

            // Neuron h2
            weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
            weights[3] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
            bias[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;

            // Neuron o1
            weights[4] -= learn_rate * d_L_d_ypred * d_ypred_d_w5;
            weights[5] -= learn_rate * d_L_d_ypred * d_ypred_d_w6;
            bias[2] -= learn_rate * d_L_d_ypred * d_ypred_d_b3;
        }

        // --- Calculate total loss at the end of each epoch
        if (epoch % 10 == 0) {
            std::vector<double> y_preds;
            for (const auto& x : data) {
                y_preds.push_back(feedForward(x));
            }
            double loss = math::mse_loss(trues, y_preds);
            std::cout << "Epoch " << epoch << " loss: " << loss << std::endl;
        }
    }
}
};