#include <ranges.h>

#include <array>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <autodiff.h>
#include <neural_network.h>

constexpr auto Sigmoid = [] (auto const& x) -> autodiff::ValueType<decltype(x)> {
    auto exp_x = autodiff::exp(x);
    return exp_x/(1 + exp_x);
};

// Really simple neural network that learns to XOR.
int main() {
    std::array xor_input_table {
        Tensor<double, 2> {0.0, 1.0},
        Tensor<double, 2> {0.0, 0.0},
        Tensor<double, 2> {1.0, 1.0},
        Tensor<double, 2> {1.0, 0.0},
    };

    std::array xor_output_table { 1.0, 0.0, 0.0, 1.0 };

    NeuralNetwork<Sigmoid, double, 2, 2, 1> net {};

    constexpr auto RMSE = [] (auto& out, auto const& expected_out) {
        return autodiff::abs(out[{0}] - expected_out);
    };

    constexpr auto SamePred = [] (auto const& out, auto const& expected_out) {
        return autodiff::abs(out[{0}] - expected_out) <= 0.2;
    };

    constexpr std::size_t EPOCHS = 100'000;
    constexpr double LEARNING_RATE = 0.1;
    constexpr std::size_t ThreadCount = 4;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Training with 1 thread\n";
    auto start_time = std::chrono::steady_clock::now();
    for (std::size_t epoch = 0; epoch < EPOCHS; epoch++) {
        auto xor_input = xor_input_table.begin();
        auto xor_output = xor_output_table.begin();

        net.train<4, 1, RMSE, SamePred>(LEARNING_RATE, xor_input, xor_output);
    }
    std::chrono::duration<double> diff = std::chrono::steady_clock::now() - start_time;
    std::cout << "Training finished in " << diff.count() << " seconds. Testing...\n";

    {
        auto xor_input = xor_input_table.begin();
        auto xor_output = xor_output_table.begin();
        auto [cost, correct] = net.feed_forward<4, 1, RMSE, SamePred>(xor_input, xor_output);
        std::cout << "Final cost: " << cost << " (" << correct << "/4 predictions)\n";
    }

    std::cout << "Training with " << ThreadCount << " threads\n";
    start_time = std::chrono::steady_clock::now();
    for (std::size_t epoch = 0; epoch < EPOCHS; epoch++) {
        auto xor_input = xor_input_table.begin();
        auto xor_output = xor_output_table.begin();

        net.train<4, ThreadCount, RMSE, SamePred>(LEARNING_RATE, xor_input, xor_output);
    }
    diff = std::chrono::steady_clock::now() - start_time;
    std::cout << "Training finished in " << diff.count() << " seconds. Testing...\n";

    {
        auto xor_input = xor_input_table.begin();
        auto xor_output = xor_output_table.begin();
        auto [cost, correct] = net.feed_forward<4, 1, RMSE, SamePred>(xor_input, xor_output);
        std::cout << "Final cost: " << cost << " (" << correct << "/4 predictions)\n";
    }

    return 0;
}
