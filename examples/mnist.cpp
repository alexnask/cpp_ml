#include <iostream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <set>

#include <neural_network.h>
#include <ranges.h>

// To big endian: std::ranges::swap(b, b + sizeof(TYPE))

void print_params(auto const& nn) {
    auto cout_it = std::ranges::ostream_iterator<double> { std::cout, " " };

    std::cout << "Weights: ";
    nn->read_weights(cout_it);
    std::cout << '\n';

    std::cout << "Biases: ";
    nn->read_biases(cout_it);
    std::cout << '\n';
}

template <typename NumT>
void read_params_from_file(char const* path, auto& nn) {
    struct ReadIt {
        using difference_type [[maybe_unused]] = std::ptrdiff_t;
        using value_type = NumT;
        using pointer [[maybe_unused]] = NumT const*;
        using reference = NumT const&;
        using iterator_category [[maybe_unused]] = std::input_iterator_tag;

        std::ifstream *fin;
        value_type current_val;

        reference operator*() {
            fin->read(reinterpret_cast<char*>(&current_val), sizeof(NumT));
            return current_val;
        }

        ReadIt& operator++() {
            return *this;
        }

        ReadIt operator++(int) {
            return *this;
        }

        struct sentinel {
            bool operator==(ReadIt const& it) const {
                return !it.fin || !(*it.fin);
            }

            bool operator!=(ReadIt const& it) const {
                return it.fin && !!(*it.fin);
            }
        };

        bool operator==(sentinel const&) const {
            return !fin || !(*fin);
        }

        bool operator!=(sentinel const&) const {
            return fin && !!(*fin);
        }
    };

    std::ifstream fin { path, std::ios::in | std::ios::binary };
    if (!fin) {
        std::cout << "File '" << path << "' could not be opened, keeping default parameters.\n";
        return;
    }

    std::uint32_t magic;
    if (!fin.read(reinterpret_cast<char*>(&magic), sizeof(magic))) return;

    if (magic != 0xFABA) return;

    std::size_t param_count;
    if (!fin.read(reinterpret_cast<char*>(&param_count), sizeof(param_count))) return;

    if (param_count != nn->param_count()) {
        std::cout << "Expected " << nn->param_count() << " parameters in file, got " << param_count << ", aborting.\n";
        return;
    }

    auto begin = ReadIt { &fin, 0 };
    auto end = typename ReadIt::sentinel {};
    auto in_range = std::ranges::subrange(begin, end);

    nn->write_weights(in_range);
    nn->write_biases(in_range);
}

template <typename NumT>
void write_params_to_file(char const* path, auto& nn) {
    std::ofstream fout { path, std::ios::out | std::ios::trunc | std::ios::binary };
    if (!fout) {
        std::cout << "File '" << path << "' could not be opened for writing.\n";
        return;
    }

    constexpr std::uint32_t magic = 0xFABA;
    if (!fout.write(reinterpret_cast<char const*>(&magic), sizeof(magic))) return;

    constexpr auto param_count = std::remove_reference_t<decltype(nn)>::element_type::param_count();
    if (!fout.write(reinterpret_cast<char const*>(&param_count), sizeof(param_count))) return;

    struct WriteIt {
        struct wrapper_t {
            std::ofstream *fout;

            wrapper_t& operator=(NumT const& val) {
                fout->write(reinterpret_cast<char const*>(&val), sizeof(NumT));
                return *this;
            }
        };

        using difference_type [[maybe_unused]] = std::ptrdiff_t;
        using value_type = wrapper_t;
        using pointer [[maybe_unused]] = wrapper_t*;
        using reference = wrapper_t&;
        using iterator_category [[maybe_unused]] = std::output_iterator_tag;

        value_type wrapper;

        reference operator*() {
            return wrapper;
        }

        WriteIt& operator++() {
            return *this;
        }

        WriteIt operator++(int) {
            return *this;
        }
    };

    auto begin = WriteIt { { &fout } };

    nn->read_weights(begin);
    nn->read_biases(begin);
}

template <typename T>
void switch_endianess(T& t) {
    auto ptr = reinterpret_cast<char*>(&t);
    std::ranges::reverse(ptr, ptr + sizeof(T));
}

auto label_data_it(std::string const& path) {
    struct label_it {
        using reference = Tensor<double, 10> const&;

        std::ifstream fin;
        Tensor<double, 10> current_val;
        std::uint32_t elem_count;

        label_it(std::ifstream&& _fin, Tensor<double, 10> const& _current_val, std::uint32_t _elem_count) :
                    fin(std::move(_fin)), current_val(_current_val), elem_count(_elem_count) {
            read_value();
        }

        reference operator*() {
            return current_val;
        }

        label_it& operator++() {
            read_value();
            return *this;
        }

    private:
        void read_value() {
            uint8_t c;
            fin.read(reinterpret_cast<char*>(&c), 1);

            std::ranges::fill(current_val, 0);

            // c: [0, 9]
            assert(c < 10);
            current_val[{c}] = 1;
        }
    };

    std::ifstream fin { path, std::ios::in | std::ios::binary };
    if (!fin) {
        throw std::runtime_error { std::string { "File '" } + path + "' could not be opened.\n" };
    }

    // TODO: We are assuming little endian, check it.
    std::uint32_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    switch_endianess(magic);

    if (magic != 2049) {
        throw std::runtime_error { std::string { "File '" } + path + "' not a valid MNIST image data file.\n" };
    }

    std::uint32_t n_labels;
    fin.read(reinterpret_cast<char*>(&n_labels), sizeof(n_labels));
    switch_endianess(n_labels);

    std::cout << "File '" << path << "' contains " << n_labels << " labels.\n";

    return label_it { std::move(fin), {}, n_labels };
}

auto image_data_it(std::string const& path) {
    struct image_it {
        using reference = Tensor<double, 28*28> const&;

        std::ifstream fin;
        Tensor<double, 28*28> current_val;
        std::uint32_t elem_count;

        image_it(std::ifstream&& _fin, Tensor<double, 28*28> const& _current_val, std::uint32_t _elem_count) :
                    fin(std::move(_fin)), current_val(_current_val), elem_count(_elem_count) {
            read_value();
        }

        reference operator*() {
            return current_val;
        }

        image_it& operator++() {
            read_value();
            return *this;
        }

    private:
        void read_value() {
            uint8_t c;
            for (std::size_t i = 0; i < 28*28; i++) {
                fin.read(reinterpret_cast<char*>(&c), 1);
                current_val[{i}] = c / 255.f;
            }
        }
    };

    std::ifstream fin { path, std::ios::in | std::ios::binary };
    if (!fin) {
        throw std::runtime_error { std::string { "File '" } + path + "' could not be opened.\n" };
    }

    // Assuming little endian.
    std::uint32_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    switch_endianess(magic);

    if (magic != 2051) {
        throw std::runtime_error { std::string { "File '" } + path + "' not a valid MNIST image data file.\n" };
    }

    std::uint32_t n_images;
    fin.read(reinterpret_cast<char*>(&n_images), sizeof(n_images));
    switch_endianess(n_images);

    std::cout << "File '" << path << "' contains " << n_images << " images.\n";

    std::uint32_t val;
    fin.read(reinterpret_cast<char*>(&val), sizeof(val));
    switch_endianess(val);
    assert(val == 28);
    fin.read(reinterpret_cast<char*>(&val), sizeof(val));
    switch_endianess(val);
    assert(val == 28);

    return image_it { std::move(fin), {}, n_images };
}

template <typename InputIt, typename IndexIt>
auto filter_with_indices(InputIt& input_it, IndexIt const& index_it) {
    struct iterator {
        using reference = /* typename */ InputIt::reference;

        InputIt& input_it;
        IndexIt index_it;
        std::size_t last_index;

        iterator(InputIt& _input_it, IndexIt const& _index_it) : input_it(_input_it), index_it(_index_it), last_index(0) {}

        reference operator*() {
            return *input_it;
        }

        iterator& operator++() {
            read_value();
            return *this;
        }

    private:
        void read_value() {
            auto index = *index_it;
            ++index_it;
            while (last_index < index) {
                ++input_it;
                last_index++;
            }
            assert(last_index == index);
        }
    };

    return iterator { input_it, index_it };
}

int main(int argc, char const* argv[]) {
    constexpr std::size_t Threads = 4;

    constexpr auto ReLU = [](auto const& x) -> autodiff::ValueType<decltype(x)> {
        if (x < 0) return 0;
        return x;
    };

    // Make a Neural Network, initialize all parameters to 0.
    auto nn = std::make_unique<NeuralNetwork<ReLU, double, 28*28, 32, 10>>();

    char const* params_path = "examples/mnist.params";
    std::string data_path = "examples/data/mnist/";

    if (argc > 1) {
        data_path = argv[1];
    
        if (argc > 2) {
            params_path = argv[2];
        }
    }

    read_params_from_file<double>(params_path, nn);

    constexpr auto param_count = decltype(nn)::element_type::param_count();
    std::cout << "Neural network parameter count: " << param_count << '\n';
    std::cout << "Size of neural network in memory: " << sizeof(decltype(nn)::element_type) << " bytes \n";

    constexpr auto CostFunction = [] <typename AutoDiffVal> (Tensor<AutoDiffVal, 10>& out, Tensor<double, 10> const& expected_out)
                                 -> autodiff::ValueType<AutoDiffVal> {

        // Softmax normalization.
        auto temp = std::make_unique<autodiff::val<double, param_count>>(0);

        for (auto& elem : out) {
            elem = autodiff::exp(elem);
            *temp += elem;
        }

        out /= *temp;
        *temp = 0;

        // Calculate cross entropy.
        constexpr double epsilon = 1e-15;
        std::size_t i = 0;
        for (auto const& elem : out) {
            *temp -= expected_out[{i++}] * autodiff::log(elem + epsilon);
        }

        return *temp;
    };

    constexpr auto SamePred = [] <typename AutoDiffVal> (Tensor<AutoDiffVal, 10>& out, Tensor<double, 10> const& expected_out) {
        auto prediction_idx = std::ranges::max_element(out) - out.begin();
        auto true_idx = std::ranges::max_element(expected_out) - expected_out.begin();

        return prediction_idx == true_idx;
    };

    // TODO: Time how long it takes with diff Batch Sizes then parallelize inside each batch.
    constexpr std::size_t BatchSize = 20;
    constexpr std::size_t TrainingLoops = 20;
    std::size_t correct_predictions = 0;
    {
        auto training_images_it = image_data_it(data_path + "train-images.idx3-ubyte");
        auto training_labels_it = label_data_it(data_path + "train-labels.idx1-ubyte");

        assert(training_images_it.elem_count == training_labels_it.elem_count);

        // Choose random indices to train from.
        std::set<std::size_t> indices;
        {
            std::random_device rd;
            std::mt19937 gen { rd() };
            std::uniform_int_distribution<std::size_t> distr { 0, training_images_it.elem_count-1 };

            while (indices.size() != TrainingLoops * BatchSize) {
                indices.emplace(distr(gen));
            }
        }

        auto input_it = filter_with_indices(training_images_it, indices.begin());
        auto true_output_it = filter_with_indices(training_labels_it, indices.begin());

        for (std::size_t loop = 0; loop < TrainingLoops; loop++) {
            std::cout << "LOOP " << loop + 1 << '/' << TrainingLoops << '\n';
            auto [batch_cost, batch_correct_predictions] = nn->train<BatchSize, Threads, CostFunction, SamePred>(0.01, input_it, true_output_it);
            std::cout << "Batch cost: " << batch_cost << "\n\n";

            correct_predictions += batch_correct_predictions;
        }

        write_params_to_file<double>(params_path, nn);
    }
    std::cout << "Training prediction accuracy: " << 100.f * correct_predictions/(BatchSize * TrainingLoops) << "% (" << correct_predictions << '/' << BatchSize * TrainingLoops << ")\n";

    std::cout << "Testing...\n";
    constexpr std::size_t TestingLoops = 10;
    correct_predictions = 0;
    {
        auto testing_images_it = image_data_it(data_path + "t10k-images.idx3-ubyte");
        auto testing_labels_it = label_data_it(data_path + "t10k-labels.idx1-ubyte");
    
        assert(testing_images_it.elem_count == testing_labels_it.elem_count);

        // Choose random indices to train from.
        std::set<std::size_t> indices;
        {
            std::random_device rd;
            std::mt19937 gen { rd() };
            std::uniform_int_distribution<std::size_t> distr { 0, testing_images_it.elem_count-1 };

            while (indices.size() != TestingLoops * BatchSize) {
                indices.emplace(distr(gen));
            }
        }

        auto input_it = filter_with_indices(testing_images_it, indices.begin());
        auto true_output_it = filter_with_indices(testing_labels_it, indices.begin());

        for (std::size_t loop = 0; loop < TestingLoops; loop++) {
            std::cout << "LOOP " << loop + 1 << '/' << TestingLoops << '\n';
            auto [batch_cost, batch_correct_predictions] = nn->feed_forward<BatchSize, Threads, CostFunction, SamePred>(input_it, true_output_it);
            std::cout << "Batch cost: " << batch_cost << "\n\n";

            correct_predictions += batch_correct_predictions;
        }
    }
    std::cout << "Testing prediction accuracy: " << 100.f * correct_predictions/(TestingLoops * BatchSize) << "% (" << correct_predictions << '/' << TestingLoops * BatchSize << ")\n";

    return 0;
}
