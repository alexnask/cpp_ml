#ifndef __NEURAL_NETWORK_H
#define __NEURAL_NETWORK_H

#include "ranges.h"

#include <random>
#include <future>
#include <array>
#include <algorithm>
#include <utility>

#include "autodiff.h"
#include "tensors.h"

namespace details {

// Homemade tuple like that constructs tensors from pairs of size_t's
template <typename NumT, std::size_t First, std::size_t Second, std::size_t... Rest>
struct WeightStorage : public WeightStorage<NumT, Second, Rest...> {
    Tensor<NumT, First, Second> data;

    WeightStorage() : WeightStorage<NumT, Second, Rest...>(), data() {}

    template <std::size_t Index>
    auto& get() {
        if constexpr(Index == 0) {
            return data;
        }
        else {
            return WeightStorage<NumT, Second, Rest...>::template get<Index - 1>();
        }
    }

    template <std::size_t Index>
    auto const& get() const {
        if constexpr(Index == 0) {
            return data;
        }
        else {
            return WeightStorage<NumT, Second, Rest...>::template get<Index - 1>();
        }
    }
};

template <typename NumT, std::size_t First, std::size_t Second>
struct WeightStorage<NumT, First, Second> {
    Tensor<NumT, First, Second> data;

    WeightStorage() : data() {}

    template <std::size_t Index>
    auto& get() {
        if constexpr(Index == 0) {
            return data;
        }
        else {
            static_assert("Cannot retrieve item from weight storage.");
            return data;
        }
    }

    template <std::size_t Index>
    auto const& get() const {
        if constexpr(Index == 0) {
            return data;
        }
        else {
            static_assert("Cannot retrieve item from weight storage.");
            return data;
        }
    }
};

// Same as above but with biases and 1 size_t at a time.
template <typename NumT, std::size_t First, std::size_t... Rest>
struct BiasStorage : public BiasStorage<NumT, Rest...> {
    Tensor<NumT, First> data;

    BiasStorage() : BiasStorage<NumT, Rest...>(), data() {}

    template <std::size_t Index>
    auto& get() {
        if constexpr(Index == 0) {
            return data;
        }
        else {
            return BiasStorage<NumT, Rest...>::template get<Index - 1>();
        }
    }

    template <std::size_t Index>
    auto const& get() const {
        if constexpr(Index == 0) {
            return data;
        }
        else {
            return BiasStorage<NumT, Rest...>::template get<Index - 1>();
        }
    }
};

template <typename NumT, std::size_t First>
struct BiasStorage<NumT, First> {
    Tensor<NumT, First> data;

    BiasStorage() : data() {}

    template <std::size_t Index>
    auto& get() {
        if constexpr(Index == 0) {
            return data;
        }
        else {
            static_assert("Cannot retrieve item from bias storage.");
            return data;
        }
    }

    template <std::size_t Index>
    auto const& get() const {
        if constexpr(Index == 0) {
            return data;
        }
        else {
            static_assert("Cannot retrieve item from bias storage.");
            return data;
        }
    }
};

// This struct is used to drop the first variadic template argument before passing to BiasStorage.
template <typename NumT, std::size_t First, std::size_t... Rest>
struct BiasStorageStart : public BiasStorage<NumT, Rest...> {
    BiasStorageStart() : BiasStorage<NumT, Rest...>() {}
};

template <std::size_t I>
auto& get(auto& t) {
    return t.template get<I>();
}

template <std::size_t... Sizes>
constexpr std::size_t param_count(std::index_sequence<Sizes...>) {
    constexpr std::size_t SizeArr[] { Sizes... };

    std::size_t sum = 0;

    for (std::size_t i = 1; i < sizeof...(Sizes); i++) {
        sum += SizeArr[i] * (1 + SizeArr[i-1]);
    }

    return sum;
}

} // End of namespace details.

template <typename T, typename NumT>
concept Activation = std::is_invocable_v<T, autodiff::var<NumT, 1>> &&
                     std::is_same_v<autodiff::val<NumT, 1>, std::invoke_result_t<T, autodiff::val<NumT, 1>>>;


template <auto activation, typename NumT, std::size_t... Sizes>
requires Activation<decltype(activation), NumT>
class NeuralNetwork {
    constexpr static auto LayerCount = sizeof...(Sizes);

    static_assert(LayerCount >= 3, "Neural network needs at least an input, a hidden and an output layer.");

    static constexpr auto SizeSeq = std::index_sequence<Sizes...> {};

    static constexpr auto InputSize = details::get(SizeSeq, 0);
    static constexpr auto OutputSize = details::get(SizeSeq, LayerCount - 1);

    static constexpr auto HiddenLayers = LayerCount - 2;

    static constexpr auto ParameterCount = details::param_count(SizeSeq);

    using DiffT = autodiff::var<NumT, ParameterCount>;
    using ValT = autodiff::val<NumT, ParameterCount>;

    details::WeightStorage<DiffT, Sizes...> weight_data;
    details::BiasStorageStart<DiffT, Sizes...> bias_data;

    template <std::size_t Step, std::size_t S>
    inline auto calculate_step(std::unique_ptr<Tensor<ValT, S>> prev) const {
        if constexpr (Step == HiddenLayers + 1) {
            return prev;
        }
        else {
            // Weights: (Sizes[Step], Sizes[Step + 1])
            auto const& weights = details::get<Step>(weight_data);
            // Biases: (Sizes[Step + 1])
            auto const& biases = details::get<Step>(bias_data);

            constexpr auto PrevLayerSize = details::get(SizeSeq, Step);
            constexpr auto LayerSize = details::get(SizeSeq, Step + 1);
            static_assert(std::is_same_v<decltype(weights), Tensor<DiffT, PrevLayerSize, LayerSize> const&>);
            static_assert(std::is_same_v<decltype(biases), Tensor<DiffT, LayerSize> const&>);

            auto out = std::make_unique<Tensor<ValT, LayerSize>>();
            for (std::size_t i = 0; i < LayerSize; i++) {
                (*prev) *= weights.template view<0>({ i });
                (*out)[{i}] = activation(std::accumulate(prev->begin(), prev->end(), ValT { biases[{i}] }));
            }

            return calculate_step<Step + 1>(std::move(out));
        }
    }

    template <std::size_t I>
    void apply_to_weights(auto&& f) {
        if constexpr(I < LayerCount - 1) {
            f(details::get<I>(weight_data));
            apply_to_weights<I + 1>(std::forward<decltype(f)>(f));
        }
    }

    template <std::size_t I>
    void apply_to_biases(auto&& f) {
        if constexpr(I < LayerCount - 1) {
            f(details::get<I>(bias_data));
            apply_to_biases<I + 1>(std::forward<decltype(f)>(f));
        }
    }

    template <std::size_t I>
    void apply_to_weights(auto&& f) const {
        if constexpr(I < LayerCount - 1) {
            f(details::get<I>(weight_data));
            apply_to_weights<I + 1>(std::forward<decltype(f)>(f));
        }
    }

    template <std::size_t I>
    void apply_to_biases(auto&& f) const {
        if constexpr(I < LayerCount - 1) {
            f(details::get<I>(bias_data));
            apply_to_biases<I + 1>(std::forward<decltype(f)>(f));
        }
    }

public:
    NeuralNetwork() {
        std::random_device rd {};
        std::mt19937 gen { rd() };
        std::normal_distribution<NumT> distr { 0 };

        std::size_t index = 0;
        apply_to_weights<0>([&distr, &gen, &index] (auto& tensor) {
            for (auto& elem : tensor) {
                elem = { distr(gen), index++ };
            }
        });

        apply_to_biases<0>([&distr, &gen, &index] (auto& tensor) {
            for (auto& elem: tensor) {
                elem = { distr(gen), index++ };
            }
        });

        assert(index == ParameterCount);
    }

    constexpr static std::size_t param_count() {
        return ParameterCount;
    }

    auto feed_forward(Tensor<NumT, InputSize> const& input) {
        auto true_input = std::make_unique<Tensor<ValT, InputSize>>();
        std::ranges::copy(input | std::views::transform([](NumT const& x) { return ValT { x }; } ), std::ranges::begin(*true_input));
        return calculate_step<0>(std::move(true_input));
    }

    template <std::size_t BatchSize, std::size_t NThreads, typename It>
    auto feed_forward(It& it) {
        constexpr auto Loops = (BatchSize + NThreads - 1) / NThreads;

        auto out = std::make_unique<Tensor<ValT, OutputSize, BatchSize>>();
        std::size_t batch_idx = 0;
        for (std::size_t loop = 0; loop < Loops; loop++) {
            std::array<std::future<std::unique_ptr<Tensor<ValT, OutputSize>>>, NThreads> futures;

            for (std::size_t i = 0; i < NThreads && batch_idx + i < BatchSize; i++) {
                auto input = *it;
                ++it;

                futures[i] = std::async(std::launch::async, [this] (auto&& tensor) {
                    return feed_forward(tensor);
                }, std::move(input));
            }

            for (std::size_t i = 0; i < NThreads && batch_idx < BatchSize; i++, batch_idx++) {
                auto out_view = out->template view<0>({ batch_idx });
                std::ranges::copy(*futures[i].get(), out_view.begin());
            }
        }

        return out;
    }

    // TODO: This is the same as train() minus a single call, abstract away into a private helper that returns std::move(batch_cost), correct_predictions
    // Returns batch cost and correct predictions.
    template <std::size_t BatchSize, std::size_t NThreads, auto CostF, auto SamePred, typename InputIt, typename ExpOutputIt>
    requires std::is_invocable_v<decltype(CostF), Tensor<ValT, OutputSize>&, Tensor<NumT, OutputSize> const&> &&
             std::is_same_v<ValT, std::invoke_result_t<decltype(CostF), Tensor<ValT, OutputSize>&, Tensor<NumT, OutputSize> const&>> &&
             std::is_invocable_v<decltype(SamePred), Tensor<ValT, OutputSize>&, Tensor<NumT, OutputSize> const&> &&
             std::is_same_v<bool, std::invoke_result_t<decltype(SamePred), Tensor<ValT, OutputSize>&, Tensor<NumT, OutputSize> const&>>
    std::pair<NumT, std::size_t> feed_forward(InputIt& input_it, ExpOutputIt& expected_output_it) {
        constexpr auto Loops = (BatchSize + NThreads - 1) / NThreads;

        auto batch_cost = std::make_unique<ValT>(0);
        std::size_t batch_idx = 0;
        std::size_t correct_predicitons = 0;

        for (std::size_t loop = 0; loop < Loops; loop++) {
            std::array<std::future<std::unique_ptr<Tensor<ValT, OutputSize>>>, NThreads> futures;

            for (std::size_t i = 0; i < NThreads && batch_idx + i < BatchSize; i++) {
                auto input = *input_it;
                ++input_it;

                futures[i] = std::async(std::launch::async, [this] (auto&& tensor) {
                    return feed_forward(tensor);
                }, std::move(input));
            }

            for (std::size_t i = 0; i < NThreads && batch_idx < BatchSize; i++, batch_idx++) {
                auto output = futures[i].get();
                auto expected_output = *expected_output_it;
                ++expected_output_it;

                *batch_cost += CostF(*output, expected_output);
                if (SamePred(*output, expected_output)) {
                    correct_predicitons++;
                }
            }
        }
        *batch_cost /= BatchSize;

        return { batch_cost->value(), correct_predicitons };
    }

    // Returns batch cost and correct predictions.
    template <std::size_t BatchSize, std::size_t NThreads, auto CostF, auto SamePred, typename InputIt, typename ExpOutputIt>
    requires std::is_invocable_v<decltype(CostF), Tensor<ValT, OutputSize>&, Tensor<NumT, OutputSize> const&> &&
             std::is_same_v<ValT, std::invoke_result_t<decltype(CostF), Tensor<ValT, OutputSize>&, Tensor<NumT, OutputSize> const&>> &&
             std::is_invocable_v<decltype(SamePred), Tensor<ValT, OutputSize>&, Tensor<NumT, OutputSize> const&> &&
             std::is_same_v<bool, std::invoke_result_t<decltype(SamePred), Tensor<ValT, OutputSize>&, Tensor<NumT, OutputSize> const&>>
    std::pair<NumT, std::size_t> train(NumT const& learning_rate, InputIt& input_it, ExpOutputIt& expected_output_it) {
        constexpr auto Loops = (BatchSize + NThreads - 1) / NThreads;

        auto batch_cost = std::make_unique<ValT>(0);
        std::size_t batch_idx = 0;
        std::size_t correct_predicitons = 0;

        for (std::size_t loop = 0; loop < Loops; loop++) {
            std::array<std::future<std::unique_ptr<Tensor<ValT, OutputSize>>>, NThreads> futures;

            for (std::size_t i = 0; i < NThreads && batch_idx + i < BatchSize; i++) {
                auto input = *input_it;
                ++input_it;

                futures[i] = std::async(std::launch::async, [this] (auto&& tensor) {
                    return feed_forward(tensor);
                }, std::move(input));
            }

            for (std::size_t i = 0; i < NThreads && batch_idx < BatchSize; i++, batch_idx++) {
                auto output = futures[i].get();
                auto expected_output = *expected_output_it;
                ++expected_output_it;

                *batch_cost += CostF(*output, expected_output);
                if (SamePred(*output, expected_output)) {
                    correct_predicitons++;
                }
            }
        }
        *batch_cost /= BatchSize;
        update_parameters(learning_rate, *batch_cost);

        return { batch_cost->value(), correct_predicitons };
    }

    void update_parameters(NumT const& learning_rate, autodiff::val<NumT, ParameterCount> const& cost) {
        std::size_t index = 0;
        apply_to_weights<0>([learning_rate, cost, &index] (auto& tensor) {
            for (auto& elem : tensor) {
                elem -= learning_rate * cost.derivative(index);
                index++;
            }
        });

        apply_to_biases<0>([learning_rate, cost, &index] (auto& tensor) {
            for (auto& elem: tensor) {
                elem -= learning_rate * cost.derivative(index);
                index++;
            }
        });

        assert(index == ParameterCount);
    }

    // Read/write methods for weights and biases using ranges.
    template <typename Rng>
    requires std::ranges::input_range<Rng> && std::is_same_v<std::ranges::range_value_t<Rng>, NumT>
    void write_weights(Rng const& rng) {
        auto current_it = rng.begin();
        auto end_it = rng.end();

        apply_to_weights<0>([&current_it, end_it] (auto& tensor) {
            for (auto& elem : tensor) {
                if (current_it == end_it) return;

                elem = *current_it;
                current_it++;
            }
        });
    }

    template <typename Rng>
    requires std::ranges::input_range<Rng> && std::is_same_v<std::ranges::range_value_t<Rng>, NumT>
    void write_biases(Rng const& rng) {
        auto current_it = rng.begin();
        auto end_it = rng.end();

        apply_to_biases<0>([&current_it, end_it] (auto& tensor) {
            for (auto& elem : tensor) {
                if (current_it == end_it) return;

                elem = *current_it;
                current_it++;
            }
        });
    }

    template <typename Rng>
    requires std::ranges::output_range<Rng> && std::is_same_v<std::ranges::range_value_t<Rng>, NumT>
    void read_weights(Rng const& rng) const {
        auto current_it = rng.begin();
        auto end_it = rng.end();

        apply_to_weights<0>([&current_it, end_it] (auto& tensor) {
            for (auto const& elem : tensor) {
                if (current_it == end_it) return;

                *current_it = elem;
                current_it++;
            }
        });
    }

    template <typename Rng>
    requires std::ranges::output_range<Rng> && std::is_same_v<std::ranges::range_value_t<Rng>, NumT>
    void read_biases(Rng const& rng) const {
        auto current_it = rng.begin();
        auto end_it = rng.end();

        apply_to_biases<0>([&current_it, end_it] (auto& tensor) {
            for (auto const& elem : tensor) {
                if (current_it == end_it) return;

                *current_it = elem;
                current_it++;
            }
        });
    }

    // Single iterator versions
    template <typename It>
    requires std::ranges::output_iterator<It, NumT>
    void read_weights(It const& it) const {
        auto current_it = it;

        apply_to_weights<0>([&current_it] (auto& tensor) {
            for (auto const& elem : tensor) {
                *current_it = elem.value();
                current_it++;
            }
        });
    }

    template <typename It>
    requires std::ranges::output_iterator<It, NumT>
    void read_biases(It const& it) const {
        auto current_it = it;

        apply_to_biases<0>([&current_it] (auto& tensor) {
            for (auto const& elem : tensor) {
                *current_it = elem.value();
                current_it++;
            }
        });
    }
};

#endif
