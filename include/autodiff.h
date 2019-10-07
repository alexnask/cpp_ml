#ifndef __AUTODIFF_H
#define __AUTODIFF_H

#include <array>
#include <tuple>
#include <optional>
#include <utility>
#include <type_traits>
#include <cstddef>
#include <cmath>
#include <cassert>

#include "ranges.h"

namespace autodiff {

// Forward declaration.
template <typename NumT, std::size_t Variables>
requires std::is_arithmetic_v<NumT> && !std::is_const_v<NumT>
class var;

// TODO: Add operators with var overloads.
// TODO: Add bool operators.

// Autodifferentiable value.
// An autodiff::var contains the computed value and the derivative wrt each variable.
// The n-th order derivative wrt a variable x such that x=x0 of a function f is computed in the following way:
//     We evaluate f(x0 + ε) as a polynomial.
//     d^n f/dx^n = coeff of ε^n * n!
// Here, we only calculate the first derivatives.
template <typename NumT, std::size_t Variables>
requires std::is_arithmetic_v<NumT> && !std::is_const_v<NumT>
class val {
    // First element: value
    // Rest: first order derivative wrt each variable (coeff of ε).
    std::array<NumT, Variables + 1> data;

public:
    val() {
        data[0] = 0;
        std::ranges::fill(data.begin() + 1, data.end(), 0);
    }

    val(NumT const& val) {
        data[0] = val;
        std::ranges::fill(data.begin() + 1, data.end(), 0);
    }

    val(NumT const& val, std::size_t var_index) {
        assert(var_index < Variables);

        data[0] = val;
        std::ranges::fill(data.begin() + 1, data.end(), 0);
        data[var_index + 1] = 1;
    }

    val(var<NumT, Variables> const& var) : val(var.value(), var.index()) {}

    // This invalidates derivative data.
    val& operator=(NumT const& val) {
        data[0] = val;
        std::ranges::fill(data.begin() + 1, data.end(), 0);
        return *this;
    }

    val& operator=(var<NumT, Variables> const& val) {
        data[0] = val.value();
        std::ranges::fill(data.begin() + 1, data.end(), 0);
        data[1 + val.index()] = 1;
        return *this;
    }

    NumT const& value() const {
        return data[0];
    }

    NumT const& derivative(std::size_t var_index) const {
        assert(var_index < Variables);
        return data[var_index + 1];
    }

    // Operators.
    val& operator+=(NumT const& rhs) {
        data[0] += rhs;
        return *this;
    }

    val operator+(NumT const& rhs) const {
        val ret = *this;
        ret += rhs;
        return ret;
    }

    val& operator+=(val const& rhs) {
        auto it = rhs.data.begin();
        for (auto& elem : data) {
            elem += *it;
            it++;
        }
        assert(it == rhs.data.end());
        return *this;
    }

    val operator+(val const& rhs) const {
        val ret = *this;
        ret += rhs;
        return ret;
    }

    val& operator+=(var<NumT, Variables> const& rhs) {
        data[0] += rhs.value();
        data[1 + rhs.index()]++;
        return *this;
    }

    val operator+(var<NumT, Variables> const& rhs) const {
        val ret = *this;
        ret += rhs;
        return ret;
    }

    val& operator-=(NumT const& rhs) {
        data[0] -= rhs;
        return *this;
    }

    val operator-(NumT const& rhs) const {
        val ret = *this;
        ret -= rhs;
        return ret;
    }

    val& operator-=(val const& rhs) {
        auto it = rhs.data.begin();
        for (auto& elem : data) {
            elem -= *it;
            it++;
        }
        assert(it == rhs.data.end());
        return *this;
    }

    val operator-(val const& rhs) const {
        val ret = *this;
        ret -= rhs;
        return ret;
    }

    val& operator-=(var<NumT, Variables> const& rhs) {
        data[0] -= rhs.value();
        data[1 + rhs.index()]--;
        return *this;
    }

    val operator-(var<NumT, Variables> const& rhs) const {
        val ret = *this;
        ret -= rhs;
        return ret;
    }

    val operator*(NumT const& rhs) const {
        val ret = *this;
        ret *= rhs;
        return ret;
    }

    val& operator*=(val const& rhs) {
        auto this_value = data[0];
        auto const& rhs_value = rhs.data[0];

        data[0] *= rhs_value;

        auto this_it = data.begin() + 1;
        auto rhs_it = rhs.data.begin() + 1;

        while (this_it != data.end()) {
            auto this_deriv = *this_it;
            auto rhs_deriv = *rhs_it;

            *this_it = this_value * rhs_deriv + rhs_value * this_deriv;

            this_it++;
            rhs_it++;
        }

        assert(rhs_it == rhs.data.end());
        return *this;
    }

    val operator*(val const& rhs) const {
        val ret = *this;
        ret *= rhs;
        return ret;
    }

    val& operator*=(var<NumT, Variables> const& rhs) {
        auto this_value = data[0];
        data[0] *= rhs.value();

        for (std::size_t i = 0; i < Variables; i++) {
            auto this_deriv = data[i + 1];

            data[i + 1] = this_value * rhs.derivative(i) + rhs.value() * this_deriv;
        }

        return *this;
    }

    val operator*(var<NumT, Variables> const& rhs) const {
        val ret = *this;
        ret *= rhs;
        return ret;
    }

    val& operator/=(NumT const& rhs) {
        for (auto& elem : data) {
            elem /= rhs;
        }
        return *this;
    }

    val operator/(NumT const& rhs) const {
        val ret = *this;
        ret /= rhs;
        return ret;
    }

    val& operator/=(val const& rhs) {
        auto const& rhs_value = rhs.data[0];

        data[0] /= rhs_value;
        auto const& new_value = data[0];

        auto this_it = data.begin() + 1;
        auto rhs_it = rhs.data.begin() + 1;

        while (this_it != data.end()) {
            auto this_deriv = *this_it;
            auto rhs_deriv = *rhs_it;

            *this_it = (this_deriv - new_value * rhs_deriv) / rhs_value;

            this_it++;
            rhs_it++;
        }

        assert(rhs_it == rhs.data.end());
        return *this;
    }

    val operator/(val const& rhs) const {
        val ret = *this;
        ret /= rhs;
        return ret;
    }

    val& operator/=(var<NumT, Variables> const& rhs) {
        auto this_value = data[0];
        data[0] /= rhs.value();
        auto const& new_value = data[0];

        for (std::size_t i = 0; i < Variables; i++) {
            auto this_deriv = data[i + 1];

            data[i + 1] =  (this_deriv - new_value * rhs.derivative(i)) / rhs.value();
        }

        return *this;
    }

    val operator/(var<NumT, Variables> const& rhs) const {
        val ret = *this;
        ret /= rhs;
        return ret;
    }

    bool operator<(NumT const& rhs) const {
        return data[0] < rhs;
    }

    bool operator<(val const& rhs) const {
        return data[0] < rhs.data[0];
    }

    bool operator<=(val const& rhs) const {
        return data[0] <= rhs.data[0];
    }

    bool operator>(val const& rhs) const {
        return data[0] > rhs.data[0];
    }

    bool operator>=(val const& rhs) const {
        return data[0] >= rhs.data[0];
    }

    bool operator==(val const& rhs) const {
        return data[0] == rhs.data[0];
    }

    bool operator!=(val const& rhs) const {
        return data[0] != rhs.data[0];
    }

    template <auto F, auto D>
    val apply_function() const {
        val ret = *this;

        auto value_deriv = D(ret.data[0]);
        ret.data[0] = F(ret.data[0]);

        auto it = ret.data.begin() + 1;
        while (it != ret.data.end()) {
            *it *= value_deriv;
            it++;
        }

        return ret;
    }
};

// Autodifferentiable variable
template <typename NumT, std::size_t Variables>
requires std::is_arithmetic_v<NumT> && !std::is_const_v<NumT>
class var {
    NumT var_value;
    std::size_t var_index;

public:
    var() = default;

    var(NumT const& val, std::size_t _var_index) : var_value(val), var_index(_var_index) {
        assert(var_index < Variables);
    }

    var& operator=(NumT const& val) {
        var_value = val;
        return *this;
    }

    NumT const& value() const {
        return var_value;
    }

    std::size_t index() const {
        return var_index;
    }

    NumT derivative(std::size_t idx) const {
        assert(idx < Variables);
        if (idx == var_index) return 1;
        return 0;
    }

    // Operators.
    var& operator+=(NumT const& rhs) {
        var_value += rhs;
        return *this;
    }

    auto operator+(NumT const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret += rhs;
        return ret;
    }

    var& operator+=(var const& rhs) {
        var_value += rhs.var_value;
        return *this;
    }

    auto operator+(var const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret += rhs;
        return ret;
    }

    var& operator+=(val<NumT, Variables> const& rhs) {
        var_value += rhs.value();
        return *this;
    }

    auto operator+(val<NumT, Variables> const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret += rhs;
        return ret;
    }

    var& operator-=(NumT const& rhs) {
        var_value -= rhs;
        return *this;
    }

    auto operator-(NumT const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret -= rhs;
        return ret;
    }

    var& operator-=(var const& rhs) {
        var_value -= rhs.var_value;
        return *this;
    }

    auto operator-(var const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret -= rhs;
        return ret;
    }

    var& operator-=(val<NumT, Variables> const& rhs) {
        var_value -= rhs.value();
        return *this;
    }

    auto operator-(val<NumT, Variables> const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret -= rhs;
        return ret;
    }

    var& operator*=(NumT const& rhs) {
        var_value *= rhs;
        return *this;
    }

    auto operator*(NumT const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret *= rhs;
        return ret;
    }

    var& operator*=(var const& rhs) {
        var_value *= rhs.var_value;
        return *this;
    }

    auto operator*(var const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret *= rhs;
        return ret;
    }

    var& operator*=(val<NumT, Variables> const& rhs) {
        var_value *= rhs.value();
        return *this;
    }

    auto operator*(val<NumT, Variables> const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret *= rhs;
        return ret;
    }

    var& operator/=(NumT const& rhs) {
        var_value /= rhs;
        return *this;
    }

    auto operator/(NumT const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret /= rhs;
        return ret;
    }

    var& operator/=(var const& rhs) {
        var_value /= rhs.var_value;
        return *this;
    }

    auto operator/(var const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret /= rhs;
        return ret;
    }

    var& operator/=(val<NumT, Variables> const& rhs) {
        var_value /= rhs.value();
        return *this;
    }

    auto operator/(val<NumT, Variables> const& rhs) const {
        val<NumT, Variables> ret { *this };
        ret /= rhs;
        return ret;
    }

    bool operator<(NumT const& rhs) const {
        return var_value < rhs;
    }
};

// TODO: Var versions.
// TODO: Assign versions.
// Operators.
template <typename T, typename V, std::size_t Vars>
requires std::is_convertible_v<T, V>
auto operator+(T const& lhs, val<V, Vars> const& rhs) {
    return rhs + lhs;
}

template <typename T, typename V, std::size_t Vars>
requires std::is_convertible_v<T, V>
auto operator-(T const& lhs, val<V, Vars> const& rhs) {
    return rhs + (-lhs);
}

template <typename T, typename V, std::size_t Vars>
requires std::is_convertible_v<T, V>
T& operator-=(T& lhs, val<V, Vars> const& rhs) {
    return lhs -= rhs.value();
}

template <typename T, typename V, std::size_t Vars>
requires std::is_convertible_v<T, V>
auto operator*(T const& lhs, val<V, Vars> const& rhs) {
    return rhs * lhs;
}

template <typename T, typename V, std::size_t Vars>
requires std::is_convertible_v<T, V>
auto operator/(T const& lhs, val<V, Vars> const& rhs) {
    val<V, Vars> ret { rhs };

    auto old_value = ret.data[0];
    ret.data[0] = lhs / ret.data[0];
    auto new_value = ret.data[0];

    auto this_it = ret.data.begin() + 1;

    while (this_it != ret.data.end()) {
        auto this_deriv = *this_it;

        *this_it = - (new_value /  old_value) * this_deriv;

        this_it++;
    }

    return ret;
}

// Functions
// TODO: Var versions
template <typename NumT, std::size_t Vars>
auto sin(val<NumT, Vars> const& v) {
    using std::sin;
    using std::cos;

    constexpr auto F = [](NumT const& x) { return sin(x); };
    constexpr auto D = [](NumT const& x) { return cos(x); };
    return v.template apply_function<F, D>();
}

template <typename NumT, std::size_t Vars>
auto exp(val<NumT, Vars> const& v) {
    using std::exp;
    using std::cos;

    constexpr auto F = [](NumT const& x) { return exp(x); };
    return v.template apply_function<F, F>();
}

template <typename NumT, std::size_t Vars>
auto log(val<NumT, Vars> const& v) {
    using std::log;

    constexpr auto F = [](NumT const& x) { return log(x); };
    constexpr auto D = [](NumT const& x) { return 1 / x; };
    return v.template apply_function<F, D>();
}

// Make functions
template <typename NumT>
auto make_single_var(NumT const& val) {
    return var<NumT, 1>(val, 0);
}

namespace details {

// make_vars helper.
template <typename NumT, typename... Args, std::size_t... Ids>
auto make_vars_impl(std::index_sequence<Ids...>, Args const&... args) {
    return std::make_tuple(var<NumT, sizeof...(Ids)> { args, Ids }...);
}

template <typename T>
using remove_const_ref = std::remove_const_t<std::remove_reference_t<T>>;

template <typename T>
struct identity {
    using type = T;
};

template <typename T>
struct val_extractor {};

template <typename NumT, std::size_t Vars>
struct val_extractor<autodiff::var<NumT, Vars>> {
    using type = autodiff::val<NumT, Vars>;
};

template <typename NumT, std::size_t Vars>
struct val_extractor<autodiff::val<NumT, Vars>> {
    using type = autodiff::val<NumT, Vars>;
};

} // End of namespace details.

template <typename NumT, typename... Args>
requires std::conjunction_v<std::is_convertible<Args, NumT>...>
auto make_vars(Args const&... args) {
    using Ids = std::make_index_sequence<sizeof...(Args)>;
    return details::make_vars_impl<NumT, Args...>(Ids{}, args...);
}

template <typename T>
using ValueType = std::conditional_t<
                      std::is_arithmetic_v<details::remove_const_ref<T>>,
                      details::identity<details::remove_const_ref<T>>,
                      details::val_extractor<details::remove_const_ref<T>>
                  >::type;

}

#endif
