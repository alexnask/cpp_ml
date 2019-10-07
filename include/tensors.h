#ifndef __TENSORS_H
#define __TENSORS_H

#include <utility>
#include <array>
#include <type_traits>
#include <iterator>
#include <cstddef>

#include <ranges.h>

// TODO: Remove self types, just use the class name...
// TODO: Use view_interface to reduce boilerplate.

namespace details {
    template <std::size_t... Sizes>
    constexpr std::size_t tensor_data_size() {
        constexpr std::size_t arr[] { Sizes... };

        std::size_t total = 1;
        for (std::size_t i = 0; i < sizeof...(Sizes); i++) {
            total *= arr[i];
        }

        return total;
    }

    // Extract an element from an index sequence.
    template<std::size_t... Ints>
    constexpr std::size_t get(std::index_sequence<Ints...>, [[maybe_unused]] std::size_t i) {
        if constexpr (sizeof...(Ints) == 0) return 0;
        else {
            constexpr std::size_t arr[] { Ints... };
            return arr[i];
        }
    }

    template <std::size_t... Left, std::size_t... Right>
    constexpr bool compatible_tensor_sizes(std::index_sequence<Left...>, std::index_sequence<Right...>) {
        constexpr auto DimL = sizeof...(Left);
        constexpr auto DimR = sizeof...(Right);

        constexpr bool left_smaller = DimL < DimR;
    
        std::size_t left_sizes[DimL] { Left... };
        std::size_t right_sizes[DimR] { Right... };

        std::size_t min_dim = left_smaller ? DimL : DimR;
        std::size_t max_dim = left_smaller ? DimR : DimL;

        for (std::size_t i = 0; i < min_dim; i++) {
            if (left_sizes[i] != right_sizes[i]) return false;
        }

        for (std::size_t i = min_dim; i < max_dim; i++) {
            if (left_smaller  && right_sizes[i] != 1) return false;
            if (!left_smaller && left_sizes[i] != 1) return false;
        }

        return true;
    }

    // Modified compatible_tensor that returns different types instead of values for use with concepts.
    template <std::size_t... Left, std::size_t... Right>
    constexpr auto compatible_tensor_mod(std::index_sequence<Left...> l, std::index_sequence<Right...> r) {
        if constexpr (compatible_tensor_sizes(l, r)) return std::true_type{};
        else return std::false_type{};
    }

    static_assert(!compatible_tensor_sizes(std::index_sequence<1, 2, 3>{}, std::index_sequence<1>{}));
    static_assert(compatible_tensor_sizes(std::index_sequence<1, 2, 3>{}, std::index_sequence<1, 2, 3>{}));
    static_assert(compatible_tensor_sizes(std::index_sequence<5, 7, 10>{}, std::index_sequence<5, 7, 10, 1, 1, 1>{}));
    static_assert(compatible_tensor_sizes(std::index_sequence<3, 2, 1, 1, 1, 1>{}, std::index_sequence<3, 2>{}));

    template <std::size_t... Sizes>
    constexpr bool is_valid_tensor_size() {
        constexpr auto Dimensions = sizeof...(Sizes);
        if constexpr (Dimensions == 0) return false;

        std::size_t sizes[Dimensions] { Sizes... };
        for (std::size_t i = 0; i < Dimensions; i++) {
            if (sizes[i] == 0) return false;
        }

        return true;
    }
}

template <typename NumT, std::size_t... Sizes>
class Tensor;

template <class T, class U>
concept Same = std::is_same_v<T, U>;

template <class T, class U>
concept Convertible = std::is_convertible_v<T, U>;

// If the tensor is one dimensional and of size 1, it can be equivalent to a single NumT.
template <typename T, typename NumT, std::size_t... Sizes>
concept SingleObjectTensor = sizeof...(Sizes) == 1 && details::get(std::index_sequence<Sizes...>{}, 0) == 1 && std::is_same_v<T, NumT>;

// If the tensor is one dimensional it can be equivalent to an array or std::array of the correct size.
template <typename T, typename NumT, std::size_t... Sizes>
concept ArrayTensor = sizeof...(Sizes) == 1 && details::get(std::index_sequence<Sizes...>{}, 0) > 0 &&
                      (std::is_same_v<T, NumT[details::get(std::index_sequence<Sizes...>{}, 0)]> ||
                      std::is_same_v<T, std::array<NumT, details::get(std::index_sequence<Sizes...>{}, 0)>>);

template <typename T, typename NumT, std::size_t... Sizes>
concept TensorView = sizeof...(Sizes) == 1 && requires { typename T::TensorType; } && std::is_same_v<typename T::TensorType::ValueType, NumT> &&
                     T::size() == details::get(std::index_sequence<Sizes...>{}, 0);

template <typename T, typename NumT, std::size_t... Sizes>
concept TensorLike =
        SingleObjectTensor<T, NumT, Sizes...> ||
        ArrayTensor<T, NumT, Sizes...> ||
        TensorView<T, NumT, Sizes...> ||
        // Otherwise, the tensor is only equivalent to other tensors with compatible tensor sizes.
        requires {
            typename T::SizeSeq;
            { details::compatible_tensor_mod(std::index_sequence<Sizes...>{}, typename T::SizeSeq{}) } -> Same<std::true_type>;
        };

template <typename T, typename NumT>
concept IsStaticCastable = requires(NumT t) {
    static_cast<T>(t);
};

template <typename ToT, typename NumT, std::size_t... Sizes>
requires IsStaticCastable<ToT, NumT>
auto tensor_cast(Tensor<NumT, Sizes...> const& tensor) {
    Tensor<ToT, Sizes...> res;

    std::ranges::copy(
        tensor | std::views::transform([](NumT const& in) { return static_cast<ToT>(in); }),
        res.begin()
    );

    return res;
}

// TODO: Add data, cdata functions (contiguous range).
// TODO: Add operators with other type of tensors and other types of tensor views (see operator*).
template <typename NumT, std::size_t... Sizes>
struct Tensor {
    static constexpr auto Dimensions = sizeof...(Sizes);
    using SizeSeq = std::index_sequence<Sizes...>;
    using ValueType = NumT;

private:
    static_assert(details::is_valid_tensor_size<Sizes...>(), "Invalid tensor size.");

    using Self = Tensor<NumT, Sizes...>;
    static constexpr auto DataSize = details::tensor_data_size<Sizes...>();

    std::array<NumT, DataSize> data;

    template <bool IsConst>
    class iterator_impl {
        using data_ptr_type = std::conditional_t<IsConst, NumT const*, NumT*>;
        using SelfIt = iterator_impl<IsConst>;

        data_ptr_type data_ptr;
        std::size_t data_idx;

        iterator_impl() = default;
        iterator_impl(SelfIt const&) = default;
        iterator_impl(SelfIt&) = default;
        iterator_impl(SelfIt&&) = default;

        SelfIt& operator=(SelfIt const&) = default;
        SelfIt& operator=(SelfIt&) = default;
        SelfIt& operator=(SelfIt&&) = default;

        ~iterator_impl() = default;

        iterator_impl(data_ptr_type _data, std::size_t idx) : data_ptr(_data), data_idx(idx) {}

    public:
        using difference_type = std::ptrdiff_t;
        using value_type = NumT;
        using pointer = std::conditional_t<IsConst, NumT const*, NumT*>;
        using reference = std::conditional_t<IsConst, NumT const&, NumT&>;
        using iterator_category = std::random_access_iterator_tag;

        auto index() const {
            std::array<std::size_t, Dimensions> res_arr;
            std::ranges::fill(res_arr, 0);

            std::size_t curr = data_idx;
            std::size_t prev_size = 1;

            for (std::size_t i = 0; i < Dimensions; i++) {
                res_arr[i] = (curr / prev_size) % details::get(SizeSeq{}, i);
                std::size_t to_rem = res_arr[i] * prev_size;
                curr -= to_rem;
                prev_size *= details::get(SizeSeq{}, i);
            }

            return res_arr;
        }

        auto& operator+=(difference_type m) {
            data_idx += m;
            return *this;
        }

        auto operator+(difference_type m) const {
            auto copy = *this;
            return copy += m;
        }

        auto& operator-=(difference_type m) {
            return *this += (-m);
        }

        auto operator-(difference_type m) const {
            auto copy = *this;
            return copy += (-m);
        }

        difference_type operator-(SelfIt const& rhs) const {
            return data_idx - rhs.data_idx;
        }

        reference operator[](difference_type n) const {
            assert(data_idx + n < DataSize);
            return *(*this + n);
        }

        reference operator*() const {
            assert(data_idx < DataSize);
            return data_ptr[data_idx];
        }

        pointer operator->() const {
            assert(data_idx < DataSize);
            return &data_ptr[data_idx];
        }

        auto& operator++() {
            data_idx++;
            return *this;
        }

        auto operator++(int) {
            auto copy = *this;
            data_idx++;
            return copy;
        }

        auto& operator--() {
            data_idx--;
            return *this;
        }

        auto operator--(int) {
            auto copy = *this;
            data_idx--;
            return copy;
        }
    
        bool operator<(SelfIt const& rhs) const {
            return data_idx < rhs.data_idx;
        }

        bool operator<=(SelfIt const& rhs) const {
            return data_idx <= rhs.data_idx;
        }

        bool operator>(SelfIt const& rhs) const {
            return data_idx > rhs.data_idx;
        }

        bool operator>=(SelfIt const& rhs) const {
            return data_idx >= rhs.data_idx;
        }

        bool operator==(SelfIt const& rhs) const {
            return data_idx == rhs.data_idx;
        }
    
        bool operator!=(SelfIt const& rhs) const {
            return data_idx != rhs.data_idx;
        }
    };

    // Tensor view, fixes values for all but 1 axis.
    template<std::size_t VariableIndex, bool IsConst>
    class view_impl {
        static_assert(VariableIndex < Dimensions, "Invalid TensorView variable dimension.");
        using SelfView = view_impl<VariableIndex, IsConst>;

        class view_iterator {
            SelfView const* view_ptr;
            std::size_t idx;

            view_iterator() = default;
            view_iterator(view_iterator const&) = default;
            view_iterator(view_iterator&) = default;
            view_iterator(view_iterator&&) = default;

            view_iterator& operator=(view_iterator const&) = default;
            view_iterator& operator=(view_iterator&) = default;
            view_iterator& operator=(view_iterator&&) = default;

            ~view_iterator() = default;

            view_iterator(SelfView const* _view_ptr, std::size_t _idx) : view_ptr(_view_ptr), idx(_idx) {}

        public:
            using difference_type = std::ptrdiff_t;
            using value_type = NumT;
            using pointer = std::conditional_t<IsConst, NumT const*, NumT*>;
            using reference = std::conditional_t<IsConst, NumT const&, NumT&>;
            using iterator_category = std::random_access_iterator_tag;

            auto& operator+=(difference_type m) {
                idx += m;
                return *this;
            }

            auto operator+(difference_type m) const {
                auto copy = *this;
                return copy += m;
            }

            auto& operator-=(difference_type m) {
                return *this += (-m);
            }

            auto operator-(difference_type m) const {
                auto copy = *this;
                return copy += (-m);
            }

            difference_type operator-(view_iterator const& rhs) const {
                return idx - rhs.idx;
            }

            reference operator[](difference_type n) const {
                return *(*this + n);
            }

            reference operator*() const {
                return view_ptr->get(idx);
            }

            pointer operator->() const {
                return &view_ptr->get(idx);
            }

            auto& operator++() {
                idx++;
                return *this;
            }

            auto operator++(int) {
                auto copy = *this;
                idx++;
                return copy;
            }

            auto& operator--() {
                idx--;
                return *this;
            }

            auto operator--(int) {
                auto copy = *this;
                idx--;
                return copy;
            }
        
            bool operator<(view_iterator const& rhs) const {
                return idx < rhs.idx;
            }

            bool operator<=(view_iterator const& rhs) const {
                return idx <= rhs.idx;
            }

            bool operator>(view_iterator const& rhs) const {
                return idx > rhs.idx;
            }

            bool operator>=(view_iterator const& rhs) const {
                return idx >= rhs.idx;
            }

            bool operator==(view_iterator const& rhs) const {
                return idx == rhs.idx;
            }
        
            bool operator!=(view_iterator const& rhs) const {
                return idx != rhs.idx;
            }
        };

        using data_ptr_type = std::conditional_t<IsConst, NumT const*, NumT*>;
        using reference = std::conditional_t<IsConst, NumT const&, NumT&>;

        static constexpr auto VariableDimSize = details::get(SizeSeq{}, VariableIndex);

        data_ptr_type data_ptr;
        std::array<std::size_t, Dimensions-1> stable_indices;

        view_impl(data_ptr_type data, std::array<std::size_t, Dimensions-1> _stable_indices) : data_ptr(data), stable_indices(_stable_indices) {
            // Bounds checking on the values for the rest of the dimensions.
            std::size_t dim_idx = 0;
            for (std::size_t i = 0; i < Dimensions; i++) {
                if (i == VariableIndex) continue;
                assert (stable_indices[dim_idx] < details::get(SizeSeq{}, i));
                dim_idx++;
            }
        }

    public:
        using TensorType = Self;

        static constexpr std::size_t size() {
            return VariableDimSize;
        }

        reference get(std::size_t index) const {
            assert(index < size());

            std::size_t final_index = 0;
            std::size_t accum = 1;

            std::size_t dim_idx = 0;
            for (std::size_t i = 0; i < Dimensions; i++) {
                if (i == VariableIndex) {
                    final_index += index * accum;
                }
                else {
                    final_index += stable_indices[dim_idx] * accum;
                    dim_idx++;
                }

                accum *= details::get(SizeSeq{}, i);
            }

            return data_ptr[final_index];
        }

        // Add another TensorLike compatible object.
        SelfView& operator+=(TensorLike<NumT, VariableDimSize> const& rhs) {
            auto it = std::ranges::begin(rhs);
            for (std::size_t i = 0; i < VariableDimSize; i++) {
                get(i) += *it;
                it++;
            }
            assert(it == std::ranges::end(rhs));
            
            return *this;
        }

        Tensor<NumT, VariableDimSize> operator+(TensorLike<NumT, VariableDimSize> const& rhs) const {
            auto this_it = begin();
            auto rhs_it = std::ranges::begin(rhs);

            Tensor<NumT, VariableDimSize> ret;

            for (auto& elem : ret) {
                elem = *this_it + *rhs_it;
                this_it++;
                rhs_it++;
            }

            assert(this_it == end() && rhs_it == std::ranges::end(rhs));

            return ret;
        }

        // Add a constant to all elements of the tensor.
        SelfView& operator+=(NumT const& rhs) {
            for (std::size_t i = 0; i < VariableDimSize; i++) {
                get(i) += rhs;
            }

            return *this;
        }

        Tensor<NumT, VariableDimSize> operator+(NumT const& rhs) const {
            Tensor<NumT, VariableDimSize> ret;
            std::ranges::copy(*this | std::views::transform([rhs] (NumT const& x) { return x + rhs; }), ret.begin());
            return ret;
        }

        // Multiply another TensorLike compatible object.
        SelfView& operator*=(TensorLike<NumT, VariableDimSize> const& rhs) {
            auto it = std::ranges::begin(rhs);
            for (std::size_t i = 0; i < VariableDimSize; i++) {
                get(i) *= *it;
                it++;
            }
            assert(it == std::ranges::end(rhs));
            
            return *this;
        }

        template <typename ViewT>
        requires requires { typename ViewT::TensorType; } && ViewT::size() == VariableDimSize &&
        requires(NumT nt, typename ViewT::TensorType::ValueType ot) { nt *= ot; }
        SelfView& operator*=(ViewT const& rhs) {
            auto it = std::ranges::begin(rhs);
            for (auto& elem : *this) {
                elem *= *it;
                it++;
            }
            assert(it == std::ranges::end(rhs));
            
            return *this;
        }

        Tensor<NumT, VariableDimSize> operator*(TensorLike<NumT, VariableDimSize> const& rhs) const {
            auto this_it = begin();
            auto rhs_it = std::ranges::begin(rhs);

            Tensor<NumT, VariableDimSize> ret;

            for (auto& elem : ret) {
                elem = *this_it * *rhs_it;
                this_it++;
                rhs_it++;
            }

            assert(this_it == end() && rhs_it == std::ranges::end(rhs));

            return ret;
        }

        // Multiply a constant to all elements of the tensor.
        SelfView& operator*=(NumT const& rhs) {
            for (std::size_t i = 0; i < VariableDimSize; i++) {
                get(i) *= rhs;
            }

            return *this;
        }

        Tensor<NumT, VariableDimSize> operator*(NumT const& rhs) const {
            Tensor<NumT, VariableDimSize> ret;
            std::ranges::copy(*this | std::views::transform([rhs] (NumT const& x) { return x * rhs; }), ret.begin());
            return ret;
        }

        // Divide another TensorLike compatible object.
        SelfView& operator/=(TensorLike<NumT, VariableDimSize> const& rhs) {
            auto it = std::ranges::begin(rhs);
            for (std::size_t i = 0; i < VariableDimSize; i++) {
                get(i) /= *it;
                it++;
            }
            assert(it == std::ranges::end(rhs));
            
            return *this;
        }

        Tensor<NumT, VariableDimSize> operator/(TensorLike<NumT, VariableDimSize> const& rhs) const {
            auto this_it = begin();
            auto rhs_it = std::ranges::begin(rhs);

            Tensor<NumT, VariableDimSize> ret;

            for (auto& elem : ret) {
                elem = *this_it / *rhs_it;
                this_it++;
                rhs_it++;
            }

            assert(this_it == end() && rhs_it == std::ranges::end(rhs));

            return ret;
        }

        // Multiply a constant to all elements of the tensor.
        SelfView& operator/=(NumT const& rhs) {
            for (std::size_t i = 0; i < VariableDimSize; i++) {
                get(i) /= rhs;
            }

            return *this;
        }

        Tensor<NumT, VariableDimSize> operator/(NumT const& rhs) const {
            Tensor<NumT, VariableDimSize> ret;
            std::ranges::copy(*this | std::views::transform([rhs] (NumT const& x) { return x / rhs; }), ret.begin());
            return ret;
        }

        reference operator[](std::size_t index) const {
            return get(index);
        }

        view_iterator begin() const {
            return { this, 0 };
        }

        view_iterator end() const {
            return { this, VariableDimSize };
        }
    };
public:
    Tensor() = default;

    using iterator = iterator_impl<false>;
    using const_iterator = iterator_impl<true>;

    // Default value constructor.
    Tensor(NumT const& val) : data() {
        std::ranges::fill(data, val);
    }

    // Add another TensorLike compatible object.
    Self& operator+=(TensorLike<NumT, Sizes...> const& rhs) {
        auto it = std::ranges::begin(rhs);
        for (auto& elem : data) {
            elem += *it;
            it++;
        }
        assert(it == std::ranges::end(rhs));
        
        return *this;
    }

    template <typename OtherT, std::size_t... OtherSizes>
    requires details::compatible_tensor_sizes(SizeSeq {}, std::index_sequence<OtherSizes...>{}) &&
    requires(NumT nt, OtherT ot) { nt += ot; }
    Self& operator+=(Tensor<OtherT, OtherSizes...> const& rhs) {
        auto it = std::ranges::begin(rhs);
        for (auto& elem : data) {
            elem += *it;
            it++;
        }
        assert(it == std::ranges::end(rhs));
        
        return *this;
    }

    Self operator+(TensorLike<NumT, Sizes...> const& rhs) const {
        auto res = *this;
        res += rhs;
        return res;
    }

    // Add a constant to all elements of the tensor.
    Self& operator+=(NumT const& rhs) {
        for (auto& elem: data) {
            elem += rhs;
        }

        return *this;
    }

    Self operator+(NumT const& rhs) const {
        auto res = *this;
        res += rhs;
        return res;
    }

    // Multiply another TensorLike compatible object.
    Self& operator*=(TensorLike<NumT, Sizes...> const& rhs) {
        auto it = std::ranges::begin(rhs);
        for (auto& elem : data) {
            elem *= *it;
            it++;
        }
        assert(it == std::ranges::end(rhs));
        
        return *this;
    }

    template <typename OtherT, std::size_t... OtherSizes>
    requires details::compatible_tensor_sizes(SizeSeq {}, std::index_sequence<OtherSizes...>{}) &&
    requires(NumT nt, OtherT ot) { nt *= ot; }
    Self& operator*=(Tensor<OtherT, OtherSizes...> const& rhs) {
        auto it = std::ranges::begin(rhs);
        for (auto& elem : data) {
            elem *= *it;
            it++;
        }
        assert(it == std::ranges::end(rhs));
        
        return *this;
    }

    template <typename ViewT>
    requires Dimensions == 1 && requires { typename ViewT::TensorType; } && ViewT::size() == details::get(SizeSeq{}, 0) &&
    requires(NumT nt, typename ViewT::TensorType::ValueType ot) { nt *= ot; }
    Self& operator*=(ViewT const& rhs) {
        auto it = std::ranges::begin(rhs);
        for (auto& elem : data) {
            elem *= *it;
            it++;
        }
        assert(it == std::ranges::end(rhs));
        
        return *this;
    }

    Self operator*(TensorLike<NumT, Sizes...> const& rhs) const {
        auto res = *this;
        res *= rhs;
        return res;
    }

    template <typename ViewT>
    requires Dimensions == 1 && requires { typename ViewT::TensorType; } && ViewT::size() == details::get(SizeSeq{}, 0) &&
    requires(NumT nt, typename ViewT::TensorType::ValueType ot) { nt *= ot; }
    Self operator*(ViewT const& rhs) const {
        auto res = *this;
        res *= rhs;
        return res;
    }

//     template <typename T, typename NumT, std::size_t... Sizes>
// concept TensorView = sizeof...(Sizes) == 1 && requires { typename T::TensorType; } && std::is_same_v<typename T::TensorType::ValueType, NumT> &&
//                      T::size() == details::get(std::index_sequence<Sizes...>{}, 0);

    // Multiply a constant to all elements of the tensor.
    Self& operator*=(NumT const& rhs) {
        for (auto& elem: data) {
            elem *= rhs;
        }

        return *this;
    }

    Self operator*(NumT const& rhs) const {
        auto res = *this;
        res *= rhs;
        return res;
    }

    // Divide another TensorLike compatible object.
    Self& operator/=(TensorLike<NumT, Sizes...> const& rhs) {
        auto it = std::ranges::begin(rhs);
        for (auto& elem : data) {
            elem /= *it;
            it++;
        }
        assert(it == std::ranges::end(rhs));
        
        return *this;
    }

    Self operator/(TensorLike<NumT, Sizes...> const& rhs) const {
        auto res = *this;
        res /= rhs;
        return res;
    }

    // Multiply a constant to all elements of the tensor.
    Self& operator/=(NumT const& rhs) {
        for (auto& elem: data) {
            elem /= rhs;
        }

        return *this;
    }

    Self operator/(NumT const& rhs) const {
        auto res = *this;
        res /= rhs;
        return res;
    }

    NumT const& get(std::array<std::size_t, Dimensions> const& index) const {
        std::size_t final_index = 0;
        // Accumulates the size that must be skipped per index for the next column of 'index'. 
        std::size_t accum = 1;

        for (std::size_t i = 0; i < Dimensions; i++) {
            final_index += index[i] * accum;
            accum *= details::get(SizeSeq{}, i);
        }

        assert(final_index < DataSize);
        return data[final_index];
    }

    NumT& get(std::array<std::size_t, Dimensions> const& index) {
        return const_cast<NumT&>(const_cast<Self const*>(this)->get(index));
    }

    NumT const& operator[](std::array<std::size_t, Dimensions> const& index) const {
        return get(index);
    }

    NumT& operator[](std::array<std::size_t, Dimensions> const& index) {
        return get(index);
    }

    iterator begin() {
        return { &data[0], 0 };
    }

    const_iterator begin() const {
        return { &data[0], 0 };
    }

    iterator end() {
        return { &data[0], DataSize };
    }

    const_iterator end() const {
        return { &data[0], DataSize };
    }

    const_iterator cbegin() const {
        return { &data[0], 0 };
    }

    const_iterator cend() const {
        return { &data[0], DataSize };
    }

    template <std::size_t VariableIndex>
    using view_t = view_impl<VariableIndex, false>;

    template <std::size_t VariableIndex>
    using const_view_t = view_impl<VariableIndex, true>;

    template <std::size_t VariableIndex>
    view_t<VariableIndex> view(std::array<std::size_t, Dimensions-1> const& stable_indices) {
        return { &data[0], stable_indices };
    }

    template <std::size_t VariableIndex>
    const_view_t<VariableIndex> view(std::array<std::size_t, Dimensions-1> const& stable_indices) const {
        return { &data[0], stable_indices };
    }

    template <std::size_t VariableIndex>
    const_view_t<VariableIndex> cview(std::array<std::size_t, Dimensions-1> const& stable_indices) const {
        return { &data[0], stable_indices };
    }
};

// Reverse order numeric operators

// Tensor numeric operations.

// Add number and tensor.
template <typename NumT, std::size_t... Sizes>
Tensor<NumT, Sizes...> operator+(NumT const& lhs, Tensor<NumT, Sizes...> const& rhs) {
    Tensor<NumT, Sizes...> ret;
    std::ranges::copy(rhs | std::views::transform([lhs] (NumT const& x) { return x + lhs; }), ret.begin());
    return ret;
}

// Add number and tensor view.
template <typename NumT, typename T>
auto operator+(NumT const& lhs, T const& rhs) requires requires { typename T::TensorType; } && std::is_same_v<typename T::TensorType::ValueType, NumT> {
    Tensor<NumT, T::size()> ret;
    std::ranges::copy(rhs | std::views::transform([lhs] (NumT const& x) { return x + lhs; }), ret.begin());
    return ret;
}

// Multiply number and tensor.
template <typename NumT, std::size_t... Sizes>
Tensor<NumT, Sizes...> operator*(NumT const& lhs, Tensor<NumT, Sizes...> const& rhs) {
    Tensor<NumT, Sizes...> ret;
    std::ranges::copy(rhs | std::views::transform([lhs] (NumT const& x) { return x * lhs; }), ret.begin());
    return ret;
}

// Multiply number and tensor view.
template <typename NumT, typename T>
auto operator*(NumT const& lhs, T const& rhs) requires requires { typename T::TensorType; } && std::is_same_v<typename T::TensorType::ValueType, NumT> {
    Tensor<NumT, T::size()> ret;
    std::ranges::copy(rhs | std::views::transform([lhs] (NumT const& x) { return x * lhs; }), ret.begin());
    return ret;
}

template <bool IsConst, typename NumT, std::size_t... Sizes>
typename Tensor<NumT, Sizes...>::template iterator_impl<IsConst> operator+(std::ptrdiff_t m, typename Tensor<NumT, Sizes...>::template iterator_impl<IsConst> const& iter) {
    return iter + m;
}

template <std::size_t Index, bool IsConst, typename NumT, std::size_t... Sizes>
typename Tensor<NumT, Sizes...>::template view_impl<Index, IsConst>::view_iterator operator+(std::ptrdiff_t m, typename Tensor<NumT, Sizes...>::template view_impl<Index, IsConst>::view_iterator const& iter) {
    return iter + m;
}

static_assert(TensorLike<int, int, 1>);
static_assert(TensorLike<double[2], double, 2>);
static_assert(!TensorLike<bool[5], double, 7, 1>);
static_assert(TensorLike<std::array<bool, 1>, bool, 1>);
static_assert(TensorLike<Tensor<long long int, 1, 2, 3>, long long int, 1, 2, 3>);
static_assert(TensorLike<Tensor<std::size_t, 7, 3>, std::size_t, 7, 3, 1, 1>);

#endif
