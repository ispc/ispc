#pragma once

namespace ispc {

template <typename Base, typename Item>
class ConsumerRef final : public Base {
    Base &base;
  public:

    static std::unique_ptr<Base> make(Base &base) {
        return std::unique_ptr<Base>(new ConsumerRef(base));
    }

    ConsumerRef(Base &base_) : base(base_) {}

    void Consume(const Item &item) override {
        base.Consume(item);
    }
};

} // namespace ispc
