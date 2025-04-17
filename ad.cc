#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include "ad.hh"

// Persistent object
std::vector<std::shared_ptr<VarData>> tapeOfVars;

// VarData class implementation

VarData::VarData(double val, double gr, std::vector<std::shared_ptr<VarData>> ch, std::vector<double> der) 
    : m_value(val), m_grad(gr), m_children(ch), m_derivatives(der) {}

VarData::~VarData() {}

// Backward pass to compute gradients
void VarData::backward() {
    m_grad = 1.0; // Initialize gradient of the root node
    _backward();
}

// Recursive backward pass
void VarData::_backward() const {
    for (size_t i = 0; i < m_children.size(); ++i) {
        m_children[i]->getGrad() += m_derivatives[i] * m_grad;
        m_children[i]->_backward();
    }
}

// Clear the tape of variables
void Var::clearTape() {
    tapeOfVars.clear();
}

// Var class implementation

// Constructor 
Var::Var(double val, double gr, std::vector<std::shared_ptr<VarData>> ch, std::vector<double> der) {
    m_data = Var::createData(val, gr, ch, der);
    tapeOfVars.push_back(m_data);
}

Var::Var(std::shared_ptr<VarData> cdata) : m_data(cdata) {}

// Copy constructor
Var::Var(const Var& other) : m_data(other.getData()) {}

Var::~Var() {}

// Factory method to create a Var object on the heap
std::shared_ptr<VarData> Var::createData(double val, double gr, std::vector<std::shared_ptr<VarData>> ch, std::vector<double> der) {
    return std::make_shared<VarData>(val, gr, ch, der);
}

// Overload assignment operator
Var& Var::operator=(const Var& other) {
    if (this != &other) {
        m_data = other.getData();
    }
    return *this;
}

// Helper function to create a new Var object
Var createVar(double value, const std::vector<std::shared_ptr<VarData>>& children, const std::vector<double>& derivatives) {
    Var result = Var(value);
    for (const auto& child : children) {
        result.getData()->getChildren().push_back(child);
    }
    result.getData()->setDerivatives(derivatives);
    return result;
}

// Overload addition operator (Var + Var)
Var Var::operator+(const Var& other) const {
    return createVar(m_data->getValue() + other.getData()->getValue(), {m_data, other.getData()}, {1.0, 1.0});
}

// Overload addition operator (Var + double)
Var Var::operator+(double other) const {
    return createVar(m_data->getValue() + other, {m_data}, {1.0});
}

// Overload multiplication operator (Var * Var)
Var Var::operator*(const Var& other) const {
    return createVar(m_data->getValue() * other.getData()->getValue(), {m_data, other.getData()}, {other.getData()->getValue(), m_data->getValue()});
}

// Overload multiplication operator (Var * double)
Var Var::operator*(double other) const {
    return createVar(m_data->getValue() * other, {m_data}, {other});
}

// Overload subtraction operator (Var - Var)
Var Var::operator-(const Var& other) const {
    return createVar(m_data->getValue() - other.getData()->getValue(), {m_data, other.getData()}, {1.0, -1.0});
}

// Overload subtraction operator (Var - double)
Var Var::operator-(double other) const {
    return createVar(m_data->getValue() - other, {m_data}, {1.0});
}

// Overload division operator (Var / Var)
Var Var::operator/(const Var& other) const {
    if (other.getData()->getValue() == 0) {
        throw std::runtime_error("Division by zero");
    }
    return createVar(m_data->getValue() / other.getData()->getValue(), {m_data, other.getData()}, {1.0 / other.getData()->getValue(), -m_data->getValue() / (other.getData()->getValue() * other.getData()->getValue())});
}

// Overload division operator (Var / double)
Var Var::operator/(double other) const {
    if (other == 0) {
        throw std::runtime_error("Division by zero");
    }
    return createVar(m_data->getValue() / other, {m_data}, {1.0 / other});
}

// Overload exponential function
Var exp(const Var& var) {
    return createVar(std::exp(var.getData()->getValue()), {var.getData()}, {std::exp(var.getData()->getValue())});
}

// Overload logarithm function
Var log(const Var& var) {
    return createVar(std::log(var.getData()->getValue()), {var.getData()}, {1.0 / var.getData()->getValue()});
}

// Overload square root function
Var sqrt(const Var& var) {
    return createVar(std::sqrt(var.getData()->getValue()), {var.getData()}, {1.0 / (2.0 * std::sqrt(var.getData()->getValue()))});
}

// Backward pass to compute gradients
void Var::backward() {
    m_data->backward();
}