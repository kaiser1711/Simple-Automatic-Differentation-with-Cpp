#ifndef AD_H
#define AD_H

#include <vector>
#include <cmath>
#include <memory>

// Forward declaration
class Var;
class VarData;  

// Persistent object
extern std::vector<std::shared_ptr<VarData>> tapeOfVars;

class VarData {
public:
    // Constructor
    VarData(double val, double gr, std::vector<std::shared_ptr<VarData>> ch, std::vector<double> der);

    // Backward pass to compute gradients
    void backward();

    // Destructor
    ~VarData();

    // Getters
    double& getValue() {
        return m_value;
    }

    double& getGrad() {
        return m_grad;
    }

    std::vector<std::shared_ptr<VarData>>& getChildren() {
        return m_children;
    }   

    std::vector<double>& getDerivatives() {
        return m_derivatives;
    }

    // Setters
    void setValue(double value) {
        m_value = value;
    }

    void setGrad(double grad) {
        m_grad = grad;
    }

    void setChildren(const std::vector<std::shared_ptr<VarData>>& children) {
        m_children = children;
    }

    void setDerivatives(const std::vector<double>& derivatives) {
        m_derivatives = derivatives;
    }

private:
    double m_value;
    double m_grad;
    std::vector<std::shared_ptr<VarData>> m_children;
    std::vector<double> m_derivatives;

    // Recursive backward pass
    void _backward() const;
};

// Var class represents a variable in the computational graph
class Var {
public:
    double& value() { return m_data->getValue(); }
    double& grad() { return m_data->getGrad(); }

    // Constructor
    Var(std::shared_ptr<VarData> data);
    Var(double val = 0.0, double gr = 0.0, std::vector<std::shared_ptr<VarData>> ch = {}, std::vector<double> der = {});

    // Destructor
    ~Var();

    // Copy constructor
    Var(const Var& other);

    static std::shared_ptr<VarData> createData(double val = 0.0, double gr = 0.0, std::vector<std::shared_ptr<VarData>> ch = {}, std::vector<double> der = {});
    static void clearTape();

    void clearObject();

    // Overload assignment operator
    Var& operator=(const Var& other);

    // Overload addition operator (Var + Var)
    Var operator+(const Var& other) const;

    // Overload addition operator (Var + double)
    Var operator+(double other) const;

    // Overload multiplication operator (Var * Var)
    Var operator*(const Var& other) const;

    // Overload multiplication operator (Var * double)
    Var operator*(double other) const;

    // Overload subtraction operator (Var - Var)
    Var operator-(const Var& other) const;

    // Overload subtraction operator (Var - double)
    Var operator-(double other) const;

    // Overload division operator (Var / Var)
    Var operator/(const Var& other) const;

    // Overload division operator (Var / double)
    Var operator/(double other) const;

    // Overload exponential function
    friend Var exp(const Var& var);

    // Overload logarithm function
    friend Var log(const Var& var);

    // Overload square root function
    friend Var sqrt(const Var& var);

    // Backward pass to compute gradients
    void backward();

    // Getters
    std::shared_ptr<VarData> getData() const {
        return m_data;
    }

    // Setters
    void setData(std::shared_ptr<VarData> data) {
        m_data = data;
    }

private:
    std::shared_ptr<VarData> m_data;
};

// Overload addition operator (double + Var)
inline Var operator+(double lhs, const Var& rhs) {
    return rhs + lhs;
}

// Overload subtraction operator (double - Var)
inline Var operator-(double lhs, const Var& rhs) {
    return Var(lhs) - rhs;
}

// Overload multiplication operator (double * Var)
inline Var operator*(double lhs, const Var& rhs) {
    return rhs * lhs;
}

// Overload division operator (double / Var)
inline Var operator/(double lhs, const Var& rhs) {
    return Var(lhs) / rhs;
}

#endif // AD_H