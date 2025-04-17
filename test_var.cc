#include <cassert>
#include <iostream>
#include "ad.hh"

void test_addition() {
    Var x = 2.0;
    Var y = 3.0;
    Var z = x + y;
    z.backward();

    assert(z.value() == 5.0);
    assert(x.grad() == 1.0);
    assert(y.grad() == 1.0);
}

void test_subtraction() {
    Var x = 2.0;
    Var y = 3.0;
    Var z = x - y;
    z.backward();

    assert(z.value() == -1.0);
    assert(x.grad() == 1.0);
    assert(y.grad() == -1.0);
}

void test_multiplication() {
    Var x = 2.0;
    Var y = 3.0;
    Var z = x * y;
    z.backward();

    assert(z.value() == 6.0);
    assert(x.grad() == 3.0);
    assert(y.grad() == 2.0);
}

void test_division() {
    Var x = 6.0;
    Var y = 3.0;
    Var z = x / y;
    z.backward();

    assert(z.value() == 2.0);
    assert(x.grad() == 1.0 / 3.0);
    assert(y.grad() == -6.0 / 9.0);
}

void test_exponential() {
    Var x = 2.0;
    Var z = exp(x);
    z.backward();

    assert(z.value() == std::exp(2.0));
    assert(x.grad() == std::exp(2.0));
}

void test_logarithm() {
    Var x = 2.0;
    Var z = log(x);
    z.backward();

    assert(z.value() == std::log(2.0));
    assert(x.grad() == 1.0 / 2.0);
}

void test_loop() {
    Var x = 2.0;
    Var y = 3.0;
    Var z = 0.0;

    for (size_t i = 0; i < 10; ++i) {
        z =  x + z + y;
    }

    z.backward();

    assert(z.value() == 50.0);
    assert(x.grad() == 10.0);
    assert(y.grad() == 10.0);
}

void test_sqrt() {
    Var x = 4.0;
    Var z = sqrt(x);
    z.backward();

    assert(z.value() == 2.0);
    assert(x.grad() == 0.25);
}

void test_tape() {
    Var::clearTape();

    Var x = 2.0;
    Var y = 3.0;
    Var z = x + y;
    z.backward();

    assert(tapeOfVars.size() == 3);

    Var::clearTape();

    assert(tapeOfVars.size() == 0);
}

void test_complicated_sequence() {
    Var a,b,c;
    Var x = 4.0;
    Var y = 3.0;
    Var z = 2.0;

    a = x + y;
    b = a * z;
    c = b / x;
    a = c - y;

    a.backward();

    assert(a.value() == 0.5);
    assert(b.value() == 14.0);
    assert(c.value() == 3.5);
    assert(x.grad() == -0.375);
    assert(y.grad() == -0.5);
    assert(z.grad() == 1.75);
}


int main() {
    test_addition();
    test_subtraction();
    test_multiplication();
    test_division();
    test_exponential();
    test_logarithm();
    test_loop();
    test_sqrt();
    test_tape();
    test_complicated_sequence();

    std::cout << "All tests passed!" << std::endl;

    Var::clearTape();

    return 0;
}