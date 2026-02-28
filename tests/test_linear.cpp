#include <iostream>
#include "layers/linear.h"

int main() {
    Linear lin(4, 3);

    Tensor x(2, 4);
    for (int i = 0; i < x.rows; i++)
        for (int j = 0; j < x.cols; j++)
            x(i,j) = 0.1f * (i + j);

    Tensor y = lin.forward(x);

    std::cout << "Linear output:\n";
    for (int i = 0; i < y.rows; i++) {
        for (int j = 0; j < y.cols; j++)
            std::cout << y(i,j) << " ";
        std::cout << "\n";
    }
}