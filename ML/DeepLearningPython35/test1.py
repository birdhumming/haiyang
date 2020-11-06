"""
    Testing code for different neural network configurations.
    Adapted for Python 3.5.2

    Usage in shell:
        python3.5 test.py

    Network (network.py and network2.py) parameters:
        2nd param is epochs count
        3rd param is batch size
        4th param is learning rate (eta)

    Author:
        Michał Dobrzański, 2016
        dobrzanski.michal.daniel@gmail.com
"""

# ----------------------
# - read the input data:
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
# ---------------------
# - network.py example:
import network

# exercise solution for only 2 layers of networks - 

net = network.Network([784, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

