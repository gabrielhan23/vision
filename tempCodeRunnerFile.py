dn(LAYER_1, INPUT, )  # shape(128,784)
b1 = np.random.randn(LAYER_1)  # shape(128)

w2 = np.random.randn(LAYER_2, LAYER_1)  # shape(64, 128)
b2 = np.random.randn(LAYER_2)  # shape(64)


w3 = np.random.randn(OUTPUT, LAYER_2)  # shape (10, 64)
b3 = np.random.ran