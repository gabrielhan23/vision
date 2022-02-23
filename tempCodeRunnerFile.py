(len(input)*len(input[0]))

    
    z1 = np.dot(w1,a0) - b1
    a1 = [sigmoid(x) for x in z1]

    z2 = np.dot(w2,a1) - b2
    a2 = [sigmoid(x) for x in z2]

    z3 = np.dot(w3,a2)