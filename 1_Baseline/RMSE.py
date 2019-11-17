
def RMSE(mse,test):
    """"calculate RMSE"""
    return np.sqrt(1.0 * mse / test.nnz)

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)