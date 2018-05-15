import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    train_num = X.shape[0]
    class_num = W.shape[1]
    scores = X.dot(W)
    for i in range(0, train_num):
        row = scores[i, :]
        row -= np.max(row)
        exp_row = np.exp(row)
        # print(row.shape)
        poss_row = exp_row/np.sum(exp_row)
        true_class = y[i]
        loss -= np.log(poss_row[true_class])
        dW[:, true_class] -= X[i]
        for j in range(class_num):
            dW[:, j] += poss_row[j]*X[i]

    loss /= train_num
    dW /= train_num
    loss += np.sum(W*W)
    dW += 2 * reg * W*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    train_num = X.shape[0]
    class_num = W.shape[1]
    dimension = X.shape[1]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)		# To keep the result as column vector rather than row vector
    exp_scores = np.exp(scores)
    poss_scores = exp_scores/np.reshape(np.sum(exp_scores, axis=1), (-1, 1))
    loss -= np.mean(np.log(exp_scores[np.arange(train_num), y]/np.sum(exp_scores, axis=1)))
    loss += np.sum(W*W)

    # The following three lines follow the instructions :
    # http://cs231n.github.io/neural-networks-case-study/#grad
    dscores = poss_scores
    dscores[np.arange(train_num), y] -= 1
    dscores /= train_num
    dW += np.dot(X.T, dscores)
    dW += 2*reg*W*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
