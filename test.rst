
Parameter estimates for correlated regressors
=============================================

By JB Poline and Matthew Brett.

.. nbplot::

    >>> # Compatibility with Python 3
    >>> from __future__ import print_function  # print('me') instead of print 'me'
    >>> from __future__ import division  # 1/2 == 0.5, not 0

.. nbplot::

    >>> # Array and plotting libraries
    >>> import numpy as np
    >>> import numpy.linalg as npl
    >>> import matplotlib.pyplot as plt

.. mpl-interactive::

.. nbplot::

    >>> # Display plots inside the notebook

.. nbplot::

    >>> # Make numpy print 4 significant digits for prettiness
    >>> np.set_printoptions(precision=4, suppress=True)

.. nbplot::

    >>> # Set random number seed to make random numbers reproducible
    >>> np.random.seed(42)

The linear model again
----------------------

We have some vector of data :math:`\vec{y}` with values
:math:`y_1, y_2, ..., y_N`.

We have one or more vectors of regressors
:math:`\vec{x_1}, \vec{x_2}, ..., \vec{x_P}`, where :math:`\vec{x_1}`
has values :math:`x_{1,1}, x_{1,2}, ..., x_{1,N}`, and :math:`\vec{x_p}`
has values :math:`x_{p,1}, x_{p,2}, ..., x_{p,N}`.

Our linear model says that:

.. math::


   \vec{y} = \beta_1 \vec{x_1} + \beta_2 \vec{x_2} +  ... + \beta_P \vec{x_P} + \vec{\varepsilon}

Here:

-  :math:`\beta_1, \beta_2, ... \beta_P` are scaling coefficients for
   vectors :math:`\vec{x_1}, \vec{x_2}, ..., \vec{x_P}` respectively;
-  :math:`\vec{\varepsilon} = \varepsilon_1, \varepsilon_2, ... \varepsilon_N`
   are the remaining unexplained errors for each observation.

Usually one of vectors :math:`\vec{x}` is a vector of constant value 1.
This models the intercept of the regression model. We will write this
special vector as :math:`\vec{1}`.

As we saw in the `introduction to the general linear
model <http://perrin.dynevor.org/glm_intro.html>`__, we can express this
same linear model as matrices. We:

-  assemble the :math:`\vec{x_p}` vectors as columns in a design matrix
   :math:`\mathbf{X} = [\vec{x_1}, \vec{x_2}, ... \vec{x_P}]`;
-  assemble the :math:`\beta_p` coefficients into a vector
   :math:`\vec{\beta} = \beta_1, \beta_2, ..., \beta_P`.

Then matrix multiplication does the rest:

.. math::


   \vec{y} = \mathbf{X} \cdot \vec{\beta} + \vec{\varepsilon}

Models with correlated regressors
---------------------------------

Some correlated regressors
~~~~~~~~~~~~~~~~~~~~~~~~~~

Imagine we have a TR (image) every 2 seconds, for 30 seconds. Here are
the times of the TR onsets, in seconds:

.. nbplot::

    >>> times = np.arange(0, 30, 2)
    >>> times
    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

Now we make a hemodynamic response function (HRF) shape for an event
starting at time 0. Call this ``hrf1``:

.. nbplot::

    >>> # Gamma distribution from scipy
    >>> from scipy.stats import gamma
    >>>
    >>> # Make SPM-like HRF shape with gamma for peak and undershoot
    >>> hrf1 = gamma.pdf(times, 6) - 0.35 * gamma.pdf(times, 12)
    >>> # Scale area under curve to 1
    >>> hrf1 = hrf1 / np.sum(hrf1)
    >>> # Plot
    >>> plt.plot(times, hrf1)
    >>> plt.title('HRF at t=0 (not mean-centered)')
    <...>



Now we make another HRF starting at t=2 (at the beginning of the second
TR):

.. nbplot::

    >>> # HRF starting at t=2
    >>> hrf2 = np.zeros(hrf1.shape)
    >>> hrf2[1:] = hrf1[0:-1]

For simplicity, we remove the mean from these regressors. We do this to
make the HRF regressors independent of the mean (:math:`\vec{1}`)
regressor, but it may not be clear yet why that is a good idea.

.. nbplot::

    >>> # Remove the mean from both HRF regressors
    >>> hrf1 = (hrf1 - hrf1.mean())
    >>> hrf2 = (hrf2 - hrf2.mean())
    >>> plt.plot(times, hrf1, label='hrf1 start t=0')
    >>> plt.plot(times, hrf2, label='hrf2 start t=2')
    >>> plt.legend()
    <...>



These ``hrf1`` and ``hrf2`` regressors are correlated. The Pearson
correlation coefficient between the HRFs is:

.. nbplot::

    >>> np.corrcoef(hrf1, hrf2)
    array([[ 1.    ,  0.7023],
           [ 0.7023,  1.    ]])

Some simulated data
~~~~~~~~~~~~~~~~~~~

Now we are going to make some simulated data from the *signal* formed
from the correlated regressors, plus some random *noise*.

The *signal* comes from the sum of ``hrf1`` and ``hrf2``. This simulates
the occurence of two events, one starting at t=0, one at t=2, both
causing an HRF response:

.. nbplot::

    >>> signal = hrf1 + hrf2
    >>> plt.plot(hrf1, label='hrf1')
    >>> plt.plot(hrf2, label='hrf2')
    >>> plt.plot(signal, label='signal (combined hrfs)')
    >>> plt.legend()
    <...>



The simulated data is this signal combined with some random noise:

.. nbplot::

    >>> noise = np.random.normal(size=times.shape)
    >>> Y = signal + noise
    >>> plt.plot(times, signal, label='signal')
    >>> plt.plot(times, Y, '+', label='signal + noise')
    >>> plt.legend()
    <...>



We are going apply several linear models to these simulated data.

All our models include a regressor of a vector of ones, :math:`\vec{1}`,
modeling the mean.

We will call our ``hrf1`` vector :math:`\vec{h_1}`. Call ``hrf2`` :
:math:`\vec{h_2}`.

Our models are:

-  A model with :math:`\vec{x}` vectors :math:`\vec{h_1}, \vec{1}` -
   single HRF model);
-  A model with :math:`\vec{h_1}, \vec{h_2}, \vec{1}` - both HRFs model;
-  A model with :math:`\vec{h_1}, \vec{w}, \vec{1}`, where
   :math:`\vec{w}` is :math:`\vec{h_2}` (``hrf2``) *orthogonalized with
   respect to* :math:`\vec{h_1}` (``hrf1``). We explain what we mean by
   this further down the page.

First, the model with :math:`\vec{h_1}, \vec{1}`:

.. nbplot::

    >>> # Design matrix for single HRF model
    >>> X_s = np.vstack((hrf1, np.ones_like(hrf1))).T
    >>> plt.imshow(X_s, interpolation='nearest', cmap='gray')
    >>> plt.title('Model with hrf1 regressor')
    <...>



Simulating the effect of noise on parameter estimates
-----------------------------------------------------

Remember that the students-t statistic is:

.. math::


   t = \frac{c^T \hat\beta}{\sqrt{\mathrm{var}(c^T \hat\beta)}}

where :math:`c^T` is a row vector of contrast weights,
:math:`\hat{\beta}` is our vector of estimated parameters, and
:math:`\mathrm{var}(c^T \hat\beta)` is the variance of
:math:`c^T \hat\beta`.

On the assumption of zero mean normally distributed independent noise:

.. math::


   \mathrm{var}(c^T \hat\beta) = \hat{\sigma}^2 c^T (X^T X)^+ c

where :math:`\hat{\sigma}^2` is our estimate of variance in the
residuals, and :math:`(X^T X)^+` is the
`pseudo-inverse <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse>`__
of :math:`X^T X`.

Therefore:

.. math::


   t = \frac{c^T \hat\beta}{\sqrt{\hat{\sigma}^2 c^T (X^T X)^+ c}}

We will see that this expection of variance correctly predicts that
parameter estimates for correlated regressors will have higher variance
for a given level of noise (where the level of noise can be captured by
:math:`\hat{\sigma}^2`).

Put another way, the parameter estimates of correlated regressors are
more susceptible to the effect of noise.

We can look at the variability of the parameter estimates, by estimating
our models on many different simulated data vectors.

Each of the data vectors are made of the *signal* (the sum of ``hrf1``
and ``hrf2``) and some *noise*. We take a new sample of noise for each
data vector:

.. nbplot::

    >>> # Create array of simulated data vectors in columns
    >>> n_times = len(times) # number of elements in single data vector
    >>> n_data_vectors = 100000
    >>> # Make many noise vectors (new noise for each column)
    >>> noise_vectors = np.random.normal(size=(n_times, n_data_vectors))
    >>> # add signal to make data vectors
    >>> # Use numpy broadcasting to add vector elementwise to 2D array
    >>> Ys = noise_vectors + signal.reshape(n_times, 1)
    >>> Ys.shape
    (15, 100000)

We first fit the model with only the first HRF regressor to every
(signal + noise) sample vector.

We will fit our model for each data vector, to make a estimated
parameter vector for each data vector. We can stack these estimated
parameter vectors into a 2 by ``n_data_vectors`` array.

Call this array :math:`\beta^s` (where the :math:`s` superscript is for
the *single* HRF model).

.. nbplot::

    >>> # Fit X_one to signals + noise
    >>> n_regressors = X_s.shape[1]
    >>> # beta (parameter estimate) matrix, one column per data vector
    >>> B_s = np.zeros((n_regressors, n_data_vectors))
    >>> # Estimate the parameters of the model for each data vector
    >>> X_pinv = npl.pinv(X_s)
    >>> for i in range(n_data_vectors):
    ...     B_s[:, i] = X_pinv.dot(Ys[:, i])

In fact, because of the way that matrix multiplications works, we can do
exactly the same calculation as we did in the loop above, in one matrix
multiplication:

.. nbplot::

    >>> B_again = X_pinv.dot(Ys)
    >>> assert np.allclose(B_s, B_again)

We will use this trick to estimate the parameter matrices for the rest
of our models.

Let us look at the variance of the first parameter estimate (the
parameter for the ``hrf1`` regressor).

Here is the variance we observe:

.. nbplot::

    >>> plt.hist(B_s[0], bins=100)
    >>> print('Observed B[0] variance for single hrf model:', np.var(B_s[0]))

    Observed B[0] variance for single hrf model: 2.2014181026



We can compare the observed variance of the first parameter estimate
with that expected from the formula above:

.. math::


   \mathrm{var}(c^T \hat\beta) = \hat{\sigma}^2 c^T (X^T X)^+ c

To select only the first regressor, we use a contrast vector of

.. math::


   c = \left[
   \begin{array}{\cvec}
   1 \\
   0 \\
   \end{array}
   \right]

Our :math:`\hat{\sigma}^2` will be close to 1, because we added noise
with variance 1:

.. nbplot::

    >>> # Estimate sigma^2 for every data vector
    >>> predicted = X_s.dot(B_s)
    >>> residuals = Ys - predicted
    >>> # Residuals have N-P degrees of freedom
    >>> N = n_times
    >>> P = npl.matrix_rank(X_s) # number of independent columns in design
    >>> sigma_hat_squared = (residuals ** 2).sum(axis=0) / (N - P)
    >>> print('Mean sigma^2 estimate:', np.mean(sigma_hat_squared))

    Mean sigma^2 estimate: 1.01981361247

Because :math:`\hat{\sigma}^2 \approx 1`,
:math:`\mathrm{var}(c^T \hat\beta) \approx c^T (X^T X)^+ c`:

.. nbplot::

    >>> C_s = np.array([[1], [0]]) # column vector
    >>> # c.T{X.T X}+ c
    >>> C_s.T.dot(npl.pinv(X_s.T.dot(X_s)).dot(C_s))
    array([[ 2.2051]])

Notice that the mean of the parameter estimates for
$:raw-latex:`\vec{h_1}` (``hrf1``), is somewhere above one, even though
we only added 1 times the first HRF as the signal:

.. nbplot::

    >>> print('Observed B[0] mean for single hrf model:', np.mean(B_s[0]))

    Observed B[0] mean for single hrf model: 1.70147967742

This is because the single first regresssor has to fit *both*
:math:`\vec{h_1}` in the signal, and as much as possible of
:math:`\vec{h_2}` in the signal, because there is nothing else in the
model to fit :math:`\vec{h_2}`.

Now let us construct the model with both HRFs as regressors:

.. nbplot::

    >>> # Design matrix for both HRFs model
    >>> X_b = np.vstack((hrf1, hrf2, np.ones_like(hrf1))).T
    >>> plt.imshow(X_b, interpolation='nearest', cmap='gray')
    >>> plt.title('Model with hrf1, hrf2 regressors')
    <...>



We will call the resulting 3 by ``n_data_vectors`` parameter array :
:math:`\beta^b` (where the :math:`b` superscript is for *both* HRF
regressors).

We will use the matrix multiplication trick above to fit all the data
vectors at the same time:

.. nbplot::

    >>> # Fit X_both to signals + noise
    >>> B_b = npl.pinv(X_b).dot(Ys)

What estimates do we get for the first regressor, when we have both
regressors in the model?

.. nbplot::

    >>> plt.hist(B_b[0], bins=100)
    >>> print('Observed B[0] mean for two hrf model', np.mean(B_b[0]))
    >>> print('Observed B[0] variance for two hrf model', np.var(B_b[0]))

    Observed B[0] mean for two hrf model 1.00299905499
    Observed B[0] variance for two hrf model 4.35228828591



Two things have happened now we added the second (correlated)
:math:`\vec{h_2}` regressor. First, the mean of the parameter for the
:math:`\vec{h_1}` regressor has dropped to 1, because
:math:`\beta^b_1 \vec{h_1}` is no longer having to model the signal due
to :math:`\vec{h_2}`. Second, the variability of the estimate has
increased. This is what the bottom half of the t-statistic predicts:

.. nbplot::

    >>> # Predicted variance for hrf1 parameter in both HRF model
    >>> C_b = np.array([[1], [0], [0]])  # column vector
    >>> C_b.T.dot(npl.pinv(X_b.T.dot(X_b)).dot(C_b))
    array([[ 4.3517]])

The estimate of the parameter for :math:`\vec{h_2}` has a mean of around
1, like the parameter estimates for :math:`\vec{h_1}`:

.. nbplot::

    >>> plt.hist(B_b[1], bins=100)
    >>> print('Observed B[1] mean for two hrf model', np.mean(B_b[1]))

    Observed B[1] mean for two hrf model 0.994534456872



This mean of 1 is what we expect because we have
:math:`\vec{h_1} + \vec{h_2}` in the signal. Not surprisingly, the
:math:`\vec{h_2}` parameter estimate has a similar variability to that
for the :math:`\vec{h_1}` parameter estimate:

.. nbplot::

    >>> print('Observed B[1] variance for two hrf model', np.var(B_b[1]))

    Observed B[1] variance for two hrf model 4.37230501038

The observed variance is very similar to the predicted variance:

.. nbplot::

    >>> C_b_1 = np.array([0, 1, 0])[:, None]  # column vector
    >>> C_b_1.T.dot(npl.pinv(X_b.T.dot(X_b)).dot(C_b_1))
    array([[ 4.3519]])

The parameter estimates for :math:`\vec{h_1}` and :math:`\vec{h_2}` are
anti-correlated:

.. nbplot::

    >>> # Relationship of estimated parameter of hrf1 and hrf2
    >>> plt.plot(B_b[0], B_b[1], '.')
    >>> np.corrcoef(B_b[0], B_b[1])
    array([[ 1.   , -0.703],
           [-0.703,  1.   ]])



Orthogonalizing hrf2 with respect to hrf1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\vec{h_2}` is correlated with $:raw-latex:`\vec{h_1}`.

We can therefore think of :math:`\vec{h_2}` as the sum of some scaling
of :math:`\vec{h_1}` plus an extra part that cannot be explained by
:math:`\vec{h_1}`.

.. math::


   \vec{h_2} = p (\vec{h_1}) + \vec{w}

where :math:`p` is some scalar, and

.. math::


   \vec{w} = \vec{h_2} - p (\vec{h_1})

To restate, we can think of :math:`\vec{h_2}` as the sum of some scalar
amount of :math:`\vec{h_1}` plus :math:`\vec{w}`.

We want to chose :math:`p` such that
:math:`\vec{w} = \vec{h_2} - p (\vec{h_1})` is orthogonal to
:math:`\vec{h_1}`. In this case :math:`\vec{w}` is the part of
:math:`\vec{h_2}` that cannot be explained by :math:`\vec{h_1}`.

If :math:`\vec{w}` is orthogonal to :math:`\vec{h_1}` then we call
:math:`\vec{w}` : :math:`\vec{h_2}` *orthogonalized with respect to*
:math:`\vec{h_1}`.

Following the same logic as for
`projection <http://practical-neuroimaging.github.io/day7.html#key-video-on-projecting-vectors>`__,
given :math:`\vec{w} - p (\vec{h_1})` is orthogonal to
:math:`\vec{h_1}`:

.. math::


   (\vec{w} - p (\vec{h_1})) \cdot \vec{h_1} = 0 \implies \\
   \vec{w} \cdot \vec{h_1} - p (\vec{h_1}) \cdot \vec{h_1} = 0 \implies \\
   \frac{\vec{w} \cdot \vec{h_1}}{\vec{h_1} \cdot \vec{h_1}} = p

Put another way, :math:`p (\vec{h_1})` such that
:math:`\vec{w} = \vec{h_2} - p (\vec{h_1})` is orthogonal to
:math:`\vec{h_1}` - is also the projection of :math:`\vec{h_2}` onto
:math:`\vec{h_1}`.

.. nbplot::

    >>> # Project hrf2 onto hrf1
    >>> p = hrf2.dot(hrf1) / hrf1.dot(hrf1)
    >>> projection = p * hrf1
    >>> # Get \vec{w} 
    >>> w = hrf2 - projection
    >>> # w and hrf1 are now orthogonal
    >>> assert np.allclose(w.dot(hrf1), 0)

.. nbplot::

    >>> # Plot the vector parts
    >>> plt.plot(times, hrf1, label=r'$\vec{h_1}$')
    >>> plt.plot(times, hrf2, label=r'$\vec{h_2}$')
    >>> plt.plot(times, projection, label=r'$p (\vec{h_1})$')
    >>> plt.plot(times, w, label=r'$\vec{w}$')
    >>> plt.legend()
    >>> # hrf1 part of hrf2, plus unique part, equals original hrf2
    >>> assert np.allclose(hrf2, projection + w)




How much of the first regressor did we find in the second regressor?

.. nbplot::

    >>> p
    0.70231917818451162

Let us rewrite our original model containing ``hrf1, hrf2``:

.. math::


   \vec{y} = \beta^b_1 \vec{h_1} + \beta^b_2 \vec{h_2} + \beta^b_3 1

where :math:`\vec{y}` is our data vector.

Now we know we can also write this as:

.. math::


   \vec{y} = \beta^b_1 \vec{h_1} + 
     \beta^b_2 (p (\vec{h_1}) + \vec{w}) + \beta^b_3 1 \\
   = \beta^b_1 \vec{h_1} + 
     \beta^b_2 p (\vec{h_1}) + \beta^b_2 \vec{w} + \beta^b_3 1 \\
   = (\beta^b_1 + p \beta^b_2) \vec{h_1} + \beta^b_2 \vec{w} + \beta^b_3 1

So, what will happen if we drop :math:`\vec{h_2}` from our model and
leave only :math:`\vec{w}`?

We have called the parameters from the model with
:math:`\vec{h_1}, \vec{h_2}` : :math:`\beta^b`. Call the parameters from
the model with :math:`\vec{h_1}, \vec{w}` : :math:`\beta^w`.

We can see that we are going to get the exact same fit to the data with
these two models if :math:`\beta^w_2 = \beta^b_2` and
:math:`\beta^w_1 = p \beta^b_2 + \beta^b_1`.

Let us try this new model and see:

.. nbplot::

    >>> X_w = np.vstack((hrf1, w, np.ones_like(hrf1))).T
    >>> plt.imshow(X_w, interpolation='nearest', cmap='gray')
    <...>



.. nbplot::

    >>> # Fit the model
    >>> B_w = npl.pinv(X_w).dot(Ys)

Let us first look at the distribution of :math:`\beta^w_1`
``== B_w[0]``.

.. nbplot::

    >>> # Distribution of parameter for hrf1 in orth model
    >>> plt.hist(B_w[0], bins=100)
    >>> print('Observed B[0] mean for two hrf orth model',
    ...       np.mean(B_w[0]))

    Observed B[0] mean for two hrf orth model 1.70147967742



Notice that :math:`\beta^w_1` now has the same values as for the single
HRF model : :math:`\beta^s_1`:

.. nbplot::

    >>> assert np.allclose(B_s[0, :], B_w[0, :])

It therefore has the same variance, and the predicted variance matches:

.. nbplot::

    >>> print('Observed B[0] variance for two hrf orth model', np.var(B_w[0]))
    >>> pred_var = C_b.T.dot(npl.pinv(X_w.T.dot(X_w)).dot(C_b))
    >>> print('Predicted B[0] variance for two hrf orth model', pred_var)

    Observed B[0] variance for two hrf orth model 2.2014181026
    Predicted B[0] variance for two hrf orth model [[ 2.2051]]

The fact that the single hrf and orthogonalized model parameters match
may make sense when we remember that adding the :math:`\vec{w}`
regressor to the model cannot change the parameter for the
:math:`\vec{h_1}` regressor as :math:`\vec{w}` is orthogonal to
:math:`\vec{h_1}`.

We predicted above that :math:`\beta^w_2` would stay the same as
:math:`\beta^b_2` from the not-orthogonalized model:

.. nbplot::

    >>> assert np.allclose(B_b[1, :], B_w[1, :])

We predicted that :math:`\beta^w_1` would become
:math:`\beta^b_1 + p \beta^b_2` from the not-orthogonalized model:

.. nbplot::

    >>> predicted_beta1 = B_b[0, :] + p * B_b[1, :]
    >>> assert np.allclose(predicted_beta1, B_w[0, :])

Our predictions were correct. So let us revise what happened:

-  We estimated our original model with correlated
   :math:`\vec{h_1}, \vec{h_2}` to get corresponding estimated
   parameters :math:`\beta^b_1, \beta^b_2`;
-  we orthogonalized :math:`\vec{h_2}` with respect to :math:`\vec{h_1}`
   to give :math:`p` and :math:`\vec{w}`;
-  we replaced :math:`\vec{h_2}` with :math:`\vec{w}` in the model, and
   re-estimated, giving new parameters :math:`\beta^w_1` for
   :math:`\vec{h_1}`, :math:`\beta^w_2` for :math:`\vec{w}`;
-  :math:`\beta^w_2 = \beta^b_2` - the parameter for the new
   orthogonalized regressor is unchanged from the non-orthogonalized
   case;
-  :math:`\beta^w_1 = \beta^b_1 + p \beta^b_2` - the parameter for the
   *unchanged* regressor has increased by :math:`\beta^b_2` times the
   amount of :math:`\vec{h_2}` present in :math:`\vec{h_1}`.

Here we show some example parameters from the three model fits:

.. nbplot::

    >>> # Example parameters from the single hrf model
    >>> B_s[:,:5]
    array([[ 0.0032,  1.2188,  2.289 ,  0.2179,  3.45  ],
           [-0.0969,  0.0488,  0.006 , -0.1493, -0.4889]])

.. nbplot::

    >>> # Example parameters from the non-orth two-hrf model
    >>> B_b[:,:5]
    array([[-1.3014,  1.676 ,  0.3588, -0.6153,  0.8819],
           [ 1.8574, -0.6509,  2.7482,  1.1865,  3.6566],
           [-0.0969,  0.0488,  0.006 , -0.1493, -0.4889]])

.. nbplot::

    >>> # Example parameters from the orth model
    >>> B_w[:,:5]
    array([[ 0.0032,  1.2188,  2.289 ,  0.2179,  3.45  ],
           [ 1.8574, -0.6509,  2.7482,  1.1865,  3.6566],
           [-0.0969,  0.0488,  0.006 , -0.1493, -0.4889]])

.. nbplot::

    >>> # The parameter for the hrf1 regressor in the non-orth model
    >>> # is correlated with the parameter for the hrf1 regressor
    >>> # in the orth model.
    >>> plt.plot(B_b[0], B_w[0], '.')
    >>> plt.title('Orth and non-orth hrf1 parameters correlate')
    >>> np.corrcoef(B_b[0], B_w[0])
    array([[ 1.    ,  0.7103],
           [ 0.7103,  1.    ]])



.. nbplot::

    >>> # Relationship of estimated parameters for hrf1 and orthogonalized hrf2
    >>> # (they should be independent)
    >>> plt.plot(B_w[0], B_w[1], '+')
    >>> plt.title('hrf1 and orth hrf2 parameters are independent')
    >>> np.corrcoef(B_w[0], B_w[1])
    array([[ 1.    ,  0.0013],
           [ 0.0013,  1.    ]])



