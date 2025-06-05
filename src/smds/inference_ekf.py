import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jscipy
from jax import lax, vmap
from jax import jacfwd, jacrev
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import List, Optional, NamedTuple, Optional, Union, Callable

from src.utils.utils import psd_solve, symmetrize, inv_via_cholesky, rotate_subspace
from src.lds.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
from src.types import PRNGKey

# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x

FnStateToState = Callable[ [Float[Array, "state_dim"]], Float[Array, "state_dim"]]
FnStateAndInputToState = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "state_dim"]]
FnStateToEmission = Callable[ [Float[Array, "state_dim"]], Float[Array, "emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, "state_dim"], Float[Array, "input_dim"] ], Float[Array, "emission_dim"]]

class ParamsNLGSSM(NamedTuple):
    """Parameters for a NLGSSM model.

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    :param dynamics_function: $f$
    :param dynamics_covariance: $Q$
    :param emissions_function: $h$
    :param emissions_covariance: $R$
    :param initial_mean: $m$
    :param initial_covariance: $S$

    """

    initial_mean: Float[Array, "state_dim"]
    initial_covariance: Float[Array, "state_dim state_dim"]
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_covariance: Float[Array, "state_dim state_dim"]
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_covariance: Float[Array, "emission_dim emission_dim"]

def _predict(m, P, Q):
    r"""Predict next mean and covariance using first-order additive EKF

        p(z_{t+1}) = \int N(z_t | m, S) N(z_{t+1} | f(z_t, u), Q)
                    = N(z_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)

    Args:
        m (D_hid,): prior mean.
        P (D_hid,D_hid): prior covariance.
        f (Callable): dynamics function.
        F (Callable): Jacobian of dynamics function.
        Q (D_hid,D_hid): dynamics covariance matrix.
        u (D_in,): inputs.

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    return m, P + Q


def _condition_on(m, P, h, H, R, u, y, num_iter):
    r"""Condition a Gaussian potential on a new observation.

       p(z_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t | y_{1:t-1}, u_{1:t-1}) p(y_t | z_t, u_t)
         = N(z_t | m, S) N(y_t | h_t(z_t, u_t), R_t)
         = N(z_t | mm, SS)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = h(m, u)
         S = R + H(m,u) * P * H(m,u)'
         K = P * H(m, u)' * S^{-1}
         SS = P - K * S * K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         h (Callable): emission function.
         H (Callable): Jacobian of emission function.
         R (D_obs,D_obs): emission covariance matrix.
         u (D_in,): inputs.
         y (D_obs,): observation.
         num_iter (int): number of re-linearizations around posterior for update step.

     Returns:
         mu_cond (D_hid,): filtered mean.
         Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    def _step(carry, _):
        prior_mean, prior_cov = carry
        H_x = H(prior_mean, u)
        S = R + H_x @ prior_cov @ H_x.T
        K = psd_solve(S, H_x @ prior_cov).T
        posterior_cov = prior_cov - K @ S @ K.T
        posterior_mean = prior_mean + K @ (y - h(prior_mean, u))
        return (posterior_mean, posterior_cov), None

    # Iterate re-linearization over posterior mean and covariance
    carry = (m, P)
    (mu_cond, Sigma_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, symmetrize(Sigma_cond)

_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)

def extended_kalman_filter_augmented_state(
    params: ParamsNLGSSM,
    model_params,
    emissions: Float[Array, "ntime emission_dim"],
    conditions,
    output_fields: Optional[List[str]] = ["filtered_means", "filtered_covariances", "predicted_means",
                                              "predicted_covariances"],
    block_masks = None,
    trial_masks = None,
    num_iters = 1,
) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered velocity estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".
        trial_masks: trial masks.
    Returns:
        post: posterior object.

    """

    num_blocks, num_trials_per_block, num_timesteps, emissions_dim = emissions.shape
    dim_x = model_params.initial.mean.shape[-1]
    dim_v = params.initial_mean.shape[-1]
    
    # Dynamics and emission functions and their Jacobians
    h = params.emission_function
    H = jacrev(h, argnums=0, has_aux=True)

    initial_velocity_mean = params.initial_mean
    initial_velocity_cov = params.initial_covariance

    initial_state_means = model_params.initial.mean
    initial_state_covs = model_params.initial.cov

    tau = params.dynamics_covariance

    initial_condition = conditions[0, 0]

    def _step(carry, block_id):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters
        A = model_params.dynamics.weights
        Q = model_params.dynamics.cov
        R = model_params.emissions.cov
        b = _zeros_if_none(model_params.dynamics.bias, (dim_x,))

        y = emissions[block_id]
        next_block_condition = conditions[block_id+1, 0]
        block_mask = block_masks[block_id]
        trial_mask = trial_masks[block_id]

        def _inner_step(inner_carry, r):
            ll, _, _, _pred_mean, _pred_cov = inner_carry

            # Get parameters and inputs for time index t
            y_r = y[r]
            next_trial_condition = conditions[block_id, r+1]
            trial_mask_r = trial_mask[r]

            def _inner_inner_step(inner_inner_carry, t):
                ll, _, _, _pred_mean, _pred_cov = inner_inner_carry

                y_t = y_r[t]

                # Get the Jacobian of the emission function
                H_u, y_pred = H(_pred_mean)  # (N x (V+D)), N

                # Get the innovation covariance
                s_k = H_u @ _pred_cov @ H_u.T + R
                s_k = symmetrize(s_k)

                # Update the log likelihood
                ll += MVN(y_pred, s_k).log_prob(jnp.atleast_1d(y_t))
                # jax.debug.print('ll: {ll}', ll=ll)

                def update_step(carry, _):
                    prior_mean, prior_cov = carry
                    # Get the Jacobian of the emission function
                    H_u, y_pred = H(prior_mean)  # (N x (V+D)), N

                    # Get the innovation covariance
                    s_k = H_u @ prior_cov @ H_u.T + R
                    s_k = symmetrize(s_k)

                    # Get the Kalman gain
                    K = psd_solve(s_k, H_u @ prior_cov).T

                    # Get the filtered mean
                    filtered_mean = prior_mean + K @ (y_t - y_pred) 

                    # Get the filtered covariance
                    filtered_cov = prior_cov - K @ s_k @ K.T
                    filtered_cov = symmetrize(filtered_cov)

                    return (filtered_mean, filtered_cov), None
                
                (filtered_mean, filtered_cov), _ = jax.lax.scan(update_step, 
                                                           (_pred_mean, _pred_cov), 
                                                           jnp.arange(num_iters))

                pred_mean = filtered_mean.at[:dim_x].set(A @ filtered_mean[:dim_x] + b)
                pred_cov = filtered_cov.at[:dim_x].set(A @ filtered_cov[:dim_x])
                pred_cov = pred_cov.at[:, :dim_x].set(pred_cov[:, :dim_x] @ A.T)
                pred_cov = pred_cov.at[:dim_x, :dim_x].add(Q)

                return (ll, filtered_mean, filtered_cov, pred_mean, pred_cov), None

            init_carry = (ll, jnp.zeros_like(_pred_mean), jnp.zeros_like(_pred_cov), _pred_mean, _pred_cov)

            def inner_true_fun(inputs):
                # Scan over time steps
                (ll, filtered_mean, filtered_cov, _, _), _ = lax.scan(_inner_inner_step, 
                                                                      inputs, 
                                                                      jnp.arange(num_timesteps))
                return ll, filtered_mean, filtered_cov
            
            def inner_false_fun(inputs):
                ll, _, _, _pred_mean, _pred_cov = inputs
                return ll, _pred_mean, _pred_cov
            
            ll, filtered_mean, filtered_cov = jax.lax.cond(trial_mask_r, inner_true_fun, inner_false_fun, init_carry)
            
            # Get the predicted mean and covariance across trials but within block
            pred_mean = filtered_mean.at[:dim_x].set(initial_state_means[next_trial_condition])
            pred_cov = filtered_cov.at[:dim_x].set(0.0)
            pred_cov = pred_cov.at[:,:dim_x].set(0.0)
            pred_cov = pred_cov.at[:dim_x, :dim_x].set(initial_state_covs[next_trial_condition])

            return (ll, filtered_mean, filtered_cov, pred_mean, pred_cov), None

        def true_fun(inputs):
            (ll, filtered_mean, filtered_cov, _, _), _ = lax.scan(_inner_step, inputs, jnp.arange(num_trials_per_block))
            return ll, filtered_mean, filtered_cov

        def false_fun(inputs):
            ll, _, _, _pred_mean, _pred_cov = inputs
            return ll, _pred_mean, _pred_cov

        inputs = (ll, jnp.zeros_like(_pred_mean), jnp.zeros_like(_pred_cov), _pred_mean, _pred_cov)
        ll, filtered_mean, filtered_cov = jax.lax.cond(block_mask, true_fun, false_fun, inputs)

        # Get the predicted mean and covariance across blocks
        pred_mean = filtered_mean.at[:dim_x].set(initial_state_means[next_block_condition])
        pred_cov = filtered_cov.at[:dim_x].set(0.0)
        pred_cov = pred_cov.at[:,:dim_x].set(0.0)
        pred_cov = pred_cov.at[:dim_x, :dim_x].set(initial_state_covs[next_block_condition])
        pred_cov = pred_cov.at[dim_x:, dim_x:].add(tau)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean[dim_x:],
            "filtered_covariances": filtered_cov[dim_x:, dim_x:],
        }

        return carry, outputs

    # Run the extended Kalman filter
    carry = (0.0, 
             jnp.concatenate([initial_state_means[initial_condition], initial_velocity_mean]), 
             jscipy.linalg.block_diag(initial_state_covs[initial_condition], initial_velocity_cov))
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_blocks))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered

def extended_kalman_filter(
    params: ParamsNLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    output_fields: Optional[List[str]]=["filtered_means", "filtered_covariances"], # "predicted_means", "predicted_covariances"],
    trial_masks = None,
    num_iters = 1,
) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 1).
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

    """
    num_trials = len(emissions)

    # Dynamics and emission functions and their Jacobians
    h = params.emission_function
    H = jacfwd(h, argnums=0, has_aux=True)
    # HH = hessian(h)

    Q = params.dynamics_covariance
    dv = Q.shape[-1]

    def _step(carry, t):
        ll, _pred_mean, _pred_cov = carry

        # Get parameters and inputs for time index t
        R = params.emission_covariance[t]
        y = emissions[t]
        trial_mask = trial_masks[t]

        def true_fun(inputs):
            def _update_step(carry, _):
                _pred_mean, _pred_cov = carry
                # Get the Jacobian of the emission function
                H_x, y_pred = H(_pred_mean)  # (ND x V), ND
                # y_pred = h(_pred_mean)  # ND

                s_k = H_x @ _pred_cov @ H_x.T + jscipy.linalg.block_diag(*R)

                # Condition on this emission
                K = psd_solve(s_k, H_x @ _pred_cov).T
                filtered_cov = _pred_cov -  K @ s_k @ K.T
                filtered_mean = _pred_mean + K @ (y.flatten() - y_pred)
                filtered_cov = symmetrize(filtered_cov)
                return (filtered_mean, filtered_cov), None
            
            (filtered_mean, filtered_cov), _ = lax.scan(_update_step, inputs, jnp.arange(num_iters))
            return filtered_mean, filtered_cov

        def false_fun(inputs):
            _pred_mean, _pred_cov = inputs
            return _pred_mean, _pred_cov

        inputs = (_pred_mean, _pred_cov)
        filtered_mean, filtered_cov = jax.lax.cond(trial_mask, true_fun, false_fun, inputs)

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, Q)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_covariances": filtered_cov,
            # "predicted_means": _pred_mean,
            # "predicted_covariances": _pred_cov,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Run the extended Kalman filter
    carry = (0.0, params.initial_mean, params.initial_covariance)
    (ll, *_), outputs = lax.scan(_step, carry, jnp.arange(num_trials))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered


def extended_kalman_smoother(
    params: ParamsNLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
    trial_masks = None,
    num_iters = 1,
) -> PosteriorGSSMSmoothed:
    r"""Run an extended Kalman (RTS) smoother.

    Args:
        params: model parameters.
        emissions: observation sequence.
        filtered_posterior: optional output from filtering step.
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    num_trials = len(emissions)

    # Get filtered posterior
    if filtered_posterior is None:
        filtered_posterior = extended_kalman_filter(params, emissions, trial_masks=trial_masks, num_iters=num_iters)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next, smoothed_cov_sum, smoothed_cc_sum = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time index t
        Q = params.dynamics_covariance

        # Prediction step
        m_pred = filtered_mean
        S_pred = filtered_cov + Q
        G = psd_solve(S_pred, filtered_cov).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T
        smoothed_cov = symmetrize(smoothed_cov)
        smoothed_cov_sum += smoothed_cov

        # Compute the smoothed expectation of z_t z_{t+1}^T
        smoothed_cc_sum += G @ smoothed_cov_next + jnp.outer(smoothed_mean, smoothed_mean_next)

        return ((smoothed_mean, smoothed_cov, smoothed_cov_sum, smoothed_cc_sum),
                smoothed_mean)

    dof = filtered_covs.shape[-1]
    smoothed_cross_cov_sum_init = jnp.zeros((dof, dof))
    smoothed_cov_sum_init = jnp.zeros((dof, dof))
    # Run the extended Kalman smoother
    ((_, smoothed_cov_0, smoothed_cov_sum, smoothed_cross_cov_sum),
     smoothed_means) = lax.scan(
        _step,
        (filtered_means[-1], filtered_covs[-1],
         smoothed_cov_sum_init, smoothed_cross_cov_sum_init),
        (jnp.arange(num_trials - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True,
    )

    # Concatenate the arrays and return
    smoothed_means = jnp.vstack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_cross = smoothed_cross_cov_sum
    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances_0=smoothed_cov_0,
        smoothed_covariances_p=smoothed_cov_sum,
        smoothed_covariances_n=smoothed_cov_sum - smoothed_cov_0 + filtered_covs[-1],
        smoothed_cross_covariances=smoothed_cross,
    )