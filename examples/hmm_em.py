from functools import partial

import jax
import jax.numpy as np
from jax.scipy.special import logsumexp
from jax import lax, random
from jax import jit, grad


def log_normalizer(params, seq):
  pi, A, B = map(np.exp, params)
  def marginalize(carry, obs):
    alpha, logprob = carry
    alpha = np.dot(alpha, A) * B[:, obs]
    new_carry = alpha / alpha.sum(), logprob + np.log(alpha.sum())
    return new_carry, None
  (_, logprob), _ = lax.scan(marginalize, (pi, 0.), seq)
  return logprob

@jit
def EM(params, seq, num_steps):
  def update(i, params):
    moments = grad(log_normalizer)(params, seq)                    # E step
    params = [np.log(normalize(x, smooth=1e-2)) for x in moments]  # M step
    return params
  return lax.fori_loop(0, num_steps, update, params)

def normalize(x, smooth):
  x = x + smooth
  return x / x.sum(-1, keepdims=True)

@partial(jit, static_argnums=(2,))
def sample(key, params, length):
  log_pi, log_A, log_B = params
  def sample_one(state, key):
    k1, k2 = random.split(key)
    obs = random.categorical(k1, log_B[state])
    state = random.categorical(k2, log_A[state])
    return state, obs
  k1, k2 = random.split(key)
  init_state = random.categorical(k1, log_pi)
  _, seq = lax.scan(sample_one, init_state, random.split(k2, length))
  return seq


### demo

from collections import defaultdict
from itertools import count

def build_dataset(string):
  encodings = defaultdict(partial(next, count()))
  data = np.array([encodings[c] for c in string])
  decodings = {i:c for c, i in encodings.items()}
  return data, decodings

if __name__ == '__main__':
  with open(jax.core.__file__, 'r') as f:
    seq, decodings = build_dataset(f.read())

  num_states = 200
  num_obs = len(decodings)

  keys = map(random.PRNGKey, count())
  log_pi = random.normal(next(keys), (num_states,))
  log_A = random.normal(next(keys), (num_states, num_states))
  log_B = random.normal(next(keys), (num_states, num_obs))
  params = [log_pi, log_A, log_B]

  new_params = EM(params, seq, 50)

  sampled_seq = sample(next(keys), new_params, 200)
  print(''.join(decodings[i] for i in sampled_seq))
