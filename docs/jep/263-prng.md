(prng-design-jep)=
# JAX PRNG Design
We want a PRNG design that
1. is **expressive** in that it is convenient to use and it doesn’t constrain the user’s ability to write numerical programs with exactly the behavior that they want,
2. enables **reproducible** program execution in a backend-independent way,
3. has semantics that are **invariant to `@jit` compilation boundaries and device backends**,
4. enables **vectorization for generating array values** using SIMD hardware,
5. is **parallelizable** in that it doesn’t add sequencing constraints between random function calls that otherwise would have no data dependence,
6. scales to **multi-replica, multi-core, and distributed computation**,
7. **fits with JAX and XLA semantics** and design philosophies (which are ultimately motivated by other practical concerns).

As a corollary of these we believe the design should be functional. Another corollary is that, at least given current hardware constraints, we’re going to do the PRNG in software.

> TLDR
> **JAX PRNG = [Threefry counter PRNG](http://www.thesalmons.org/john/random123/papers/random123sc11.pdf) + a functional array-oriented [splitting model](https://dl.acm.org/citation.cfm?id=2503784)**

## Contents
* [Three programming models and toy example programs](#three-programming-models-and-toy-example-programs)
* [Design](#design)
* [More realistic example user programs](#more-realistic-example-user-programs)
* [Tradeoffs and alternatives](#tradeoffs-and-alternatives)


## Three programming models and toy example programs

Here’s a toy example of a **stateful global** PRNG like the one often used in Numpy programs:

```python
def foo(): return bar() + baz()
def bar(): return rand(RNG, (3, 4))
def baz(): return rand(RNG, (3, 4))
def main():
  global RNG
  RNG = RandomState(0)
  return foo()
```

To achieve reproducibility here we would need to control the order of evaluation for bar() and baz() even though there is no explicit data dependence from one to the other. This kind of sequencing requirement stemming from reproducibility (#2) violates parallelizability (#5) and doesn’t fit with JAX or XLA’s functional semantics (#6) in which subexpressions can be evaluated in any order. Even if we didn’t require reproducibility and thus allowed any evaluation order, parallelization across calls (#5) would still be made difficult by the need to update shared state. Moreover, because the same PRNG state would need to be accessed and maintained in both Python and any compiled code, this model would likely lead to engineering challenges to achieve compilation invariance (#3) and scaling to multiple replicas (#6). Finally, the expressiveness is limited (#1) because there is no way for foo() to call bar() or baz() without affecting its own (implicit) PRNG state.

Whether the model supports vectorization (#4) depends on some additional details. In Numpy, PRNG vectorization is limited by a *sequential-equivalent guarantee*:

```python
In [1]: rng = np.random.RandomState(0)

In [2]: rng.randn(2)
Out[2]: array([1.76405235, 0.40015721])

In [3]: rng = np.random.RandomState(0)

In [4]: np.stack([rng.randn() for _ in range(2)])
Out[4]: array([1.76405235, 0.40015721])
```

To allow for vectorization (#4) within primitive PRNG function calls that generate arrays (e.g. to rand() with a shape argument), we drop this sequential-equivalent guarantee. This vectorization can be supported by any of the three programming models discussed in this section, though it motivates the implementation in terms of a counter-based PRNG as described in the next section.

The stateful PRNG user programming model is not promising. Here’s an example of a functional model but lacking a key ingredient that we call splitting:

```python
def foo(rng_1):
   y, rng_2 = baz(rng_1)
   z, rng_3 = bar(rng_2)
   return y + z, rng_3

def bar(x, rng):
  val, new_rng = rand(rng, (3, 4))
  return val, new_rng

def baz(x, rng):
  val, new_rng = rand(rng, (3, 4))
  return val, new_rng

def main():
  foo(RandomState(0))
```

This model explicitly threads the PRNG state through all functions (primitive or non-primitive) that generate random values: that is, every random function must both accept and return the state. Now there is an explicit data dependence between the call to baz() and the call to bar() in foo(), so the data flow (and hence sequencing) is made explicit and fits with JAX’s existing semantics (#7), unlike in the previous model. This explicit threading can also make the semantics invariant to compilation boundaries (#3).

Explicit threading is inconvenient for the programmer. But worse, it hasn’t actually improved the expressiveness (#1): there is still no way for foo() to call into bar() or baz() while maintaining its own PRNG state. Without knowledge of their callers or the subroutines they call, functions must defensively pass in and return the rng state everywhere. Moreover, it also doesn’t improve the prospects for parallelization (#5) or scaling to multiple replicas (#6) because everything is still sequential, even if the sequencing is made explicit in the functional programming sense.

In short, making the code functional by explicitly threading state isn’t enough to achieve our expressiveness (#1) and performance (#5, #6) goals.

The key problem in both the previous models is that there’s too much sequencing. To reduce the amount of sequential dependence we use **functional [splittable](https://dl.acm.org/citation.cfm?id=2503784) PRNGs**. Splitting is a mechanism to ‘fork’ a new PRNG state into two PRNG states while maintaining the usual desirable PRNG properties (the two new streams are computationally parallelizable and produce independent random values, i.e. they behave like [multistreams](http://www.thesalmons.org/john/random123/papers/random123sc11.pdf)).

```python
def foo(rng_1):
   rng_2, rng_3 = split(rng_1, 2)
   return bar(rng_2) + baz(rng_3)

def bar(x, rng):
  return rand(rng, (3, 4))

def baz(x, rng):
  return rand(rng, (3, 4))

def main():
  foo(RandomState(0))
```

Some points to notice:

1. there is no sequential dependence between the calls to bar() and baz() and they can be evaluated in either order without affecting the value of the result, which solves the remaining performance goals (#5, #6),
2. functions do not need to return updated versions of PRNGs and it is straightforward to call a random subroutine without affecting existing PRNG states, improving the expressiveness (#1) from the other functional model.

The example doesn’t show it, but as a consequence of the choice (2) the only way to advance the PRNG state is to call split(). That is, we have two ways to achieve (1), and they differ in whether they burden the user program with explicit calls to split(), as in the above example, or instead burden the user program with explicit threading. We prefer the former, i.e. the version with explicit splitting, because we can easily implement the explicit-threading version in terms of it.

## Design

We can use the *counter-based PRNG* design, and in particular the Threefry hash function, as described in [Parallel random numbers: as easy as 1, 2, 3](http://www.thesalmons.org/john/random123/papers/random123sc11.pdf). We use the counter to achieve efficient vectorization: for a given key we can generate an array of values in a vectorized fashion by mapping the hash function over a range of integers [k + 1, …, k + sample_size]. We use the key together with the hash function to implement [splittable PRNGs](https://dl.acm.org/citation.cfm?id=2503784): that is, splitting is a way to generate two new keys from an existing one.

```haskell
type Sample = Int256
type Key = Sample  -- important identification for splitting
type Count = Int32

hash :: Key -> Count -> Int256  -- output type equal to Key and Sample

split :: Key -> (Key, Key)
split key = (hash key 0, hash key 1)

draw_samples :: Key -> Int -> [Sample]
draw_samples key n = map (hash key) [1..n]
```

Surprisingly, drawing a sample is very similar to splitting! The key is the difference in the type of the output (even though the types are identified): in one case the value is to be used in forming random samples of interest (e.g. turning random bits into a Float representing a random normal) while in the other case the value is to be used as a key for further hashing.

The asymmetry in the hash function arguments, of type Key and Count, is that the latter is trivial and computationally cheap to advance by an arbitrary amount, since we just need to increase the integer value, while the former is only advanced by hashing. That’s why we use the count argument for vectorization.

## More realistic example user programs

Here’s what a training loop on the host might look like when the step requires a PRNG (maybe for dropout or for VAE training):

```python
rng = lax.rng.new_rng()
for i in xrange(num_steps):
  rng, rng_input = lax.rng.split(rng)
  params = compiled_update(rng_input, params, next(batches))
```

Notice that we’re burdening the user with explicit splitting of the rng, but the rng does not need to be returned from the code at all.

Here’s how we can use this PRNG model with the stax neural net builder library to implement dropout:

```python
def Dropout(rate, mode='train'):
  def init_fun(input_shape):
    return input_shape, ()
  def apply_fun(rng, params, inputs):
    if mode == 'train':
      keep = lax.random.bernoulli(rng, rate, inputs.shape)
      return np.where(keep, inputs / rate, 0)
    else:
      return inputs
  return init_fun, apply_fun
```

The rng value here is just the key used for the hash, not a special object. The rng argument is passed to every apply_fun, and so it needs to be handled in the serial and parallel combinators with splitting:

```python
def serial(*layers):
  init_funs, apply_funs = zip(*layers)
  def init_fun(input_shape):
    ...
  def apply_fun(rng, params, inputs):
    rngs = split(rng, len(layers))
    for rng, param, apply_fun in zip(rngs, params, apply_funs):
      inputs = apply_fun(rng, param, inputs)
    return inputs
  return init_fun, apply_fun

def parallel(*layers):
  init_funs, apply_funs = zip(*layers)
  def init_fun(input_shape):
    ...
  def apply_fun(rng, params, inputs):
    rngs = split(rng, len(layers))
    return [f(r, p, x) for f, r, p, x in zip(apply_funs, rngs, params, inputs)]
  return init_fun, apply_fun
```

Here we’re using a simple extended version of split that can produce multiple copies.

## Tradeoffs and alternatives

1. We're not exploiting any device hardware PRNG
   - We don’t currently have enough control over the hardware PRNG’s state for all backends.
   - Even if we did, it would be backend-dependent and we might have to introduce sequential dependencies between random calls to ensure deterministic ordering and hence reproducibility.
   - We don’t know of any workloads for which the software PRNG should become a bottleneck.
   - We could consider providing an additional API that allows access to a hardware PRNG for users who want to give up other desiderata (like strict reproducibility).
2. We give up the sequential equivalent guarantee, in which creating a random array in one call produces the same values as creating the flattened array one random element at a time.
   - This property is likely incompatible with vectorization (a high priority).
   - We don’t know of any users or examples for which this property is important.
   - Users could write a layer on top of this API to provide this guarantee.
3. We can't follow the `numpy.random` API exactly.
