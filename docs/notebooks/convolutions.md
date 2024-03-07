---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"id": "TVT_MVvc02AA"}

# Generalized Convolutions in JAX

<!--* freshness: { reviewed: '2024-04-08' } *-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/notebooks/convolutions.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/notebooks/convolutions.ipynb)

JAX provides a number of interfaces to compute convolutions across data, including:

- {func}`jax.numpy.convolve` (also {func}`jax.numpy.correlate`)
- {func}`jax.scipy.signal.convolve` (also {func}`~jax.scipy.signal.correlate`)
- {func}`jax.scipy.signal.convolve2d` (also {func}`~jax.scipy.signal.correlate2d`)
- {func}`jax.lax.conv_general_dilated`

For basic convolution operations, the `jax.numpy` and `jax.scipy` operations are usually sufficient. If you want to do more general batched multi-dimensional convolution, the `jax.lax` function is where you should start.

+++ {"id": "ewZEn2X12-Ng"}

## Basic One-dimensional Convolution

Basic one-dimensional convolution is implemented by {func}`jax.numpy.convolve`, which provides a JAX interface for {func}`numpy.convolve`. Here is a simple example of 1D smoothing implemented via a convolution:

```{code-cell} ipython3
:id: 0qYLpeZO3Z9-
:outputId: 4f6717ac-a062-4a85-8330-d57bf80de384

import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp
import numpy as np

key = random.key(1701)

x = jnp.linspace(0, 10, 500)
y = jnp.sin(x) + 0.2 * random.normal(key, shape=(500,))

window = jnp.ones(10) / 10
y_smooth = jnp.convolve(y, window, mode='same')

plt.plot(x, y, 'lightgray')
plt.plot(x, y_smooth, 'black');
```

+++ {"id": "dYX1tCVB4XOW"}

The `mode` parameter controls how boundary conditions are treated; here we use `mode='same'` to ensure that the output is the same size as the input.

For more information, see the {func}`jax.numpy.convolve` documentation, or the documentation associated with the original {func}`numpy.convolve` function.

+++ {"id": "5ndvLDIH4rv6"}

## Basic N-dimensional Convolution

For *N*-dimensional convolution, {func}`jax.scipy.signal.convolve` provides a similar interface to that of {func}`jax.numpy.convolve`, generalized to *N* dimensions.

For example, here is a simple approach to de-noising an image based on convolution with a Gaussian filter:

```{code-cell} ipython3
:id: Jk5qdnbv6QgT
:outputId: 292205eb-aa09-446f-eec2-af8c23cfc718

from scipy import misc
import jax.scipy as jsp

fig, ax = plt.subplots(1, 3, figsize=(12, 5))

# Load a sample image; compute mean() to convert from RGB to grayscale.
image = jnp.array(misc.face().mean(-1))
ax[0].imshow(image, cmap='binary_r')
ax[0].set_title('original')

# Create a noisy version by adding random Gaussian noise
key = random.key(1701)
noisy_image = image + 50 * random.normal(key, image.shape)
ax[1].imshow(noisy_image, cmap='binary_r')
ax[1].set_title('noisy')

# Smooth the noisy image with a 2D Gaussian smoothing kernel.
x = jnp.linspace(-3, 3, 7)
window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
smooth_image = jsp.signal.convolve(noisy_image, window, mode='same')
ax[2].imshow(smooth_image, cmap='binary_r')
ax[2].set_title('smoothed');
```

+++ {"id": "Op-NhXy39z2U"}

Like in the one-dimensional case, we use `mode='same'` to specify how we would like edges to be handled. For more information on available options in *N*-dimensional convolutions, see the {func}`jax.scipy.signal.convolve` documentation.

+++ {"id": "bxuUjFVG-v1h"}

## General Convolutions

+++ {"id": "0pcn2LeS-03b"}

For the more general types of batched convolutions often useful in the context of building deep neural networks, JAX and XLA offer the very general N-dimensional __conv_general_dilated__ function, but it's not very obvious how to use it.  We'll give some examples of the common use-cases.

A survey of the family of convolutional operators, [a guide to convolutional arithmetic](https://arxiv.org/abs/1603.07285), is highly recommended reading!

Let's define a simple diagonal edge kernel:

```{code-cell} ipython3
:id: Yud1Y3ss-x1K
:outputId: 3185fba5-1ad7-462f-96ba-7ed1b0c3d5a2

# 2D kernel - HWIO layout
kernel = jnp.zeros((3, 3, 3, 3), dtype=jnp.float32)
kernel += jnp.array([[1, 1, 0],
                     [1, 0,-1],
                     [0,-1,-1]])[:, :, jnp.newaxis, jnp.newaxis]

print("Edge Conv kernel:")
plt.imshow(kernel[:, :, 0, 0]);
```

+++ {"id": "dITPaPdh_cMI"}

And we'll make a simple synthetic image:

```{code-cell} ipython3
:id: cpbGsIGa_Qyx
:outputId: d7c5d21f-c3a0-42e9-c9bc-3da1a508c0e7

# NHWC layout
img = jnp.zeros((1, 200, 198, 3), dtype=jnp.float32)
for k in range(3):
  x = 30 + 60*k
  y = 20 + 60*k
  img = img.at[0, x:x+10, y:y+10, k].set(1.0)

print("Original Image:")
plt.imshow(img[0]);
```

+++ {"id": "_m90y74OWorG"}

### lax.conv and lax.conv_with_general_padding

+++ {"id": "Pv9_QPDnWssM"}

These are the simple convenience functions for convolutions

️⚠️ The convenience `lax.conv`, `lax.conv_with_general_padding` helper function assume __NCHW__ images and __OIHW__ kernels.

```{code-cell} ipython3
:id: kppxbxpZW0nb
:outputId: 9fc5494c-b443-4e74-fe48-fac09e12378c

from jax import lax
out = lax.conv(jnp.transpose(img,[0,3,1,2]),    # lhs = NCHW image tensor
               jnp.transpose(kernel,[3,2,0,1]), # rhs = OIHW conv kernel tensor
               (1, 1),  # window strides
               'SAME') # padding mode
print("out shape: ", out.shape)
print("First output channel:")
plt.figure(figsize=(10,10))
plt.imshow(np.array(out)[0,0,:,:]);
```

```{code-cell} ipython3
:id: aonr1tWvYCW9
:outputId: 3d44d494-9620-4736-e331-c9569a4888cd

out = lax.conv_with_general_padding(
  jnp.transpose(img,[0,3,1,2]),    # lhs = NCHW image tensor
  jnp.transpose(kernel,[2,3,0,1]), # rhs = IOHW conv kernel tensor
  (1, 1),  # window strides
  ((2,2),(2,2)), # general padding 2x2
  (1,1),  # lhs/image dilation
  (1,1))  # rhs/kernel dilation
print("out shape: ", out.shape)
print("First output channel:")
plt.figure(figsize=(10,10))
plt.imshow(np.array(out)[0,0,:,:]);
```

+++ {"id": "lyOwGRez_ycJ"}

### Dimension Numbers define dimensional layout for conv_general_dilated

The important argument is the 3-tuple of axis layout arguments:
(Input Layout, Kernel Layout, Output Layout)
 - __N__ - batch dimension
 - __H__ - spatial height
 - __W__ - spatial width
 - __C__ - channel dimension
 - __I__ - kernel _input_ channel dimension
 - __O__ - kernel _output_ channel dimension

⚠️ To demonstrate the flexibility of dimension numbers we choose a __NHWC__ image and __HWIO__ kernel convention for `lax.conv_general_dilated` below.

```{code-cell} ipython3
:id: oXKebfCb_i2B
:outputId: d5a569b3-febc-4832-f725-1d5e8fd31b9b

dn = lax.conv_dimension_numbers(img.shape,     # only ndim matters, not shape
                                kernel.shape,  # only ndim matters, not shape 
                                ('NHWC', 'HWIO', 'NHWC'))  # the important bit
print(dn)
```

+++ {"id": "elZys_HzFVG6"}

#### SAME padding, no stride, no dilation

```{code-cell} ipython3
:id: rgb2T15aFVG6
:outputId: 9b33cdb0-6959-4c88-832a-b92c4e42ae72

out = lax.conv_general_dilated(img,    # lhs = image tensor
                               kernel, # rhs = conv kernel tensor
                               (1,1),  # window strides
                               'SAME', # padding mode
                               (1,1),  # lhs/image dilation
                               (1,1),  # rhs/kernel dilation
                               dn)     # dimension_numbers = lhs, rhs, out dimension permutation
print("out shape: ", out.shape)
print("First output channel:")
plt.figure(figsize=(10,10))
plt.imshow(np.array(out)[0,:,:,0]);
```

+++ {"id": "E4i3TI5JFVG9"}

#### VALID padding, no stride, no dilation

```{code-cell} ipython3
:id: 1HQwudKVFVG-
:outputId: be9d6b26-8e3e-44d9-dbd2-df2f6bbf98c2

out = lax.conv_general_dilated(img,     # lhs = image tensor
                               kernel,  # rhs = conv kernel tensor
                               (1,1),   # window strides
                               'VALID', # padding mode
                               (1,1),   # lhs/image dilation
                               (1,1),   # rhs/kernel dilation
                               dn)      # dimension_numbers = lhs, rhs, out dimension permutation
print("out shape: ", out.shape, "DIFFERENT from above!")
print("First output channel:")
plt.figure(figsize=(10,10))
plt.imshow(np.array(out)[0,:,:,0]);
```

+++ {"id": "VYKZdqLIFVHB"}

#### SAME padding, 2,2 stride, no dilation

```{code-cell} ipython3
:id: mKq2-zmmFVHC
:outputId: 14cc0114-e230-4555-a682-23e00b534863

out = lax.conv_general_dilated(img,    # lhs = image tensor
                               kernel, # rhs = conv kernel tensor
                               (2,2),  # window strides
                               'SAME', # padding mode
                               (1,1),  # lhs/image dilation
                               (1,1),  # rhs/kernel dilation
                               dn)     # dimension_numbers = lhs, rhs, out dimension permutation
print("out shape: ", out.shape, " <-- half the size of above")
plt.figure(figsize=(10,10))
print("First output channel:")
plt.imshow(np.array(out)[0,:,:,0]);
```

+++ {"id": "gPxttaiaFVHE"}

#### VALID padding, no stride, rhs kernel dilation ~ Atrous convolution (excessive to illustrate)

```{code-cell} ipython3
:id: _pGr0x6qFVHF
:outputId: 9edbccb6-d976-4b55-e0b7-e6f3b743e476

out = lax.conv_general_dilated(img,     # lhs = image tensor
                               kernel,  # rhs = conv kernel tensor
                               (1,1),   # window strides
                               'VALID', # padding mode
                               (1,1),   # lhs/image dilation
                               (12,12), # rhs/kernel dilation
                               dn)      # dimension_numbers = lhs, rhs, out dimension permutation
print("out shape: ", out.shape)
plt.figure(figsize=(10,10))
print("First output channel:")
plt.imshow(np.array(out)[0,:,:,0]);
```

+++ {"id": "v-RhEeUfFVHI"}

#### VALID padding, no stride, lhs=input dilation  ~ Transposed Convolution

```{code-cell} ipython3
:id: B9Ail8ppFVHJ
:outputId: 7aa19474-566f-4419-bfae-8286dd026c1c

out = lax.conv_general_dilated(img,               # lhs = image tensor
                               kernel,            # rhs = conv kernel tensor
                               (1,1),             # window strides
                               ((0, 0), (0, 0)),  # padding mode
                               (2,2),             # lhs/image dilation
                               (1,1),             # rhs/kernel dilation
                               dn)                # dimension_numbers = lhs, rhs, out dimension permutation
print("out shape: ", out.shape, "<-- larger than original!")
plt.figure(figsize=(10,10))
print("First output channel:")
plt.imshow(np.array(out)[0,:,:,0]);
```

+++ {"id": "A-9OagtrVDyV"}

We can use the last to, for instance, implement _transposed convolutions_:

```{code-cell} ipython3
:id: 5EYIj77-NdHE
:outputId: f45b16f7-cc6e-4593-8aca-36b4152c3dfa

# The following is equivalent to tensorflow:
# N,H,W,C = img.shape
# out = tf.nn.conv2d_transpose(img, kernel, (N,2*H,2*W,C), (1,2,2,1))

# transposed conv = 180deg kernel rotation plus LHS dilation
# rotate kernel 180deg:
kernel_rot = jnp.rot90(jnp.rot90(kernel, axes=(0,1)), axes=(0,1))
# need a custom output padding:
padding = ((2, 1), (2, 1))
out = lax.conv_general_dilated(img,     # lhs = image tensor
                               kernel_rot,  # rhs = conv kernel tensor
                               (1,1),   # window strides
                               padding, # padding mode
                               (2,2),   # lhs/image dilation
                               (1,1),   # rhs/kernel dilation
                               dn)      # dimension_numbers = lhs, rhs, out dimension permutation
print("out shape: ", out.shape, "<-- transposed_conv")
plt.figure(figsize=(10,10))
print("First output channel:")
plt.imshow(np.array(out)[0,:,:,0]);
```

+++ {"id": "v8HsE-NCmUxx"}

### 1D Convolutions

+++ {"id": "WeP0rw0tm7HK"}

You aren't limited to 2D convolutions, a simple 1D demo is below:

```{code-cell} ipython3
:id: jJ-jcAn3cig-
:outputId: 67c46ace-6adc-4c47-c1c7-1f185be5fd4b

# 1D kernel - WIO layout
kernel = jnp.array([[[1, 0, -1], [-1,  0,  1]], 
                    [[1, 1,  1], [-1, -1, -1]]], 
                    dtype=jnp.float32).transpose([2,1,0])
# 1D data - NWC layout
data = np.zeros((1, 200, 2), dtype=jnp.float32)
for i in range(2):
  for k in range(2):
      x = 35*i + 30 + 60*k
      data[0, x:x+30, k] = 1.0

print("in shapes:", data.shape, kernel.shape)

plt.figure(figsize=(10,5))
plt.plot(data[0]);
dn = lax.conv_dimension_numbers(data.shape, kernel.shape,
                                ('NWC', 'WIO', 'NWC'))
print(dn)

out = lax.conv_general_dilated(data,   # lhs = image tensor
                               kernel, # rhs = conv kernel tensor
                               (1,),   # window strides
                               'SAME', # padding mode
                               (1,),   # lhs/image dilation
                               (1,),   # rhs/kernel dilation
                               dn)     # dimension_numbers = lhs, rhs, out dimension permutation
print("out shape: ", out.shape)
plt.figure(figsize=(10,5))
plt.plot(out[0]);
```

+++ {"id": "7XOgXqCTmaPa"}

### 3D Convolutions

```{code-cell} ipython3
:id: QNvSiq5-mcLd
:outputId: c99ec88c-6d5c-4acd-c8d3-331f026f5631

import matplotlib as mpl

# Random 3D kernel - HWDIO layout
kernel = jnp.array([
  [[0, 0,  0], [0,  1,  0], [0,  0,   0]],
  [[0, -1, 0], [-1, 0, -1], [0,  -1,  0]], 
  [[0, 0,  0], [0,  1,  0], [0,  0,   0]]], 
  dtype=jnp.float32)[:, :, :, jnp.newaxis, jnp.newaxis]

# 3D data - NHWDC layout
data = jnp.zeros((1, 30, 30, 30, 1), dtype=jnp.float32)
x, y, z = np.mgrid[0:1:30j, 0:1:30j, 0:1:30j]
data += (jnp.sin(2*x*jnp.pi)*jnp.cos(2*y*jnp.pi)*jnp.cos(2*z*jnp.pi))[None,:,:,:,None]

print("in shapes:", data.shape, kernel.shape)
dn = lax.conv_dimension_numbers(data.shape, kernel.shape,
                                ('NHWDC', 'HWDIO', 'NHWDC'))
print(dn)

out = lax.conv_general_dilated(data,    # lhs = image tensor
                               kernel,  # rhs = conv kernel tensor
                               (1,1,1), # window strides
                               'SAME',  # padding mode
                               (1,1,1), # lhs/image dilation
                               (1,1,1), # rhs/kernel dilation
                               dn)      # dimension_numbers
print("out shape: ", out.shape)

# Make some simple 3d density plots:
from mpl_toolkits.mplot3d import Axes3D
def make_alpha(cmap):
  my_cmap = cmap(jnp.arange(cmap.N))
  my_cmap[:,-1] = jnp.linspace(0, 1, cmap.N)**3
  return mpl.colors.ListedColormap(my_cmap)
my_cmap = make_alpha(plt.cm.viridis)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=data.ravel(), cmap=my_cmap)
ax.axis('off')
ax.set_title('input')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=out.ravel(), cmap=my_cmap)
ax.axis('off')
ax.set_title('3D conv output');
```
