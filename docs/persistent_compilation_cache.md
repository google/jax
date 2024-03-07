# Persistent Compilation Cache

<!--* freshness: { reviewed: '2024-04-09' } *-->

JAX has an optional disk cache for compiled programs. If enabled, JAX will
store copies of compiled programs on disk, which can save recompilation time
when running the same or similar tasks repeatedly.

## Usage

The compilation cache is enabled when the
[cache-location](https://github.com/google/jax/blob/jax-v0.4.26/jax/_src/config.py#L1206)
is set. This should be done prior to the first compilation. Set the location as
follows:

```
import jax

# Make sure this is called before jax runs any operations!
jax.config.update("jax_compilation_cache_dir", "cache-location")
```

See the sections below for more detail on `cache-location`.

[`set_cache_dir()`](https://github.com/google/jax/blob/jax-v0.4.26/jax/experimental/compilation_cache/compilation_cache.py#L18)
is an alternate way of setting `cache-location`.

### Local filesystem

`cache-location` can be a directory on the local filesystem. For example:

```
import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
```

Note: the cache does not have an eviction mechanism implemented. If the
cache-location is a directory in the local filesystem, its size will continue
to grow unless files are manually deleted.

### Google Cloud

When running on Google Cloud, the compilation cache can be placed on a Google
Cloud Storage (GCS) bucket. We recommend the following configuration:

*  Create the bucket in the same region as where the workload will run.

*  Create the bucket in the same project as the workload’s VM(s). Ensure that
   permissions are set so that the VM(s) can write to the bucket.

*  There is no need for replication for smaller workloads. Larger workloads
   could benefit from replication.

*  Use “Standard” for the default storage class for the bucket.

*  Set the soft delete policy to its shortest: 7 days.

*  Set the object lifecycle to the expected duration of the workload run.
   For example, if the workload is expected to run for 10 days, set the object
   lifecycle to 10 days. That should cover restarts that occur during the entire
   run. Use `age` for the lifecycle condition and `Delete` for the action. See
   [Object Lifecycle Management](https://cloud.google.com/storage/docs/lifecycle)
   for details. If the object lifecycle is not set, the cache will continue to
   grow since there is no eviction mechanism implemented.

*  All encryption policies are supported.

Assuming that `gs://jax-cache` is the GCS bucket, set `cache-location` as
follows:

```
import jax

jax.config.update("jax_compilation_cache_dir", "gs://jax-cache")
```
