(version-support-policy)=
# Python and NumPy version support policy

<!--* freshness: { reviewed: '2024-05-02' } *-->

For NumPy and SciPy version support, JAX follows the Python scientific community's
[SPEC 0](https://scientific-python.org/specs/spec-0000/).

For Python version support, we have heard from users that a 36-month support window can
be too short, for example due to the delays in propagation of new CPython releases
to Linux vendor releases. For this reason JAX supports Python versions for at least
nine months longer than SPEC-0 recommends.

This means we support at least:

* All minor Python releases in the 45 months prior to each JAX release. For example:

  * **Python 3.9** was released October 2020, and will be supported in new JAX releases at least until **July 2024**.
  * **Python 3.10** was released October 2021, and will be supported in new JAX releases at least until **July 2025**.
  * **Python 3.11** was released October 2022, and will be supported in new JAX releases at least until **July 2026**.

* All minor NumPy releases in the 24 months prior to each JAX release. For example:

  * **NumPy 1.22** was released December 2021, and will be supported in new JAX releases at least until **December 2023**.
  * **NumPy 1.23** was released June 2022, and will be supported in new JAX releases at least until **June 2024**.
  * **NumPy 1.24** was released December 2022, and will be supported in new JAX releases at least until **December 2024**.

* All minor SciPy releases in the 24 months prior to each JAX release, starting
  with SciPy version 1.9. For example:

  * **Scipy 1.9** was released July 2022, and will be supported in new JAX releases at least until **July 2024**.
  * **Scipy 1.10** was released January 2023, and will be supported in new JAX releases at least until **January 2025**.
  * **Scipy 1.11** was released June 2023, and will be supported in new JAX releases at least until **June 2025**.

JAX releases may support older versions of Python, NumPy, and SciPy than strictly required
by this policy, but support for older versions may be dropped at any time beyond the listed
dates.
