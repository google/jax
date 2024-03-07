# Contributing to JAX

<!--* freshness: { reviewed: '2023-11-16' } *-->

Everyone can contribute to JAX, and we value everyone's contributions. There are several
ways to contribute, including:

- Answering questions on JAX's [discussions page](https://github.com/google/jax/discussions)
- Improving or expanding JAX's [documentation](http://jax.readthedocs.io/)
- Contributing to JAX's [code-base](http://github.com/google/jax/)
- Contributing in any of the above ways to the broader ecosystem of [libraries built on JAX](https://github.com/google/jax#neural-network-libraries)

The JAX project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Ways to contribute

We welcome pull requests, in particular for those issues marked with
[contributions welcome](https://github.com/google/jax/issues?q=is%3Aopen+is%3Aissue+label%3A%22contributions+welcome%22) or
[good first issue](https://github.com/google/jax/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

For other proposals, we ask that you first open a GitHub
[Issue](https://github.com/google/jax/issues/new/choose) or
[Discussion](https://github.com/google/jax/discussions)
to seek feedback on your planned contribution.

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Sign the [Google Contributor License Agreement (CLA)](https://cla.developers.google.com/).
   For more information, see the Pull Request Checklist below.

2. Fork the JAX repository by clicking the **Fork** button on the
   [repository page](http://www.github.com/google/jax). This creates
   a copy of the JAX repository in your own account.

3. Install Python >= 3.10 locally in order to run tests.

4. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/jax
   cd jax
   pip install -r build/test-requirements.txt  # Installs all testing requirements.
   pip install -e ".[cpu]"  # Installs JAX from the current directory in editable mode.
   ```

5. Add the JAX repo as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git remote add upstream https://www.github.com/google/jax
   ```

6. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

   And implement your changes using your favorite editor (we recommend
   [Visual Studio Code](https://code.visualstudio.com/)).

7. Make sure your code passes JAX's lint and type checks, by running the following from
   the top of the repository:

   ```bash
   pip install pre-commit
   pre-commit run --all
   ```

   See {ref}`linting-and-type-checking` for more details.

8. Make sure the tests pass by running the following command from the top of
   the repository:

   ```bash
   pytest -n auto tests/
   ```

   JAX's test suite is quite large, so if you know the specific test file that covers your
   changes, you can limit the tests to that; for example:

   ```bash
   pytest -n auto tests/lax_scipy_test.py
   ```

   You can narrow the tests further by using the `pytest -k` flag to match particular test
   names:

   ```bash
   pytest -n auto tests/lax_scipy_test.py -k testLogSumExp
   ```

   JAX also offers more fine-grained control over which particular tests are run;
   see {ref}`running-tests` for more information.

9. Once you are satisfied with your change, create a commit as follows (
   [how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -m "Your commit message"
   ```

   Then sync your code with the main repo:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Finally, push your commit on your development branch and create a remote
   branch in your fork that you can use to create a pull request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```

   Please ensure your contribution is a single commit (see {ref}`single-change-commits`)

10. Create a pull request from the JAX repository and send it for review.
    Check the {ref}`pr-checklist` for considerations when preparing your PR, and
    consult [GitHub Help](https://help.github.com/articles/about-pull-requests/)
    if you need more information on using pull requests.

(pr-checklist)=

## JAX pull request checklist

As you prepare a JAX pull request, here are a few things to keep in mind:

### Google contributor license agreement

Contributions to this project must be accompanied by a Google Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again. If you're not certain whether you've signed a CLA, you can open your PR
and our friendly CI bot will check for you.

(single-change-commits)=

### Single-change commits and pull requests

A git commit ought to be a self-contained, single change with a descriptive
message. This helps with review and with identifying or reverting changes if
issues are uncovered later on.

**Pull requests typically comprise a single git commit.** (In some cases, for
instance for large refactors or internal rewrites, they may contain several.)
In preparing a pull request for review, you may need to squash together
multiple commits. We ask that you do this prior to sending the PR for review if
possible. The `git rebase -i` command might be useful to this end.

(linting-and-type-checking)=

### Linting and Type-checking

JAX uses [mypy](https://mypy.readthedocs.io/) and
[ruff](https://docs.astral.sh/ruff/) to statically test code quality; the
easiest way to run these checks locally is via the
[pre-commit](https://pre-commit.com/) framework:

```bash
pip install pre-commit
pre-commit run --all
```

If your pull request touches documentation notebooks, this will also run some checks
on those (See {ref}`update-notebooks` for more details).

### Full GitHub test suite

Your PR will automatically be run through a full test suite on GitHub CI, which
covers a range of Python versions, dependency versions, and configuration options.
It's normal for these tests to turn up failures that you didn't catch locally; to
fix the issues you can push new commits to your branch.

### Restricted test suite

Once your PR has been reviewed, a JAX maintainer will mark it as `Pull Ready`. This
will trigger a larger set of tests, including tests on GPU and TPU backends that are
not available via standard GitHub CI. Detailed results of these tests are not publicly
viewable, but the JAX maintainer assigned to your PR will communicate with you regarding
any failures these might uncover; it's not uncommon, for example, that numerical tests
need different tolerances on TPU than on CPU.
