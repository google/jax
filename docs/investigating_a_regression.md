(investigating-a-regression)=
# Investigating a regression

<!--* freshness: { reviewed: '2023-11-15' } *-->

So you updated JAX and you hit a speed regression?
You have a little bit of time and are ready to investigate this?
Let's first make a JAX issue.
But if you can pinpoint the commit that triggered the regression, it will really help us.

This document explains how we identified the commit that caused a
[15% performance regression](https://github.com/google/jax/issues/17686).

## Steps

This can be done easily if the reproducer is quick enough. This is a brute
force method and not a bisection, but if the reproducer is quick enough, it
works well. This makes sure that you always test XLA and JAX commits
that are compatible. It also limits XLA recompilation.

Here is a suggested investigation strategy:
 1. You can do a brute force test of nightly containers between the 2 releases.
 2. Hourly recompilation while keeping XLA and JAX in sync.
 3. Final verification: maybe a manual check of a few commits (or a git bisect).

## Nightly investigation.

This can be done by using [JAX-Toolbox nightly
containers](https://github.com/NVIDIA/JAX-Toolbox).

- Some days, bugs prevent the container from being built, or there are temporary regressions. Just discard those days.
- So you should end up with a specific day or a few days where the regression happens.
- To automate this, you need 2 python scripts:
    - test_runner.sh: will start the containers and the test.
    - test.sh: will install missing dependencies and run the test

Here are real example scripts used for the issue: https://github.com/google/jax/issues/17686
- test_runner.sh:
```
  for m in 7 8 9; do
    for d in `seq -w 1 30`; do
      docker run -v $PWD:/dir --gpus=all ghcr.io/nvidia/jax:nightly-2023-0${m}-${d} /bin/bash /dir/test.sh &> OUT-0${m}-${d}
    done
  Done
```

- test.sh:
```
  pip install jmp pyvista numpy matplotlib Rtree trimesh jmp termcolor orbax
  git clone https://github.com/Autodesk/XLB
  cd XLB
  export PYTHONPATH=.
  export CUDA_VISIBLE_DEVICES=0 # only 1 GPU is needed

  python3 examples/performance/MLUPS3d.py 256 200
```

Then you can grep each output to see when the regression happens:
`grep MLUPS OUT*`. Here are the results we got:

```
OUT-07-06:MLUPS: 587.9240990200157
OUT-07-07:MLUPS: 587.8907972116419
OUT-07-08:MLUPS: 587.3186499464459
OUT-07-09:MLUPS: 587.3130127722537
OUT-07-10:MLUPS: 587.8526619429658
OUT-07-17:MLUPS: 570.1631097290182
OUT-07-18:MLUPS: 570.2819775617064
OUT-07-19:MLUPS: 570.1672213357352
OUT-07-20:MLUPS: 587.437153685251
OUT-07-21:MLUPS: 587.6702557143142
OUT-07-25:MLUPS: 577.3063618431178
OUT-07-26:MLUPS: 577.2362978080912
OUT-07-27:MLUPS: 577.2101850145785
OUT-07-28:MLUPS: 577.0716349809895
OUT-07-29:MLUPS: 577.4223280707176
OUT-07-30:MLUPS: 577.2255967221336
OUT-08-01:MLUPS: 577.277685388252
OUT-08-02:MLUPS: 577.0137874289354
OUT-08-03:MLUPS: 577.1333281553946
OUT-08-04:MLUPS: 577.305012020407
OUT-08-05:MLUPS: 577.2143988866626
OUT-08-06:MLUPS: 577.2409145495443
OUT-08-07:MLUPS: 577.2602819927345
OUT-08-08:MLUPS: 577.2823738293221
OUT-08-09:MLUPS: 577.3453199728248
OUT-08-11:MLUPS: 577.3161423260563
OUT-08-12:MLUPS: 577.1697775786824
OUT-08-13:MLUPS: 577.3049883393633
OUT-08-14:MLUPS: 576.9051978525331
OUT-08-15:MLUPS: 577.5331743016213
OUT-08-16:MLUPS: 577.5117505070573
OUT-08-18:MLUPS: 577.5930698237612
OUT-08-19:MLUPS: 577.3539885757353
OUT-08-20:MLUPS: 577.4190113959127
OUT-08-21:MLUPS: 577.300394253605
OUT-08-22:MLUPS: 577.4263792037783
OUT-08-23:MLUPS: 577.4087536357031
OUT-08-24:MLUPS: 577.1094728438082
OUT-08-25:  File "/XLB/examples/performance/MLUPS3d.py", line 5, in <module>
OUT-08-26:MLUPS: 537.0164618489928
OUT-08-27:MLUPS: 536.9545448661609
OUT-08-28:MLUPS: 536.2887650464874
OUT-08-29:MLUPS: 536.7178471720636
OUT-08-30:MLUPS: 536.6978912984252
OUT-09-01:MLUPS: 536.7030899164106
OUT-09-04:MLUPS: 536.5339818238837
OUT-09-05:MLUPS: 536.6507808565617
OUT-09-06:MLUPS: 536.7144494518315
OUT-09-08:MLUPS: 536.7376612408998
OUT-09-09:MLUPS: 536.7798324141778
OUT-09-10:MLUPS: 536.726157440174
OUT-09-11:MLUPS: 536.7446210750584
OUT-09-12:MLUPS: 536.6707332269023
OUT-09-13:MLUPS: 536.6777936517823
OUT-09-14:MLUPS: 536.7581523280307
OUT-09-15:MLUPS: 536.6156273667873
OUT-09-16:MLUPS: 536.7320935035265
OUT-09-17:MLUPS: 536.7104991444398
OUT-09-18:MLUPS: 536.7492269469092
OUT-09-19:MLUPS: 536.6760131792959
OUT-09-20:MLUPS: 536.7361260076634
```

This found that 8-24 was good, but 8-26 was bad. On 8-25 there was
another issue that prevented from getting results. So we need to
investigate hourly between 8-24 and 8-26. There was a smaller slowdown
earlier, lets ignore it for this example. It would be only another
hourly investigation between those dates.

## Hourly investigation.

This does a checkout of JAX and XLA at each hour between the 2 dates,
rebuilds everything and runs the test.  The scripts are structured
differently. We start the working container and keep it. Then inside
it, we only trigger incremental XLA builds except for the first
build. So it is much faster after the first iteration.

- test_runner2.sh:
```
  # Execute this script inside the container:
  # docker run -v $PWD:/dir --gpus=all ghcr.io/nvidia/jax:nightly-2023-08-24 /bin/bash
  cd /opt/xla-source
  git remote update
  cd /opt/jax-source
  git remote update
  pip install jmp pyvista numpy matplotlib Rtree trimesh jmp termcolor orbax
  cd /tmp
  git clone https://github.com/Autodesk/XLB
  cd XLB

  for d in `seq -w 24 26`; do
      for h in `seq -w 0 24`; do
          echo $m $d $h
          /bin/bash /dir/test2.sh Aug $d 2023 $h:00:00 &> OUT-08-${d}-$h
      done
  done
```

- test2.sh:
```
  echo "param: $@"
  cd /opt/xla-source
  git checkout `git rev-list -1 --before="$*" origin/main`
  git show -q
  cd /opt/jax-source
  git checkout `git rev-list -1 --before="$*" origin/main`
  git show -q

  rm /opt/jax-source/dist/jax*.whl
  build-jax.sh # The script is in the nightly container

  export PYTHONPATH=.
  export CUDA_VISIBLE_DEVICES=0 # only 1 GPU is needed

  python3 examples/performance/MLUPS3d.py 256 200
```

Now, you can execute the grep command on the new output files to see
which hours the issue appeared between.

## Final verification


With this, you need to check the JAX and XLA history between those hours. Maybe there are a few commits to test. You can use git bisect if you want to be fancy.

## Can this be improved?

Yes! If it was a crash regression, being able to do a bisect would be
useful. But it would be more complicated. If someone want to
contribute such instructions, please submit a PR ;)

For speed regressions, a bisect can hide some information. We wouldn't
see as easily that there were two regressions here.
