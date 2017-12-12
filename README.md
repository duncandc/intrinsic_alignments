# Intrinsic Alignments

A project to measure and model the intrinsic alignments of galaxy shapes in observations and simulations.

Possible avenues of research:

*  Extend the [Hirata+07](https://ui.adsabs.harvard.edu/#abs/2007MNRAS.381.1197H/abstract) measurements down to small scales, and remake position-shape cross-correlation measurements on galaxy samples split in a more logical way: luminosity bins, color-splits (and/or splits on B/T). Fit these data with an empirical model where IA correlations vary in a smooth parametric fashion with luminosity and color, using halo shape and/or tidal field orientation as the correlating vector. Would need additional empirical modeling ingredients for luminosity, color, and ellipticity distributions. 
*  Pick a fiducial IA model, generate mock shape-pos signal, fit the mock signal with the same model, including information from successively smaller scales, showing how much information is gained by including 1-halo term information (H+07 stop at R~1Mpc).
* Repeat above exercise, but using shape-shape signal, which I suspect contains significantly more information relative to shape-pos. Forecast whether LSST statistics will be sufficient to beat down shape noise enough to exploit this information. 
* Directly measure the one-point IA distributions using recently completed adiabatic hydro (using alignment between shape of gas vs. shape of halo or tidal field vector). This misses feedback physics, but just like with N-body sims there are zero free parameters, so the result is permanent, and is the definitive "baseline case" that feedback modulates. Can just use toy model CAM to paint luminosity and color. The most interesting thing to study is probably redshift evolution, which is highly uncertain in kitchen-sink hydro, but has no uncertainty in adiabatic case. 
* Treat MBII as fiducial model, fit the one-point IA distributions and try to recover the two-point functions, and/or just try to fit the one- and two-points together. Since this is basically what Francois has already done using ML algorithm, this is probably more interesting if done in conjunction with the adiabatic study.

Tools that will be needed:

* galaxy-shape TPCF
* shape-shape TPCF 
* develop model for galaxy type and alignment strength

Simulations:

* MassiveBlack II
* Adiabatic Sim

Observations:

* SDSS main sample
* BOSS LRG
* eBOSS ?