# README

### Lessons learned

1. If you see oscillating training loss, likely the descent steps are bouncing between two "valleys" - try using MomentumOptimizer.
2. Playing around with the step_size can help avoid non-covergence to nan
3. The std of the random_normal of the initial weights should be quite small (ex: `stddev=0.01`)
4. Also using RMSPropOptimizer really seems to help decrease the obj function (at least, better than the GradientDescent)