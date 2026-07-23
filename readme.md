This repo shows the difference between DDPMs and VAEs where both are evaluated using FID and IS scores.


FID score is a metric is calculated by passing a set of real images and fake images into an inception v3 classifier, the final 2048 feature vector before the classification  layer for both sets are extracted and a simimlarity between each set is computed, the used similarity metric used is called FID (Fréchet Inception Distance). FID Treats each set of feature vectors as samples from a multivariate Gaussian and it estimates a mean vector and covariance matrix for the real features and separately for the fake features. the end goal of FID is to show how the feature distribution sits closer to the real one therefore, a lower FID score implies that both distributions are more similar, or the deviation between both distribution is lower.

whereas IS computes how clear a generated image is. and whether it really represents a meaningful object.
it's computed by observing the output layer of passing the generated image through a classifier [Inception v3 in our case] and the more confident Inception-v3 is about it's predictions, the more clear and meaningfull the generated images are, otherwise, if the model keeps producing even/kind-flat probabilty distributions, it'd indicate that the generated images are not really clear.




now, VAEs operate by ...
on the other hand, DDPM operates by ...




