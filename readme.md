This repo shows the difference between DDPMs and VAEs where both are evaluated using FID and IS scores.


FID score is a metric is calculated by passing a set of real images and fake images into an inception v3 classifier, the final 2048 feature vector before the classification  layer for both sets are extracted and a simimlarity between each set is computed, the used similarity metric used is called FID (Fréchet Inception Distance). FID Treats each set of feature vectors as samples from a multivariate Gaussian and it estimates a mean vector and covariance matrix for the real features and separately for the fake features. the end goal of FID is to show how the feature distribution sits closer to the real one therefore, a lower FID score implies that both distributions are more similar, or the deviation between both distribution is lower.

whereas IS computes how clear a generated image is. and whether it really represents a meaningful object.
it's computed by observing the output layer of passing the generated image through a classifier [Inception v3 in our case] and the more confident Inception-v3 is about it's predictions, the more clear and meaningfull the generated images are, otherwise, if the model keeps producing even/kind-flat probabilty distributions, it'd indicate that the generated images are not really clear.




now, VAEs is trained as an encoder-decoder that takes an input image, passes it through an encoder, and the encoder outputs a mean (mue) and log-variance vectors (log(sigma^2)), than a bunch of random noise is generated, and get normalized using the output mean and log-variance vectors, the noise then gets passed through the decoder. the decoder outputs an image accordingly.
this paradigm ensures a bunch of things, most importantly is that unlike ordinary auto-encoders, you can sample from anywhere in your latent space, all classes are close to each other, the latent space doesnt have empty spaces, so pretty much any sampling will result in meaningful images.

reconstruction loss alone is not enough for training a VAE, because when trained using rec loss only, they tend to output meaningless blurry images.
so the reconstruction loss is often combined with  perceptual and adversarial losses to for the model to output less blurry and more meaningfull images.


on the other hand, DDPM operates by also training an encoder decodrer to iteratively modify input noise to start forming a meaningful picture.
so, during inference, it takes pure noise, and with each pass, the model outputs the the direction of which the noise should go to in order to get closer to a meaningful picutre.
the reason of the sucess behind this paradigm is because going instantly from pure noise to an output image as in VAEs is a very hard process. the distribution of the gaussian noise is too far away from the real-images manifold.
when this process is done iteratively, we start shifting step by step from the gaussian noise to the manifold.
so instead of going from distribution A to the very far distribution B in one take, the diffusion process allows the model to take better steps toward distribution B.


the following table shows the overall FID and IS scores of each models 




