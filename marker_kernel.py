import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow_probability.python.math.psd_kernels as tfk

tfk = tfp.math.psd_kernels

class MarkerAwareKernel(tfp.math.psd_kernels.PositiveSemidefiniteKernel):
    def __init__(self, spatial_dims, marker_dims, amplitude=None, length_scale=None, name="MarkerAwareKernel"):
        super().__init__(feature_ndims=1, name=name)
        self.spatial_dims = spatial_dims
        self.marker_dims = marker_dims

        self.amplitude = tf.convert_to_tensor(amplitude if amplitude is not None else 1.0, dtype=tf.float32)
        self.length_scale = length_scale if length_scale is not None else 1.0

        self.spatial_kernel = tfk.ExponentiatedQuadratic(amplitude=1.0, length_scale=self.length_scale)
        self.marker_kernel = tfk.ExponentiatedQuadratic(amplitude=1.0, length_scale=self.length_scale)

        self.full_must_link_mask = None  # Full N x N mask
        self.active_mask = tf.Variable(tf.zeros([1, 1], dtype=tf.bool),
                                       trainable=False,
                                       shape=tf.TensorShape([None, None]))
          

    def set_mask_for_indices(self, indices1, indices2=None):
        """Set the current active mask as a subset of the full mask."""
        if self.full_must_link_mask is None:
            self.active_mask.assign(tf.zeros([0, 0], dtype=tf.bool))
            return
        '''indices1 = tf.cast(indices1, tf.int64)
        if indices2 is None:
            indices2 = indices1
        else:
            indices2 = tf.cast(indices2, tf.int64)'''
        indices1 = tf.cast(indices1, tf.int64)
        indices2 = tf.cast(indices2 if indices2 is not None else indices1, tf.int64)
        mask_subset = tf.gather(self.full_must_link_mask, indices1, axis=0)
        new_mask   = tf.gather(mask_subset, indices2, axis=1)
        if self.active_mask.shape != new_mask.shape:
            self.active_mask.assign(tf.zeros_like(new_mask))  # adjust shape
        self.active_mask.assign(new_mask)
        #self.active_mask = tf.gather(mask_subset, indices2, axis=1)

    def _apply(self, x1, x2, example_ndims=0):
        spatial1, marker1 = x1[..., :self.spatial_dims], x1[..., self.spatial_dims:]
        spatial2, marker2 = x2[..., :self.spatial_dims], x2[..., self.spatial_dims:]

        K_spatial = self.spatial_kernel.matrix(spatial1, spatial2)  # [M, N]
        K_marker  = self.marker_kernel.matrix(marker1, marker2)     # [M, N]

        # Expand dims for broadcasting with amplitude
        #K_spatial = tf.expand_dims(K_spatial, axis=0)  # [1, M, N]
        #K_marker = tf.expand_dims(K_marker, axis=0)    # [1, M, N]
        K_spatial = K_spatial[tf.newaxis, ...]  # [1, M, N]
        K_marker  = K_marker[tf.newaxis, ...]   # [1, M, N]
        # Dynamically check if the active mask matches the [M, N] shape
        mask_shape = tf.shape(self.active_mask)
        shape = tf.shape(K_marker)
        M = shape[1]
        N = shape[2]
        #_, M, N = tf.unstack(tf.shape(K_marker))
        #k_shape = tf.shape(K_marker)

        #mask_valid = tf.reduce_all(tf.equal(tf.shape(self.active_mask), tf.shape(K_marker)[1:3]))
        # Only apply mask if K_marker is [M, M] and active_mask matches that
        apply_mask = tf.logical_and(
            tf.equal(M, N),
            tf.reduce_all(tf.equal(tf.shape(self.active_mask), [M, N]))
            #tf.equal(k_shape[1], k_shape[2]),
            #tf.reduce_all(tf.equal(mask_shape, [k_shape[1], k_shape[2]]))
            #tf.reduce_all(tf.equal(mask_shape, k_shape[1:3]))
        )
        #apply_mask = tf.reduce_all(tf.equal(tf.shape(self.active_mask), tf.shape(K_marker)[1:])) & \
        #             tf.equal(tf.shape(K_marker)[1], tf.shape(K_marker)[2])
        tf.print("K_marker shape:", tf.shape(K_marker))
        tf.print("Mask shape used:", tf.shape(self.active_mask))
        K_marker = tf.cond(
            apply_mask,
            lambda: tf.where(tf.expand_dims(self.active_mask.read_value(), 0),
                            tf.ones_like(K_marker),
                            K_marker),
            lambda: K_marker
        )
        tf.print("K_marker shape:", tf.shape(K_marker))
        tf.print("Mask shape used:", tf.shape(self.active_mask))
        return tf.reshape(self.amplitude, [-1, 1, 1]) * (K_spatial + K_marker)
        #amplitude = tf.reshape(self.amplitude, [-1, 1, 1])  # [L, 1, 1]
        #return amplitude * (K_spatial + K_marker)
    
    def _apply_with_example_ndims(self, x1, x2, example_ndims):
        return self._apply(x1, x2, example_ndims)

    def _apply_diag(self, x):
        """Return just the diagonal of the kernel matrix: shape [L, N]."""
        spatial = x[..., :self.spatial_dims]
        marker = x[..., self.spatial_dims:]

        K_spatial_diag = self.spatial_kernel.apply(spatial, spatial)
        K_marker_diag = self.marker_kernel.apply(marker, marker)

        amplitude = tf.reshape(self.amplitude, [-1, 1])  # [L, 1]
        return amplitude * (K_spatial_diag + K_marker_diag)

    def _matrix(self, x1, x2):
        return self._apply(x1, x2)

    def _batch_shape(self):
        return tf.TensorShape([])

    def _event_shape(self):
        return tf.TensorShape([])
    
    def apply(self, x1, x2=None, example_ndims=0, name='apply'):
        """Override to route apply calls properly to _apply or _apply_diag."""
        if x2 is None:
            return self._apply_diag(x1)
        return self._apply(x1, x2, example_ndims)

    def apply_diag(self, x):
        """Used by TFP to compute diag(k(x, x)) correctly."""
        return self._apply_diag(x)
