import tensorflow as tf
from tensorflow import keras

SEQUENCE_AXIS = -2


def merge_padding_and_attention_mask(
    inputs,
    padding_mask,
    attention_mask,
):
    """Merge padding mask with users' customized mask.

    Args:
        inputs: the input sequence.
        padding_mask: the 1D padding mask, of shape
            [batch_size, sequence_length].
        attention_mask: the 2D customized mask, of shape
            [batch_size, sequence1_length, sequence2_length].

    Return:
        A merged 2D mask or None. If only `padding_mask` is provided, the
        returned mask is padding_mask with one additional axis.
    """
    mask = padding_mask
    if hasattr(inputs, "_keras_mask"):
        if mask is None:
            # If no padding mask is explicitly provided, we look for padding
            # mask from the input data.
            mask = inputs._keras_mask
        else:
            logging.warning(
                "You are explicitly setting `padding_mask` while the `inputs` "
                "have built-in mask, so the built-in mask is ignored."
            )
    if mask is not None:
        # Add an axis for broadcasting, the attention mask should be 2D
        # (not including the batch axis).
        mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
    if attention_mask is not None:
        attention_mask = tf.cast(attention_mask, dtype=tf.int32)
        if mask is None:
            return attention_mask
        else:
            return tf.minimum(
                mask[:, tf.newaxis, :],
                attention_mask,
            )
    return mask


def clone_initializer(initializer):
    """Clones an initializer to ensure a new seed.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    # If we get a string or dict, just return as we cannot and should not clone.
    if not isinstance(initializer, keras.initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)



##### @keras.utils.register_keras_serializable(package="keras_nlp")

class PositionEmbedding(keras.layers.Layer):
    """A layer which learns a position embedding for inputs sequences.

    This class assumes that in the input tensor, the last dimension corresponds
    to the features, and the dimension before the last corresponds to the
    sequence.

    This layer optionally accepts `tf.RaggedTensor`s as inputs to process
    batches of sequences of different lengths. The one ragged dimension must be
    the dimension that corresponds to the sequence, that is, the penultimate
    dimension.

    This layer does not supporting masking, but can be combined with a
    `keras.layers.Embedding` for padding mask support.

    Args:
        sequence_length: The maximum length of the dynamic sequence.
        initializer: The initializer to use for the embedding weights. Defaults
            to `"glorot_uniform"`.
        seq_axis: The axis of the input tensor where we add the embeddings.

    Examples:

    Called directly on input.
    >>> layer = keras_nlp.layers.PositionEmbedding(sequence_length=10)
    >>> layer(tf.zeros((8, 10, 16))).shape
    TensorShape([8, 10, 16])

    Combine with a token embedding.
    ```python
    seq_length = 50
    vocab_size = 5000
    embed_dim = 128
    inputs = keras.Input(shape=(seq_length,))
    token_embeddings = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim
    )(inputs)
    position_embeddings = keras_nlp.layers.PositionEmbedding(
        sequence_length=seq_length
    )(token_embeddings)
    outputs = token_embeddings + position_embeddings
    ```

    Reference:
     - [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
    """

    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError(
                "`sequence_length` must be an Integer, received `None`."
            )
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            "embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, tf.RaggedTensor):
            bounding_shape = inputs.bounding_shape()
            position_embeddings = self._trim_and_broadcast_position_embeddings(
                bounding_shape,
            )
            # then apply row lengths to recreate the same ragged shape as inputs
            return tf.RaggedTensor.from_tensor(
                position_embeddings,
                inputs.nested_row_lengths(),
            )
        else:
            return self._trim_and_broadcast_position_embeddings(
                tf.shape(inputs),
            )

    def _trim_and_broadcast_position_embeddings(self, shape):
        input_length = shape[SEQUENCE_AXIS]
        # trim to match the length of the input sequence, which might be less
        # than the sequence_length of the layer.
        position_embeddings = self.position_embeddings[:input_length, :]
        # then broadcast to add the missing dimensions to match "shape"
        return tf.broadcast_to(position_embeddings, shape)


class TransformerEncoder(keras.layers.Layer):
    """Transformer encoder.

    This class follows the architecture of the transformer encoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up an encoder.

    This layer will correctly compute an attention mask from an implicit
    Keras padding mask (for example, by passing `mask_zero=True` to a
    `keras.layers.Embedding` layer). See the Masking and Padding
    [guide](https://keras.io/guides/understanding_masking_and_padding/)
    for more details.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in the
            `keras.layers.MultiHeadAttention` layer.
        dropout: float, defaults to 0. the dropout value, shared by
            `keras.layers.MultiHeadAttention` and feedforward network.
        activation: string or `keras.activations`, defaults to "relu". the
            activation function of feedforward network.
        layer_norm_epsilon: float, defaults to 1e-5. The epsilon value in layer
            normalization components.
        kernel_initializer: string or `keras.initializers` initializer,
            defaults to "glorot_uniform". The kernel initializer for
            the dense and multiheaded attention layers.
        bias_initializer: string or `keras.initializers` initializer,
            defaults to "zeros". The bias initializer for
            the dense and multiheaded attention layers.
        normalize_first: bool. Defaults to False. If True, the inputs to the
            attention layer and the intermediate dense layer  are normalized
            (similar to GPT-2). If set to False, outputs of attention layer and
            intermediate dense layer are normalized (similar to BERT).
        name: string, defaults to None. The name of the layer.
        **kwargs: other keyword arguments.

    Examples:

    ```python
    # Create a single transformer encoder layer.
    encoder = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the encoder.
    input = keras.Input(shape=[10, 64])
    output = encoder(input)
    model = keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = tf.random.uniform(shape=[2, 10, 64])
    output = model(input_data)
    ```

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        name=None,
        **kwargs
    ):
        # Work around for model saving
        self._input_shape = kwargs.pop("build_input_shape", None)

        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self._built = False
        self.supports_masking = True

        if self._input_shape is not None:
            self._build(self._input_shape)

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        self._input_shape = input_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = input_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        key_dim = int(hidden_dim // self.num_heads)

        # Self attention layers.
        self._self_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._self_attention_layer._build_from_signature(
            query=input_shape,
            value=input_shape,
        )
        self._self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Feedforward layers.
        self._feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

    def call(self, inputs, padding_mask=None, attention_mask=None):
        """Forward pass of the TransformerEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, hidden_dim].
            padding_mask: a boolean Tensor. It indicates if the token should be
                masked because the token is introduced due to padding.
                `padding_mask` should have shape [batch_size, sequence_length].
                False means the certain certain is masked out.
            attention_mask: a boolean Tensor. Customized mask used to mask out
                certain tokens. `attention_mask` should have shape
                [batch_size, sequence_length, sequence_length].

        Returns:
            A Tensor of the same shape as the `inputs`.
        """

        if not self._built:
            self._build(inputs.shape)

        x = inputs  # Intermediate result.

        # Compute self attention mask.
        self_attention_mask = merge_padding_and_attention_mask(
            inputs, padding_mask, attention_mask
        )

        # Self attention block.
        residual = x
        if self.normalize_first:
            x = self._self_attention_layernorm(x)
        x = self._self_attention_layer(
            query=x,
            value=x,
            attention_mask=self_attention_mask,
        )
        x = self._self_attention_dropout(x)
        x = x + residual
        if not self.normalize_first:
            x = self._self_attention_layernorm(x)

        # Feedforward block.
        residual = x
        if self.normalize_first:
            x = self._feedforward_layernorm(x)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x)
        x = x + residual
        if not self.normalize_first:
            x = self._feedforward_layernorm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "normalize_first": self.normalize_first,
                "build_input_shape": self._input_shape,
            }
        )
        return config
