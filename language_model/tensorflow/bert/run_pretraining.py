"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import absl
import collections
import modeling
import optimization
import tensorflow.compat.v1 as tf
tf.disable_resource_variables()
# from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
# from tensorflow.contrib import data as contrib_data
# from tensorflow.contrib import tpu as contrib_tpu
import distribution_utils

from tensorflow.python import debug as tf_debug

flags = absl.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("poly_power", 1.0, "The power of poly decay.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_float("opt_weight_decay", 0.01, "Weight decay.")
flags.DEFINE_float("opt_beta_1", 0.9, "lamb beta1")
flags.DEFINE_float("opt_beta_2", 0.999, "lamb beta2")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
            "num_gpus", 0,
                "Use the GPU backend if this value is set to more than zero.")


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, optimizer, poly_power,
                     start_warmup_step, opt_beta_1, opt_beta_2, opt_weight_decay):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    tf.identity(input_ids, name="input_ids")
    tf.identity(input_mask, name="input_mask")
    tf.identity(segment_ids, name="segment_ids")
    tf.identity(masked_lm_positions, name="masked_lm_positions")
    tf.identity(masked_lm_ids, name="masked_lm_ids")
    tf.identity(masked_lm_weights, name="masked_lm_weights")
    tf.identity(next_sentence_labels, name="next_sentence_labels")

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      tvar_index = {var.name.replace(":0", ""): var for var in tvars}
      assignment_map = collections.OrderedDict([
        (name, tvar_index.get(name, value))
        for name, value in assignment_map.items()
      ])

      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer, poly_power, start_warmup_step, opt_beta_1, opt_beta_2, opt_weight_decay)

      if use_tpu:
        output_spec = contrib_tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      tensor_names = [tensor.name for tensor in tf.get_default_graph().get_operations() if tensor.name.endswith('embedding_lookup')]
      import sys
      for item in tensor_names:
        print("tensor_name:", item)
        myop = tf.get_default_graph().get_operation_by_name(str(item))
        tf.print(myop.values(), output_stream=sys.stdout)
        tf.compat.v1.summary.histogram("embedlookup", myop.values())

      

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      if use_tpu:
          output_spec = contrib_tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metrics=eval_metrics,
              scaffold_fn=scaffold_fn)
      else:
          output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metric_ops=metric_fn(
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                masked_lm_weights, next_sentence_example_loss,
                next_sentence_log_probs, next_sentence_labels))

    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  tf.identity(input_tensor, name="cls/predictions/mlm_input_tensor")

  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    tf.identity(input_tensor, name="mlm_input_tensor_gather")

    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))

      tf.identity(input_tensor, name="mlm_input_tensor_transformed")      

      input_tensor = modeling.layer_norm(input_tensor)

      tf.identity(input_tensor, name="mlm_input_tensor_layernorm")

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)

    tf.identity(logits, name="mlm_logits_matmul")

    logits = tf.nn.bias_add(logits, output_bias)

    tf.identity(logits, name="mlm_logits_bias")

    log_probs = tf.nn.log_softmax(logits, axis=-1)

    tf.identity(log_probs, name="mlm_log_probs")

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    tf.identity(one_hot_labels, name='mlm_one_hot_labels')

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

    tf.identity(per_example_loss, name="mlm_per_example_loss")

    numerator = tf.reduce_sum(label_weights * per_example_loss)
    
    tf.identity(numerator, name="mlm_numerator")
    
    denominator = tf.reduce_sum(label_weights) + 1e-5

    tf.identity(denominator, name="mlm_denominator")

    loss = numerator / denominator

    tf.identity(loss, name="mlm_loss")

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)

    tf.identity(logits, name="nsp_logits_matmul")

    logits = tf.nn.bias_add(logits, output_bias)
    
    tf.identity(logits, name="nsp_logits_bias")
    
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    
    tf.identity(log_probs, name="nsp_log_probs")
   
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    tf.identity(per_example_loss, name="nsp_per_example_loss")

    loss = tf.reduce_mean(per_example_loss)
    
    tf.identity(loss, name="nsp_loss")
    
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  tf.identity(sequence_tensor, name='gather/sequence_tensor')
  tf.identity(positions, name='gather/positions')

  flat_offsets = tf.reshape( tf.range( 0, batch_size, dtype=tf.int32 ) * seq_length, [-1, 1] )

  tf.identity(flat_offsets, name='gather/flat_offsets')

  flat_positions = tf.reshape( positions + flat_offsets, [-1] )
  
  tf.identity(flat_positions, name='gather/flat_positions')
  
  flat_sequence_tensor = tf.reshape( sequence_tensor, [batch_size * seq_length, width] )
  
  tf.identity(flat_sequence_tensor, name='gather/flat_sequence_tensor')
  
  output_tensor = tf.gather( flat_sequence_tensor, flat_positions )
  
  tf.identity(output_tensor, name='gather/output_tensor')
  
  return output_tensor


def input_fn_builder(input_files,
                     batch_size,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     input_context = None,
                     num_cpu_threads=4,
                     num_eval_steps=1):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params, input_context = None):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      if input_context:
        tf.logging.info(
            'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
            input_context.input_pipeline_id, input_context.num_input_pipelines))
        d = d.shard(input_context.num_input_pipelines,
                    input_context.input_pipeline_id)
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=1000)
      d = d.repeat()
    else:
      d = tf.data.TFRecordDataset(input_files)
      d = d.take(1)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      optimizer=FLAGS.optimizer,
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step,
      opt_beta_1=FLAGS.opt_beta_1,
      opt_beta_2=FLAGS.opt_beta_2,
      opt_weight_decay=FLAGS.opt_weight_decay)

  if FLAGS.use_tpu:
    is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
    run_config = contrib_tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        keep_checkpoint_max=50000,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_summary_steps=1, 
        tpu_config=contrib_tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = contrib_tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)
  else:
    # GPU path uses MirroredStrategy.

    # Creates session config. allow_soft_placement = True, is required for
    # multi-GPU and is not harmful for other modes.
    session_config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=8,
        allow_soft_placement=True)

    distribution_strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy='mirrored',
        num_gpus=FLAGS.num_gpus,
        all_reduce_alg='nccl',
        num_packs=0)

    dist_gpu_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        model_dir=FLAGS.output_dir,
        session_config=session_config,
        keep_checkpoint_max=50000,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    )

    hparams = {"batch_size": FLAGS.train_batch_size}
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=dist_gpu_config,
        model_dir=FLAGS.output_dir,
        params=hparams,
    )

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    batch_size = FLAGS.train_batch_size
    if FLAGS.num_gpus > 1:
      batch_size = distribution_utils.per_replica_batch_size(
            batch_size, FLAGS.num_gpus)
    hparams = {"batch_size": batch_size}
    train_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=batch_size,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True,
        input_context=None,
        num_cpu_threads=8)
    if FLAGS.use_tpu:
      estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
    else:
      estimator.train(input_fn=lambda input_context=None: train_input_fn(
          params=hparams, input_context=input_context), max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    batch_size = FLAGS.eval_batch_size
    if FLAGS.num_gpus > 1:
      batch_size = distribution_utils.per_replica_batch_size(
            batch_size, FLAGS.num_gpus)
    hparams = {"batch_size": batch_size}
    eval_input_fn = input_fn_builder(
        input_files=input_files,
        batch_size=batch_size,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False,
        input_context=None,
        num_cpu_threads=8,
        num_eval_steps=FLAGS.max_eval_steps)
    
    import numpy as np
    def logging_hook_formatter(tensor_dict):
      print("Weights")
      for item in tensor_dict.keys():
        print(item)
        
        with open(item.replace('/','_') + '.np', mode='wb') as f:
          np.save(f, tensor_dict[item])
        
      import pickle
      with open('tf_intermediate_tensors.pkl', mode='wb') as f:
        pickle.dump(tensor_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    embedding_tensors = [
        'input_ids',
        'word_embedding_output',
        'my_token_type_ids',
        'token_type_embeddings_output',
        'position_embeddings',
        'embedding_summation_output',
        'output'
    ]
    embedding_tensors = ['bert/embeddings/' + s for s in embedding_tensors]

    encoder_tensors_single = [
        'attention/self/from_tensor',
        'attention/self/query_output',
        'attention/self/key_output',
        'attention/self/value_output',
        'attention/self/attention_score_output',
        'attention/self/attention_score_scaled_output',
        'attention/self/attention_score_additive_output',
        'attention/self/attention_probs_output',
        'attention/self/context_layer',
        'attention/output/dense_output',
        'attention/output/attention_output',
        'intermediate/intermediate_output',
        'output/dense_layer_output',
        'output/layer_output'
    ]
    encoder_tensors = []
    for layer_idx in range(24):
        encoder_tensors += ['layer_' + str(layer_idx) + '/' + s for s in encoder_tensors_single]
    encoder_tensors = ['bert/encoder/' + s for s in encoder_tensors]

    pooler_tensors = [
        'bert/pooler/pooler_output'
    ]

    mlm_tensors = [
        'mlm_input_tensor',
        'mlm_input_tensor_gather',
        'transform/mlm_input_tensor_transformed',
        'transform/mlm_input_tensor_layernorm',
        'mlm_logits_matmul',
        'mlm_logits_bias',
        'mlm_log_probs',
        'mlm_one_hot_labels',
        'mlm_per_example_loss',
        'mlm_numerator',
        'mlm_denominator',
        'mlm_loss'
    ]
    mlm_tensors = ['cls/predictions/' + x for x in mlm_tensors]

    nsp_tensors = [
        'nsp_logits_matmul',
        'nsp_logits_bias',
        'nsp_log_probs',
        'nsp_per_example_loss',
        'nsp_loss'
    ]
    nsp_tensors = ['cls/seq_relationship/' + x for x in nsp_tensors]

    gather_tensors = [
        'sequence_tensor',
        'positions',
        'flat_offsets',
        'flat_positions',
        'flat_sequence_tensor',
        'output_tensor'
    ]
    gather_tensors = ['gather/' + x for x in gather_tensors]

    hooks_var_list = []
    hooks_var_list += embedding_tensors
    hooks_var_list += encoder_tensors
    hooks_var_list += pooler_tensors
    hooks_var_list += mlm_tensors
    hooks_var_list += nsp_tensors
    hooks_var_list += gather_tensors

    eval_hooks = [tf.estimator.LoggingTensorHook(hooks_var_list, every_n_iter=1, formatter=logging_hook_formatter)]
    
    do_eval_only_once = True
    while do_eval_only_once:
      if FLAGS.use_tpu:
        result = estimator.evaluate(
          input_fn=eval_input_fn, steps=FLAGS.max_eval_steps, monitors=eval_hooks)
      else:
        result = estimator.evaluate(
            input_fn=lambda input_context=None: eval_input_fn(
                params=hparams, input_context=input_context),
            steps=FLAGS.max_eval_steps, hooks=eval_hooks)

      output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
      with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
          tf.logging.info("  %s = %s", key, str(result[key]))
          writer.write("%s = %s\n" % (key, str(result[key])))

      do_eval_only_once = False


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  absl.app.run(main)
