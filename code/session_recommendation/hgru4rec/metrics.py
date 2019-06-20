import tensorflow as tf


def mrr_at_k(labels, logits, k, name=None):

    with tf.name_scope(name, 'mrr_metric', [logits, labels]) as scope:

        _, pred_embedding_ids = tf.nn.top_k(logits, k)
        labels = tf.broadcast_to(
            tf.expand_dims(labels, 1),
            tf.shape(pred_embedding_ids))

        ranked_indices = tf.where(
            tf.equal(tf.cast(pred_embedding_ids, tf.int64), labels))[:, 0]

        inverse_rank = 1/(ranked_indices + 1)

        num_paddings = tf.shape(labels)[0] - tf.shape(inverse_rank)[0]
        padded_inverse_rank = tf.pad(inverse_rank, [[num_paddings, 0]])

        m_rr, update_mrr_op = tf.metrics.mean(
            padded_inverse_rank,
            name=name)

        return m_rr, update_mrr_op


def top1_loss(logits):

    logits = tf.transpose(logits)
    total_loss = tf.reduce_mean(tf.sigmoid(
        logits-tf.diag_part(logits))+tf.sigmoid(logits**2), axis=0)
    answer_loss = tf.sigmoid(tf.diag_part(logits)**2) / \
        tf.cast(tf.shape(logits)[0], tf.float32)
    return total_loss - answer_loss