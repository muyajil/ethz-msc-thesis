class HGRU4RECSummaries(object):
    train_summaries = []
    metrics_summaries = []
    eval_summaries = []


class HGRU4RECFeatures(object):
    user_embeddings = None
    session_embeddings = None
    product_embedding_ids = None
    session_changed = None


class HGRU4RecLosses(object):
    cross_entropy_loss = None
    top1_loss = None


class HGRU4RecMetrics(object):
    mrr_at_10 = None
    precision_at_10 = None
    recall_at_10 = None

    mrr_update_op = None
    precision_update_op = None
    recall_update_op = None


class HGRU4RecOps(object):
    features = HGRU4RECFeatures()
    labels = None

    metrics = HGRU4RecMetrics()
    summaries = HGRU4RECSummaries()
    global_step = None

    user_embeddings = None
    session_embeddings = None

    ranked_predictions = None

    losses = HGRU4RecLosses()
    optimizer = None
    grads_and_vars = None

    logits = None