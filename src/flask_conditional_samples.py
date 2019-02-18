#!/usr/bin/env python3

import json
import os

import fire
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

#import encoder
#import model
#import sample


app = Flask(__name__)

GENERATOR_MODEL = (None, None, None)

@app.route('/', methods=('GET', 'POST'))
def hello():
    global GENERATOR_MODEL
    (enc, sess, output) = GENERATOR_MODEL
    prompt = ""
    text = ""
    if request.method == 'POST':
        prompt = request.form['prompt']
        context_tokens = enc.encode(prompt)
        out = sess.run(output, feed_dict={context: [context_tokens]})
        text = enc.decode(out[0])
    return render_template("page.jinja", prompt=prompt, tex=text)


def init_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=None,
    length=None,
    temperature=1,
    top_k=0,
):
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    sess = tf.Session(graph=tf.Graph())
    context = tf.placeholder(tf.int32, [batch_size, None])
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    saver.restore(sess, ckpt)

    return (enc, sess, output)


if __name__ == '__main__':
    GENERATOR_MODEL = init_model()
    pass
